import { Renderer, ProbeResult } from './renderer';
import {
  View, Point, cloneView, sameView, fitPrec, pan, zoomAt, pointAt, offsetIn, precFor, rescale,
  parseDecimal, formatDecimal, dblToFixed, fixedToDbl, LS_MIN, LS_MAX,
} from './view';
import { FORMULAS, JULIA_PRESETS, Formula, Spec, specOf, makeBigStep, sx, sy, paramOf, paramLc } from './formulas';
import { customBrute, customIters, makeCustom, customInfo } from './custom';
import { PALETTES, bakePalette } from './palettes';
import type { RefRequest, RefResult, BlaResult, BlaRequest, FailResult } from './refWorker';
import { R2, MAX_POWER, TEXW } from './refWorker';

const HOME: View = { cx: -(3n << 62n), cy: 0n, prec: 64, ls: 0.35 };
/** Starting view of the current fractal. */
function homeView(): View {
  let [re, im, ls] = formula.home;
  if (spec().julia) [re, im, ls] = [0, 0, 0.6];
  else if (formula.id === 'mandelbrot' && power !== 2) [re, im, ls] = [0, 0, 0.35];
  return { cx: dblToFixed(re, 0, 64), cy: dblToFixed(im, 0, 64), prec: 64, ls };
}
let maxIter = 6_000_000; // ceiling on iterations per pixel: what the memory limits leave room for (see applyLimits)

const canvas = document.getElementById('view') as HTMLCanvasElement;
const hud = document.getElementById('hud')!;
const help = document.getElementById('help')!;
const toast = document.getElementById('toast')!;
const powerInput = document.getElementById('power') as HTMLInputElement;
const vramInput = document.getElementById('vram') as HTMLInputElement;
const vramInfo = document.getElementById('vraminfo')!;
const menu = document.getElementById('menu')!;

let renderer: Renderer;
try {
  renderer = new Renderer(canvas);
} catch (err) {
  document.getElementById('fatal')!.textContent = String((err as Error).message || err);
  throw err;
}

let formula: Formula = FORMULAS[0];
let power = 2; // z -> z^power + c (formulas with hasPower)
let juliaOn = false; // Julia set of the current formula instead of its parameter plane
const jc = { re: -0.8, im: 0.156 }; // the fixed c of a Julia set
let jcSent = { re: NaN, im: NaN }; // the c the current reference orbit was asked for
let jcTag = 0;
let morph = false; // slowly rotate c around the origin
let paramView: View | null = null; // where to return to when leaving a Julia set opened with J

function spec(): Spec {
  return specOf(formula, power, juliaOn);
}
const view: View = cloneView(HOME);
let iterMult = 1;
let palette = 0;
let cycling = true;

// --- memory limits ---------------------------------------------------------------
// A reference orbit and its BLA tables live on the graphics card, so many bytes per
// iteration: how many iterations a pixel may have is a question of video memory.
// The user sets how much of it the program may take; the ceiling follows.

const MB = 1048576;
/** What the machine has, in MB; 0 where unknown (a browser is told neither, the desktop build asks Windows). */
const sys = { vram: 0, ram: 0 };
let vramUser = Number(localStorage.getItem('fractal.vram')) || 0; // the user's limit in MB; 0: automatic
let shrink = 1; // lowered whenever memory ran out this session
let capBy = '';
let lostNote = false;
if (localStorage.getItem('fractal.lost')) {
  // the card gave up last time: do not walk into the same wall again
  localStorage.removeItem('fractal.lost');
  localStorage.removeItem('fractal.vram');
  vramUser = 0;
  lostNote = true;
}

/** The most the limit may be set to: the card's memory less a reserve for the system and for peaks. */
function vramMax(): number {
  return sys.vram ? Math.floor(sys.vram - Math.max(1024, sys.vram * 0.15)) : 24576;
}

/** The limit when the user has set none: room for six million iterations of the plain Mandelbrot set. */
function vramDefault(): number {
  return Math.min(vramMax(), Math.ceil((renderer.baseBytes() + 6e6 * 32 * 1.5) / MB / 64) * 64);
}

function vramLimit(): number {
  return Math.max(128, Math.min(vramMax(), vramUser || vramDefault()));
}

/** What one iteration of a reference orbit costs: texels of the orbit texture, bytes on the card, bytes of main memory while it is made. */
function iterCost(sp: Spec): { texels: number; gpu: number; ram: number } {
  if (sp.kind === 'custom') {
    const info = customInfo(sp.custom);
    const texels = (info.second ? 2 : 1) * (info.hasU ? 2 : 1);
    return { texels, gpu: 16 * texels, ram: 48 * texels };
  }
  const segs = sp.julia ? 2 : 1;
  const folded = sp.kind === 'abs';
  const texels = segs * (folded ? 2 : 1);
  // main memory: the orbit as doubles and as texels, the table, and the copies in flight to the card
  // (the table's working space is small: it is built in blocks, see bla.ts)
  return { texels, gpu: 16 * texels + (folded ? 24 : 16) * segs, ram: (folded ? 180 : 100) * segs };
}

/** Work the iteration ceiling out of the limits and show it. */
function applyLimits(): void {
  const c = iterCost(spec());
  const limit = vramLimit();
  // 1.5: while an orbit is being replaced by a longer one, both are on the card
  const byVram = (limit * MB - renderer.baseBytes()) / (c.gpu * 1.5);
  // one texture may take a quarter of the card at most (a Direct3D rule), and no more rows than the GPU allows
  const byTex = Math.min(((sys.vram || limit) * MB * 0.25) / (16 * c.texels), (TEXW * renderer.maxTex) / c.texels - 16);
  // the orbit and its tables are made in main memory first
  const byRam = ((sys.ram || 16384) * MB * 0.25) / c.ram;
  const cap = Math.min(byVram, byTex, byRam);
  capBy = cap === byVram ? '' : cap === byRam ? 'оперативная память' : 'предел программы';
  maxIter = Math.round(Math.max(1e6, cap * shrink));
  renderer.maxN = maxIter;
  vramInput.max = String(vramMax());
  if (document.activeElement !== vramInput) vramInput.value = String(limit);
  const mln = (maxIter / 1e6).toFixed(maxIter < 1e7 ? 1 : 0).replace('.', ',');
  vramInfo.textContent =
    `до ${mln} млн итераций на точку` +
    (capBy && shrink === 1 ? ` (дальше держит ${capBy})` : '') +
    (shrink < 1 ? ' (снижено: памяти не хватило)' : '') +
    (sys.vram ? ` · карта ${sys.vram.toLocaleString('ru-RU')} МБ, можно до ${vramMax().toLocaleString('ru-RU')}` : ' · ставьте не больше памяти видеокарты минус 1 ГБ') +
    (vramUser ? '' : ' · авто');
}

/** Memory ran out (on the card or off it): come down, and say so. */
function outOfMemory(what: string): void {
  shrink *= 0.6;
  applyLimits();
  say(`Не хватило ${what} — потолок итераций снижен до ${(maxIter / 1e6).toFixed(1).replace('.', ',')} млн`);
}

function setVram(mb: number): void {
  vramUser = Number.isFinite(mb) && mb > 0 ? Math.max(128, Math.min(vramMax(), Math.round(mb))) : 0;
  if (vramUser) localStorage.setItem('fractal.vram', String(vramUser));
  else localStorage.removeItem('fractal.vram');
  shrink = 1;
  applyLimits();
  say(`Видеопамять: ${vramLimit().toLocaleString('ru-RU')} МБ${vramUser ? '' : ' (авто)'} · до ${(maxIter / 1e6).toFixed(1).replace('.', ',')} млн итераций`);
}

{
  const dm = (navigator as any).deviceMemory as number | undefined; // capped at 8 by browsers: "8 or more"
  if (dm) sys.ram = dm >= 8 ? 16384 : dm * 1024;
  const invoke = (window as any).__TAURI__?.core?.invoke as ((cmd: string) => Promise<{ gpus: { name: string; vram: number }[]; ram: number }>) | undefined;
  if (invoke) {
    invoke('system_memory')
      .then((m) => {
        // the card this picture is drawn on, by name; failing that, the biggest
        const mine = renderer.gpuName().toLowerCase();
        const gpu = m.gpus.find((g) => g.name && mine.includes(g.name.toLowerCase())) || [...m.gpus].sort((a, b) => b.vram - a.vram)[0];
        if (gpu && gpu.vram > 0) sys.vram = Math.round(gpu.vram / MB);
        if (m.ram > 0) sys.ram = Math.round(m.ram / MB);
        applyLimits();
      })
      .catch(() => {});
  }
}
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  localStorage.setItem('fractal.lost', '1');
  document.getElementById('fatal')!.textContent =
    'Видеокарта сбросила графику — скорее всего, не хватило видеопамяти. Лимит видеопамяти возвращён к автоматическому; перезагрузите страницу.';
});

// --- location in the URL -----------------------------------------------------

function loadHash(): void {
  const q = new URLSearchParams(location.hash.slice(1));
  const z = Number(q.get('z'));
  const f = FORMULAS.find((x) => x.id === q.get('f'));
  if (f) formula = f;
  power = formula.power ?? 2;
  if (q.has('d') && formula.hasPower) power = Math.max(2, Math.min(MAX_POWER, Number(q.get('d')) | 0));
  if (q.has('jr') && q.has('ji')) {
    jc.re = Number(q.get('jr')) || 0;
    jc.im = Number(q.get('ji')) || 0;
    juliaOn = formula.kind !== 'custom';
  }
  if (q.has('p')) palette = Math.max(0, Math.min(PALETTES.length - 1, Number(q.get('p')) | 0));
  if (q.has('i')) iterMult = Math.max(0.1, Math.min(64, Number(q.get('i')) || 1));
  if (!q.has('re') || !q.has('im') || !Number.isFinite(z)) {
    Object.assign(view, homeView());
    return;
  }
  view.ls = Math.min(LS_MAX, Math.max(LS_MIN, z));
  view.prec = precFor(view.ls);
  view.cx = parseDecimal(q.get('re')!, view.prec);
  view.cy = parseDecimal(q.get('im')!, view.prec);
}

function saveHash(): void {
  const digits = Math.ceil(Math.max(0, -view.ls) * 0.30103) + 6;
  const s =
    `re=${formatDecimal(view.cx, view.prec, digits)}&im=${formatDecimal(view.cy, view.prec, digits)}` +
    `&z=${view.ls.toFixed(4)}&p=${palette}` +
    (formula.id !== 'mandelbrot' ? `&f=${formula.id}` : '') +
    (formula.hasPower && power !== (formula.power ?? 2) ? `&d=${power}` : '') +
    (spec().julia ? `&jr=${jc.re}&ji=${jc.im}` : '') +
    (iterMult !== 1 ? `&i=${iterMult.toFixed(3)}` : '');
  history.replaceState(null, '', '#' + s);
}

// --- iterations and the reference orbit --------------------------------------

function itersFor(ls: number): number {
  const depth = Math.max(0, -ls);
  if (formula.kind === 'custom') return Math.min(maxIter, Math.round(iterMult * customIters(formula.custom ?? 0, depth)));
  return Math.min(maxIter, Math.round(iterMult * (1000 + 80 * Math.pow(depth, 1.5))));
}

const worker = new Worker(new URL('./refWorker.ts', import.meta.url), { type: 'module' });
// Threads for the BLA table of a long orbit: most of the machine's, leaving some for the
// picture itself. They are started here, when the first long orbit is asked for, and
// wired straight to the reference worker.
let tablePool = false;
function startTablePool(): void {
  if (tablePool) return;
  tablePool = true;
  const count = Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) * 0.6));
  if (count < 2) return;
  try {
    const ports: MessagePort[] = [];
    for (let i = 0; i < count; i++) {
      const w = new Worker(new URL('./blaWorker.ts', import.meta.url), { type: 'module' });
      const ch = new MessageChannel();
      w.postMessage({ port: ch.port1 }, [ch.port1]);
      ports.push(ch.port2);
    }
    worker.postMessage({ type: 'pool', ports }, ports);
  } catch {
    // no threads to be had: the reference worker builds its tables alone
  }
}
let refSeq = 0;
// the deepest point seen in the last picture: the best place for a reference orbit
let refHint: { p: Point; iters: number; inside: boolean; t: number } | null = null;
let hintTries = 0; // moves to a proven-inside point that did not yield a periodic orbit
let pendingSame = false; // the orbit being computed is a longer one of the current point
// Moving the reference to a "deeper" pixel is a bet: in chaotic dust the exact orbit of that
// pixel may escape sooner than the one it replaced. A lost bet is undone, and no more are
// placed until the view has moved on.
let pendingBet: { p: Point; len: number } | null = null; // the orbit a hinted one is replacing
let revertTo: Point | null = null;
let betsOffAt: View | null = null;
let lastHintRef = 0; // when a reference was last moved to a hint
let refPending = false;
let blaPending = false;
let refMs = 0;
let refOrbitMs = 0;

worker.onmessage = (ev: MessageEvent<RefResult | BlaResult | FailResult>) => {
  const m = ev.data;
  if (m.type === 'fail') {
    if (m.id !== refSeq) return;
    refPending = false;
    outOfMemory('оперативной памяти');
  } else if (m.type === 'ref') {
    if (m.id !== refSeq) return;
    refPending = false;
    blaPending = false;
    refMs = m.ms;
    refOrbitMs = m.msOrbit ?? m.ms;
    if (!renderer.setRef(m, pendingSame)) {
      outOfMemory('видеопамяти');
      return;
    }
    if (m.cap >= 1e9) hintTries = 0;
    if (pendingBet && m.escaped && m.lenA < pendingBet.len * 1.3) {
      betsOffAt = cloneView(view);
      if (m.lenA < pendingBet.len) revertTo = pendingBet.p;
    }
    pendingBet = null;
  } else {
    blaPending = false;
    if (renderer.ref && renderer.ref.id === m.id) renderer.setBla(m, true);
  }
};

function aspect(): number {
  return renderer.W / renderer.H;
}

function manageRef(): void {
  if (refPending) return;
  const sp = spec();
  const ref = renderer.ref;
  const N = itersFor(view.ls);
  const off = renderer.refOffset(view);
  let need = !ref || !off;
  let at: Point = view;
  let same = false;
  if (betsOffAt) {
    const [bx, by] = offsetIn(betsOffAt, view);
    if (Math.abs(view.ls - betsOffAt.ls) > 0.5 || Math.hypot(bx, by) > 0.5) betsOffAt = null;
    else refHint = null;
  }
  if (refHint) {
    const [hx, hy] = offsetIn(view, refHint.p);
    if (performance.now() - refHint.t < 1500 && Math.abs(hx) < aspect() && Math.abs(hy) < 1) at = refHint.p;
    else refHint = null;
  }
  let better = false;
  let longer = false;
  if (ref && off) {
    longer = ref.cap < 1e9 && renderer.needIter > ref.cap && ref.maxIter < maxIter;
    better = ref.cap < 1e9 && !!refHint && refHint.inside && performance.now() - lastHintRef > 700 && hintTries < 3;
    const far = Math.hypot(off.dm[0], off.dm[1]) / Math.pow(2, view.ls - Math.floor(view.ls));
    need =
      // a Julia parameter that moved: redo, but only once the last picture has landed
      (sp.julia && (jc.re !== jcSent.re || jc.im !== jcSent.im) && renderer.settled()) ||
      // an orbit that neither escapes nor settles has to be stored to the last iteration:
      // a point proven inside gives one that closes on itself after a few periods
      (better) ||
      // the picture wants more iterations than this orbit holds
      (longer) ||
      ref.prec < -view.ls + 56 ||
      !(far < 1.5) ||
      // the orbit escapes long before the picture is done: pixels outlive it and
      // lose the BLA speed-up, so move the reference somewhere deeper
      (ref.escaped && ref.lenA <= Math.max(N, renderer.stats.N) && !!refHint && refHint.iters > ref.lenA * 1.3 && performance.now() - lastHintRef > 700);
  }
  if (need && ref && off && longer && !better && sp.julia === ref.spec.julia && jc.re === jcSent.re && jc.im === jcSent.im) {
    const far = Math.hypot(off.dm[0], off.dm[1]) / Math.pow(2, view.ls - Math.floor(view.ls));
    if (far < 1.5 && ref.prec >= -view.ls + 56) {
      // only its length is wrong: a longer orbit of the same point lets the picture in progress carry on
      at = ref;
      same = true;
    }
  }
  if (revertTo && ref) {
    // the orbit just tried was worse than the one before: go back to that one
    need = true;
    at = revertTo;
    same = false;
  }
  if (need) {
    if (better && !same) hintTries++;
    pendingSame = same;
    pendingBet = at !== view && !same && !revertTo && ref && ref.escaped ? { p: { cx: ref.cx, cy: ref.cy, prec: ref.prec }, len: ref.lenA } : null;
    revertTo = null;
    if (at !== view && !same) {
      // a hint is used once; the next one must come from a picture made with the new orbit
      lastHintRef = performance.now();
      refHint = null;
    }
    const prec = precFor(view.ls - 64);
    if (jc.re !== jcSent.re || jc.im !== jcSent.im) {
      jcSent = { re: jc.re, im: jc.im };
      jcTag++;
    }
    const vx = rescale(at.cx, at.prec, prec);
    const vy = rescale(at.cy, at.prec, prec);
    const par = paramOf(sp, vx, vy, prec);
    const req: RefRequest = {
      type: 'ref',
      id: ++refSeq,
      cx: par[0],
      cy: par[1],
      vx,
      vy,
      prec,
      maxIter: Math.min(maxIter, Math.ceil(Math.max(3 * itersFor(view.ls), 2 * renderer.needIter))),
      lcmax: paramLc(sp, view.ls + Math.log2(1.5 + Math.hypot(aspect(), 1)) + 2, fixedToDbl(vx, prec, 0), fixedToDbl(vy, prec, 0)),
      spec: sp,
      tag: jcTag,
      jx: dblToFixed(jc.re, 0, prec),
      jy: dblToFixed(jc.im, 0, prec),
    };
    refPending = true;
    if (req.maxIter >= 200000) startTablePool();
    worker.postMessage(req);
    return;
  }
  // the BLA table is built for a bound on |dc|; rebuild when the view outgrows it
  if (ref && off && !blaPending && sp.kind !== 'custom' && !sp.julia && ref.M > 64 && (off.lc > ref.lcmax || off.lc < ref.lcmax - 48)) {
    const req: BlaRequest = { type: 'bla', id: ref.id, lcmax: off.lc + 2 };
    blaPending = true;
    worker.postMessage(req);
  }
}

// --- input ---------------------------------------------------------------------

const keys = new Set<string>();
const pointers = new Map<number, { x: number; y: number; button: number }>();
let rightDragged = false;
let pendingZoom = 0; // octaves still to apply (positive = in)
let anchor: [number, number] = [0, 0];
let panVel: [number, number] = [0, 0];
let dragMoved = 0;
let lastDrag = 0;
let pinchDist = 0;
const mouse = { x: 0, y: 0 };
window.addEventListener('pointermove', (e) => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});

const dive = {
  on: false,
  auto: true,
  speed: 0.7, // octaves per second
  target: null as Point | null,
  aim: null as Point | null, // glides towards the target; the camera follows the aim
  rate: 0, // current zoom rate, eased towards `speed`
  pace: 1, // share of that rate the renderer can keep up with (see updateInput)
  lost: 0,
};

/** Screen position -> offset from the view centre in half-heights. */
function toOffset(clientX: number, clientY: number): [number, number] {
  const r = canvas.getBoundingClientRect();
  const x = ((clientX - r.left) / r.width) * 2 - 1;
  const y = 1 - ((clientY - r.top) / r.height) * 2;
  return [x * aspect(), y];
}

function say(text: string): void {
  toast.textContent = text;
  toast.classList.add('show');
  clearTimeout((say as any).t);
  (say as any).t = setTimeout(() => toast.classList.remove('show'), 1400);
}

function setPalette(i: number): void {
  palette = ((i % PALETTES.length) + PALETTES.length) % PALETTES.length;
  renderer.setPalette(bakePalette(PALETTES[palette]));
}

/** Start the current fractal afresh: home view, new shader, nothing carried over. */
function applyFormula(keepView = false): void {
  if (!keepView) Object.assign(view, homeView());
  pendingZoom = 0;
  panVel = [0, 0];
  if (dive.on) setDive(false);
  refHint = null;
  pendingBet = null;
  revertTo = null;
  betsOffAt = null;
  refPending = false;
  refSeq++; // any orbit still being computed is for the old formula
  jcSent = { re: NaN, im: NaN };
  renderer.setFormula(spec());
  powerInput.value = String(power);
  powerInput.disabled = !formula.hasPower;
  applyLimits();
  dirtyHash = true;
  renderMenu();
}

function setFormula(f: Formula): void {
  formula = f;
  power = f.power ?? 2;
  juliaOn = false;
  paramView = null;
  applyFormula();
  say(f.name);
}

function setPower(p: number): void {
  if (!formula.hasPower) {
    say('У этого фрактала степень не меняется');
    return;
  }
  power = p > MAX_POWER ? 2 : p < 2 ? MAX_POWER : p;
  applyFormula();
  say(`Степень ${power}`);
}

function setJulia(re: number, im: number): void {
  jc.re = re;
  jc.im = im;
  if (formula.id === 'mandelbrot') formula = FORMULAS.find((f) => f.id === 'julia')!;
  else if (formula.id !== 'julia') juliaOn = true;
  applyFormula();
}

/** J: open the Julia set of the point under the cursor, or go back to where it was picked. */
function toggleJulia(): void {
  if (formula.kind === 'custom' || formula.param) {
    say('Для этого фрактала множества Жюлиа нет');
    return;
  }
  if (spec().julia) {
    if (formula.id === 'julia') formula = FORMULAS[0];
    juliaOn = false;
    const back = paramView;
    paramView = null;
    applyFormula();
    if (back) Object.assign(view, back);
    say(formula.name);
    return;
  }
  const [ox, oy] = toOffset(mouse.x, mouse.y);
  const p = pointAt(view, ox, oy);
  const back = cloneView(view);
  setJulia(fixedToDbl(p.cx, p.prec, 0), fixedToDbl(p.cy, p.prec, 0));
  paramView = back;
  say('Жюлиа для выбранной точки · J — назад');
}

// --- the fractal menu --------------------------------------------------------------

function renderMenu(): void {
  const sp = spec();
  let html = '<h1>Фракталы <small>Tab</small></h1><div class="list">';
  let group = '';
  for (const f of FORMULAS) {
    if (f.group !== group) {
      group = f.group;
      html += `<h2>${group}</h2>`;
    }
    html += `<button data-f="${f.id}" class="${f === formula ? 'on' : ''}"><b>${f.name}</b><i>${f.desc}</i></button>`;
  }
  if (sp.julia) {
    html += '<h2>Жюлиа: готовые параметры</h2><div class="chips">';
    JULIA_PRESETS.forEach((j, i) => {
      html += `<button data-j="${i}" class="${j.re === jc.re && j.im === jc.im ? 'on' : ''}">${j.name}</button>`;
    });
    html += '</div>';
  }
  html += '<p>J — множество Жюлиа для точки под курсором<br>Правая кнопка + движение — менять параметр c<br>K — плавное вращение c</p></div>';
  menu.innerHTML = html;
}

menu.addEventListener('click', (e) => {
  const t = (e.target as HTMLElement).closest('button, h1') as HTMLElement | null;
  if (!t) return;
  if (t.tagName === 'H1') {
    document.body.classList.toggle('menuless');
    return;
  }
  if (t.dataset.f) setFormula(FORMULAS.find((f) => f.id === t.dataset.f)!);
  else if (t.dataset.j) {
    const j = JULIA_PRESETS[Number(t.dataset.j)];
    setJulia(j.re, j.im);
    say(j.name);
  }
});

function setDive(on: boolean, target: Point | null = null, out = false): void {
  dive.on = on;
  dive.auto = !target && !out;
  dive.target = target;
  dive.aim = pointAt(view, 0, 0);
  dive.rate = 0;
  dive.pace = 1;
  dive.lost = 0;
  if (on) dive.speed = (out ? -1 : 1) * (Math.abs(dive.speed) || 0.7);
  say(on ? (out ? 'Отдаление' : target ? 'Погружение к точке' : 'Автопогружение') : 'Стоп');
}

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  let d = e.deltaY;
  if (e.deltaMode === 1) d *= 33;
  else if (e.deltaMode === 2) d *= 400;
  const notches = -d / 100;
  if (dive.on) {
    dive.speed = Math.max(-4, Math.min(4, dive.speed + notches * 0.15));
    say(`Скорость ${dive.speed.toFixed(2)}`);
    return;
  }
  pendingZoom += notches * (e.ctrlKey ? 1.2 : 0.42);
  pendingZoom = Math.max(-6, Math.min(6, pendingZoom));
  anchor = toOffset(e.clientX, e.clientY);
}, { passive: false });

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button });
  if (e.button === 2) rightDragged = false;
  dragMoved = 0;
  panVel = [0, 0];
  pendingZoom = 0;
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
  }
});

canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const r = canvas.getBoundingClientRect();
  const k = 2 / r.height;
  const dx = (e.clientX - p.x) * k;
  const dy = (e.clientY - p.y) * k;
  p.x = e.clientX;
  p.y = e.clientY;
  if (pointers.size === 1 && p.button === 2) {
    // right drag: slide the Julia parameter; the finer the zoom, the finer the change
    if (spec().julia) {
      const k2 = 0.5 * Math.pow(2, view.ls);
      jc.re += dx * k2;
      jc.im -= dy * k2;
      morph = false;
      dirtyHash = true;
    }
    dragMoved += Math.abs(dx) + Math.abs(dy);
    if (dragMoved > 0.01) rightDragged = true;
  } else if (pointers.size === 1) {
    pan(view, -dx, dy);
    dragMoved += Math.abs(dx) + Math.abs(dy);
    const now = performance.now();
    const dt = Math.max(1, now - lastDrag) / 1000;
    lastDrag = now;
    panVel = [panVel[0] * 0.6 + (-dx / dt) * 0.4, panVel[1] * 0.6 + (dy / dt) * 0.4];
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const mid = toOffset((a.x + b.x) / 2, (a.y + b.y) / 2);
    pan(view, -dx / 2, dy / 2);
    if (pinchDist > 0 && dist > 0) zoomAt(view, mid[0], mid[1], -Math.log2(dist / pinchDist));
    pinchDist = dist;
    dragMoved = 1;
  }
});

function pointerUp(e: PointerEvent): void {
  if (!pointers.delete(e.pointerId)) return;
  if (pointers.size > 0 || performance.now() - lastDrag > 60) panVel = [0, 0];
  if (dive.on && dragMoved < 0.01 && e.button === 0) setDive(false);
}
canvas.addEventListener('pointerup', pointerUp);
canvas.addEventListener('pointercancel', pointerUp);

canvas.addEventListener('dblclick', (e) => {
  const [x, y] = toOffset(e.clientX, e.clientY);
  setDive(true, pointAt(view, x, y));
});
// double right click: pull back out from the current view
let lastRight = 0;
canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (rightDragged) {
    rightDragged = false;
    lastRight = 0;
    return;
  }
  const now = performance.now();
  if (now - lastRight < 400) {
    lastRight = 0;
    setDive(true, null, true);
  } else lastRight = now;
});

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target instanceof HTMLInputElement) return;
  keys.add(e.code);
  switch (e.code) {
    case 'Space':
      e.preventDefault();
      if (!e.repeat) setDive(!dive.on);
      break;
    case 'Escape':
      if (dive.on) setDive(false);
      break;
    case 'KeyR':
      Object.assign(view, homeView());
      pendingZoom = 0;
      panVel = [0, 0];
      if (dive.on) setDive(false);
      break;
    case 'KeyK':
      if (!spec().julia) say('Сначала откройте множество Жюлиа (J или меню)');
      else {
        morph = !morph;
        say(morph ? 'Параметр c плавно вращается' : 'Параметр c остановлен');
      }
      break;
    case 'KeyJ':
      toggleJulia();
      break;
    case 'Tab':
      e.preventDefault();
      document.body.classList.toggle('menuless');
      break;
    case 'KeyM':
      setPower(power + (e.shiftKey ? -1 : 1));
      break;
    case 'KeyN':
      setPalette(palette + (e.shiftKey ? -1 : 1));
      say(PALETTES[palette].name);
      break;
    case 'KeyP':
      cycling = !cycling;
      say(cycling ? 'Переливы включены' : 'Переливы выключены');
      break;
    case 'KeyH':
      document.body.classList.toggle('clean');
      break;
    case 'KeyF':
      if (document.fullscreenElement) void document.exitFullscreen();
      else void document.documentElement.requestFullscreen();
      break;
    case 'BracketRight':
      iterMult = Math.min(64, iterMult * 1.5);
      say(`Итерации ×${iterMult.toFixed(2)}`);
      break;
    case 'BracketLeft':
      iterMult = Math.max(0.1, iterMult / 1.5);
      say(`Итерации ×${iterMult.toFixed(2)}`);
      break;
    default:
      if (/^Digit\d$/.test(e.code)) {
        const n = Number(e.code.slice(5));
        setPalette(n === 0 ? 9 : n - 1);
        say(PALETTES[palette].name);
      }
  }
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => keys.clear());

function updateInput(dt: number): void {
  // wheel zoom eases in, anchored where the cursor was
  if (Math.abs(pendingZoom) > 1e-4) {
    const step = pendingZoom * (1 - Math.exp(-dt * 12));
    pendingZoom -= step;
    zoomAt(view, anchor[0], anchor[1], -step);
  } else pendingZoom = 0;

  const fast = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 3 : 1;
  let kx = 0;
  let ky = 0;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) kx -= 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) kx += 1;
  if (keys.has('KeyW') || keys.has('ArrowUp')) ky += 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) ky -= 1;
  if (kx || ky) pan(view, kx * 0.9 * fast * dt, ky * 0.9 * fast * dt);
  let kz = 0;
  if (keys.has('KeyE') || keys.has('Equal') || keys.has('NumpadAdd')) kz += 1;
  if (keys.has('KeyQ') || keys.has('Minus') || keys.has('NumpadSubtract')) kz -= 1;
  if (kz) zoomAt(view, 0, 0, -kz * 1.4 * fast * dt);

  // fling after a drag
  if (pointers.size === 0 && (panVel[0] || panVel[1])) {
    pan(view, panVel[0] * dt, panVel[1] * dt);
    const k = Math.exp(-dt * 5);
    panVel = [panVel[0] * k, panVel[1] * k];
    if (Math.hypot(panVel[0], panVel[1]) < 0.002) panVel = [0, 0];
  }

  if (dive.on) {
    let want = dive.speed;
    if (dive.lost > 0) {
      // nothing but the inside of the set in view: back out until structure returns
      want = -0.9;
      dive.lost -= dt;
    }
    // Two eased stages, so nothing ever jumps: the aim point glides to the chosen
    // target, and the camera glides to the aim while zooming about it.
    let ax = 0;
    let ay = 0;
    if (dive.target && dive.aim && pointers.size === 0) {
      const [tx, ty] = offsetIn(view, dive.target);
      const [mx, my] = offsetIn(view, dive.aim);
      if (Math.abs(tx) < aspect() * 1.5 && Math.abs(ty) < 1.5) {
        const ka = 1 - Math.exp(-dt * 1.2);
        const nx = mx + (tx - mx) * ka;
        const ny = my + (ty - my) * ka;
        dive.aim = pointAt(view, nx, ny);
        const k = 1 - Math.exp(-dt * 0.8);
        pan(view, nx * k, ny * k);
        ax = nx * (1 - k);
        ay = ny * (1 - k);
        // turn first, dive after: hold back while the aim is far from the centre
        if (want > 0) want *= Math.max(0.3, Math.min(1, 1.15 - Math.hypot(nx, ny) / 0.9));
      } else if (!dive.auto) setDive(false);
      else dive.target = null;
    }
    // The camera goes no faster than pictures come. Nobody steers a dive by hand, so
    // nothing is lost by pacing it to the renderer: every frame on the way is then a
    // finished one, and the further the picture falls behind, the slower the camera.
    const pace = Math.max(0.08, Math.min(1, 1.25 - renderer.behind / 0.3));
    dive.pace += (pace - dive.pace) * (1 - Math.exp(-dt * 5));
    want *= dive.pace;
    dive.rate += (want - dive.rate) * (1 - Math.exp(-dt * 2.5));
    zoomAt(view, ax, ay, -dive.rate * dt);
    if (view.ls >= LS_MAX && dive.rate < 0) setDive(false);
  }
}

// --- autopilot: steer towards the richest part of the picture -------------------

let lastPick = 0;
renderer.onProbe = (r: ProbeResult) => {
  if (r.active > 0.4) return; // too early to judge the picture
  const W = 64;
  const H = 36;
  const d = r.data;
  const now = performance.now();

  // reference hint: prefer a pixel that never escaped, nearest the centre
  {
    let bestD = Infinity;
    let bestIt = 0;
    let bestIn = false;
    let hx = 0;
    let hy = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const ox = ((Math.floor(((x + 0.5) / W) * r.w) + 0.5) / r.w * 2 - 1) * r.aspect;
        const oy = (Math.floor(((y + 0.5) / H) * r.h) + 0.5) / r.h * 2 - 1;
        const st = d[i + 1];
        const running = st === 0 || st === 2; // 3 is only carried over from an earlier picture
        const dist = running ? ox * ox + oy * oy : 1e9 - d[i];
        // best of all a pixel proven inside, then one still running, then the slowest to escape
        const key = st === 2 ? dist - 1e12 : dist;
        if (key < bestD) {
          bestD = key;
          bestIt = st === 2 ? Infinity : d[i];
          bestIn = st === 2;
          hx = ox;
          hy = oy;
        }
      }
    }
    refHint = { p: pointAt(r.view, hx, hy), iters: bestIt, inside: bestIn, t: now };
  }

  if (!dive.on || !dive.auto) return;
  if (now - lastPick < 500) return;
  lastPick = now;
  const val = new Float32Array(W * H);
  let escaped = 0;
  for (let i = 0; i < W * H; i++) {
    const ok = d[i * 4 + 1] === 1;
    val[i] = ok ? Math.log(d[i * 4] + 1) : -1;
    if (ok) escaped++;
  }
  if (escaped < W * H * 0.03) {
    dive.lost = 0.6;
    return;
  }
  let lo = Infinity;
  let hi = -Infinity;
  // per 3x3 block: mean and spread of log(iterations), and how much of it never escaped
  const mean = new Float32Array(W * H);
  const dev = new Float32Array(W * H);
  const avg = new Float32Array(W * H).fill(-1e9);
  let devHi = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      let s1 = 0;
      let s2 = 0;
      let n = 0;
      for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
          const v = val[(y + j) * W + x + i];
          if (v >= 0) {
            s1 += v;
            s2 += v * v;
            n++;
          }
        }
      }
      if (n < 5) continue;
      const a = s1 / n;
      const sd = Math.sqrt(Math.max(0, s2 / n - a * a));
      mean[y * W + x] = a;
      dev[y * W + x] = sd;
      avg[y * W + x] = n; // marks the block as usable; replaced by its score below
      if (a < lo) lo = a;
      if (a > hi) hi = a;
      if (sd > devHi) devHi = sd;
    }
  }
  if (!(hi > lo)) return;
  // Deep iterations alone lead into dull, half-black edges. Busy texture is what
  // looks good, so it counts for more, and blocks touching the inside count against.
  for (let i = 0; i < W * H; i++) {
    if (avg[i] < 0) continue;
    const n = avg[i];
    avg[i] = 0.4 * ((mean[i] - lo) / (hi - lo)) + 0.6 * (devHi > 0 ? dev[i] / devHi : 0) - 0.6 * ((9 - n) / 9);
  }
  const cur = dive.target ? offsetIn(r.view, dive.target) : null;
  const diag = Math.hypot(r.aspect, 1);
  let best = -Infinity;
  let here = -Infinity; // best score around the target we already have
  let bx = 0;
  let by = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const a = avg[y * W + x];
      if (a < -1e8) continue;
      const ox = (((x + 0.5) / W) * 2 - 1) * r.aspect;
      const oy = ((y + 0.5) / H) * 2 - 1;
      const s = a - 0.6 * ((ox * ox + oy * oy) / (diag * diag));
      if (cur && Math.hypot(ox - cur[0], oy - cur[1]) < 0.15 && s > here) here = s;
      if (s > best) {
        best = s;
        bx = ox;
        by = oy;
      }
    }
  }
  // change course only for a clearly better place, not for every flicker in the scores
  if (best > -Infinity && best > here + 0.15) dive.target = pointAt(r.view, bx, by);
};

// --- HUD -------------------------------------------------------------------------

function fmtZoom(ls: number): string {
  const lg = (homeView().ls - ls) * 0.30103;
  if (lg < 4) return `×${Math.pow(10, lg).toFixed(lg < 1 ? 2 : 0)}`;
  const e = Math.floor(lg);
  return `×${Math.pow(10, lg - e).toFixed(1)}·10^${e}`;
}

let fps = 60;
function updateHud(): void {
  const s = renderer.stats;
  const state = refPending
    ? 'считаю опорную орбиту…'
    : s.busy
      ? s.samples > 0 && s.scale === 1 ? `сглаживание ${s.samples}/${renderer.maxSamples}` : 'рендер…'
      : 'готово';
  hud.innerHTML =
    `<b>${fmtZoom(view.ls)}</b>` +
    `<span>${Math.round(s.N || itersFor(view.ls)).toLocaleString('ru-RU')} итераций${s.deepening ? ' ↑' : ''}${(s.N || itersFor(view.ls)) >= maxIter ? ' · потолок' : ''}</span>` +
    `<span>${PALETTES[palette].name}</span>` +
    (spec().julia ? `<span>c = ${jc.re.toFixed(5)} ${jc.im < 0 ? '−' : '+'} ${Math.abs(jc.im).toFixed(5)}i</span>` : '') +
    `<span>${formula.name}${juliaOn ? ' · Жюлиа' : ''}${formula.hasPower && power !== (formula.power ?? 2) ? ` · степень ${power}` : ''}</span>` +
    `<span>${fps.toFixed(0)} fps · ${Math.round(s.scale * 100)}%</span>` +
    `<span>${state}</span>` +
    (dive.on ? `<span class="dive">погружение ${dive.speed.toFixed(2)}</span>` : '');
}

// --- main loop -------------------------------------------------------------------

function fitCanvas(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(64, Math.min(3840, Math.round(canvas.clientWidth * dpr)));
  const h = Math.max(64, Math.min(2160, Math.round(canvas.clientHeight * dpr)));
  if (w === renderer.W && h === renderer.H) return;
  renderer.resize(w, h);
  applyLimits(); // the frame buffers take their share of the video memory first
}

let last = performance.now();
let lastView = cloneView(view);
let stillSince = last;
let unaligned = false; // the view has moved since it was last put on a picture's pixel grid
let lastHud = 0;
let lastSave = 0;
let dirtyHash = false;
// The display's frame time: the usual gap between frames. A frame that comes late is
// followed by one that comes early, and a dropped one doubles the gap; the lower quartile
// of the last gaps is neither. (The smallest gap seen would be an early frame's: far too
// short, and the time allowed for a pass with it.)
let nominal = 16.7;
const gaps = new Float64Array(64);
let gapCount = 0;

function frame(t: number): void {
  const dtMs = Math.min(100, Math.max(1, t - last));
  last = t;
  const dt = dtMs / 1000;
  fps += (1000 / dtMs - fps) * 0.05;
  gaps[gapCount++ & 63] = dtMs;
  if (gapCount < 32) nominal = Math.min(nominal, Math.max(4, dtMs));
  else if ((gapCount & 15) === 0) nominal = Math.max(2.5, Array.from(gaps).sort((a, b) => a - b)[16]);
  // at rest nothing on screen moves, so the picture may take most of each frame
  // (on a fast display no more than a frame and a half or so: the renderer keeps at most two
  // passes in hand, and that is then the longest anything on screen can be held up)
  renderer.targetMs = (window as any).__ms || (t - stillSince > 400 && !dive.on ? Math.min(13, nominal * 1.7) : Math.max(2, Math.min(9, Math.max(nominal * 0.9 - renderer.stats.restMs, renderer.stats.restMs * 0.7 + 1))));
  renderer.frameTime(dtMs, nominal);

  fitCanvas();
  updateInput(dt);
  fitPrec(view);
  const moved = !sameView(view, lastView);
  if (moved) {
    stillSince = t;
    lastView = cloneView(view);
    dirtyHash = true;
    unaligned = true;
  } else if (unaligned && !dive.on) {
    // The view has just stopped. If it was only dragged since the last careful picture,
    // a shift of under half a pixel puts it on that picture's grid, and everything
    // already worked out can be kept as it is (see Renderer.align).
    unaligned = false;
    const c = renderer.align(view);
    if (c) {
      pan(view, c[0], c[1]);
      lastView = cloneView(view);
    }
  }
  if (morph && spec().julia) {
    const a = dt * 0.12;
    [jc.re, jc.im] = [jc.re * Math.cos(a) - jc.im * Math.sin(a), jc.re * Math.sin(a) + jc.im * Math.cos(a)];
    dirtyHash = true;
  }
  manageRef();
  if (cycling) renderer.phase = (renderer.phase + dt * 0.03) % 2;
  renderer.auto = dive.on;
  renderer.frame(view, t - stillSince > 70, itersFor(view.ls));

  if (t - lastHud > 200) {
    lastHud = t;
    updateHud();
  }
  if (dirtyHash && t - lastSave > 600 && t - stillSince > 300) {
    lastSave = t;
    dirtyHash = false;
    saveHash();
  }
  requestAnimationFrame(frame);
}

loadHash();
fitPrec(view);
setPalette(palette);
fitCanvas();
applyFormula(true);
dirtyHash = false;
help.addEventListener('click', (e) => {
  if (!(e.target instanceof HTMLInputElement)) document.body.classList.toggle('helpless');
});
vramInput.addEventListener('change', () => {
  setVram(Number(vramInput.value)); // empty or 0: back to automatic
  vramInput.blur();
});
if (lostNote) say('Лимит видеопамяти сброшен: в прошлый раз видеокарта не справилась');
powerInput.max = String(MAX_POWER);
powerInput.value = String(power);
powerInput.addEventListener('change', () => {
  const p = Math.round(Number(powerInput.value));
  if (Number.isFinite(p)) setPower(Math.max(2, Math.min(MAX_POWER, p)));
  powerInput.blur();
});
requestAnimationFrame(frame);

// --- test hooks ---------------------------------------------------------------------

/** Exact smooth iteration count at job pixel (x, y), by brute-force BigInt iteration. */
function bruteNu(px: number, py: number, limit = 0): number {
  const sp = spec();
  // compare like with like: the limit the finished picture was drawn with
  const N = limit || renderer.finalN() || itersFor(view.ls);
  const prec = precFor(view.ls - 64);
  const ec = Math.floor(view.ls);
  const sm = Math.pow(2, view.ls - ec);
  const ox = (((px + 0.5) / renderer.W) * 2 - 1) * aspect() * sm;
  const oy = (((py + 0.5) / renderer.H) * 2 - 1) * sm;
  const [qx, qy] = paramOf(sp, rescale(view.cx, view.prec, prec) + dblToFixed(ox, ec, prec), rescale(view.cy, view.prec, prec) + dblToFixed(oy, ec, prec), prec);
  if (sp.kind === 'custom') return customBrute(sp, prec, [qx, qy], N);
  // the pixel is c for a parameter plane, the starting z for a Julia set
  const cx = sp.julia ? dblToFixed(jc.re, 0, prec) : qx;
  const cy = sp.julia ? dblToFixed(jc.im, 0, prec) : qy;
  const step = makeBigStep(sp, prec);
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  let zx = sp.julia ? qx : 0n;
  let zy = sp.julia ? qy : 0n;
  for (let n = 0; n <= N; n++) {
    const x = Number(zx >> SH) * INV;
    const y = Number(zy >> SH) * INV;
    const zz = x * x + y * y;
    if (zz > R2) return Math.max(0, n + 1 + (Math.log2(Math.log2(R2)) - Math.log2(Math.log2(zz))) / Math.log2(sp.power));
    step(zx, zy, cx, cy);
    zx = sx;
    zy = sy;
  }
  return -1;
}

/** Test helper: which outcome a point has - inside (-1), or for Newton the root it reaches, else 0. */
function classAt(qx: bigint, qy: bigint, prec: number): number {
  const sp = spec();
  if (sp.kind === 'custom') {
    const nu = customBrute(sp, prec, [qx, qy], 1500, true);
    return nu < 0 ? -1 : formula.id === 'newton' ? nu : 0;
  }
  [qx, qy] = paramOf(sp, qx, qy, prec);
  const step = makeBigStep(sp, prec);
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  let zx = sp.julia ? qx : 0n;
  let zy = sp.julia ? qy : 0n;
  const cx = sp.julia ? dblToFixed(jc.re, 0, prec) : qx;
  const cy = sp.julia ? dblToFixed(jc.im, 0, prec) : qy;
  for (let n = 0; n < 1500; n++) {
    const x = Number(zx >> SH) * INV;
    const y = Number(zy >> SH) * INV;
    if (x * x + y * y > R2) return 0;
    step(zx, zy, cx, cy);
    zx = sx;
    zy = sy;
  }
  return -1;
}

/** Test helper: two points of the home view with different outcomes, to bisect between. */
function findBracket(): number[] | null {
  const h = homeView();
  const half = Math.pow(2, h.ls);
  const cx = fixedToDbl(h.cx, h.prec, 0);
  const cy = fixedToDbl(h.cy, h.prec, 0);
  let first: number[] | null = null;
  for (let j = 0; j < 9; j++) {
    for (let i = 0; i < 16; i++) {
      const re = cx + ((i + 0.37) / 16 - 0.5) * 2 * half * aspect();
      const im = cy + ((j + 0.41) / 9 - 0.5) * 2 * half;
      const k = classAt(dblToFixed(re, 0, 128), dblToFixed(im, 0, 128), 128);
      if (!first) first = [re, im, k];
      else if (k !== first[2]) return [first[0], first[1], re, im];
    }
  }
  return null;
}

/** Test helper: like findBracket, but starting from a random spot of the home view. */
function randomBracket(r1: number, r2: number, r3: number, r4: number): number[] | null {
  const h = homeView();
  const half = Math.pow(2, h.ls);
  const cx = fixedToDbl(h.cx, h.prec, 0);
  const cy = fixedToDbl(h.cy, h.prec, 0);
  const at = (u: number, v: number) => [cx + (u - 0.5) * 2 * half * aspect(), cy + (v - 0.5) * 2 * half];
  const a = at(r1, r2);
  const ka = classAt(dblToFixed(a[0], 0, 128), dblToFixed(a[1], 0, 128), 128);
  // walk outwards from a second random spot until the outcome differs
  for (let i = 0; i < 60; i++) {
    const b = at((r3 + i * 0.7548776662) % 1, (r4 + i * 0.569840291) % 1);
    if (classAt(dblToFixed(b[0], 0, 128), dblToFixed(b[1], 0, 128), 128) !== ka) return [a[0], a[1], b[0], b[1]];
  }
  return null;
}

/**
 * Test helper: bisect between two points with different outcomes until the
 * boundary is pinned down to 2^-bits, then look at it from 2^(8-bits) away.
 */
function gotoBoundary(ax: number, ay: number, bx: number, by: number, bits: number): boolean {
  const prec = Math.ceil((bits + 96) / 32) * 32;
  let px = dblToFixed(ax, 0, prec);
  let py = dblToFixed(ay, 0, prec);
  let qx = dblToFixed(bx, 0, prec);
  let qy = dblToFixed(by, 0, prec);
  const ka = classAt(px, py, prec);
  if (classAt(qx, qy, prec) === ka) return false;
  for (let i = 0; i < bits + 8; i++) {
    const mx = (px + qx) >> 1n;
    const my = (py + qy) >> 1n;
    if (classAt(mx, my, prec) === ka) [px, py] = [mx, my];
    else [qx, qy] = [mx, my];
  }
  view.cx = qx;
  view.cy = qy;
  view.prec = prec;
  view.ls = 8 - bits;
  fitPrec(view);
  return true;
}

/** Test helper: the first iterates of a custom formula at a pixel, as plain numbers. */
function orbitAt(px: number, py: number, n: number): number[][] {
  const sp = spec();
  const prec = precFor(view.ls - 64);
  const ec = Math.floor(view.ls);
  const sm = Math.pow(2, view.ls - ec);
  const ox = (((px + 0.5) / renderer.W) * 2 - 1) * aspect() * sm;
  const oy = (((py + 0.5) / renderer.H) * 2 - 1) * sm;
  const q: [bigint, bigint] = [rescale(view.cx, view.prec, prec) + dblToFixed(ox, ec, prec), rescale(view.cy, view.prec, prec) + dblToFixed(oy, ec, prec)];
  const f = makeCustom(sp, prec);
  const st = f.start(q);
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push([f.toD(st.z[0]), f.toD(st.z[1])]);
    f.step(st, q);
  }
  return out;
}

(window as any).__fx = {
  orbitAt,
  gotoBoundary,
  findBracket,
  goto: (re: number, im: number, ls: number) => {
    view.ls = ls;
    view.prec = precFor(ls);
    view.cx = dblToFixed(re, 0, view.prec);
    view.cy = dblToFixed(im, 0, view.prec);
    return true;
  },
  randomBracket,
  nudge: (dx: number, dy: number) => pan(view, dx, dy),
  home: () => formula.home,
  renderer,
  view,
  dive,
  bruteNu,
  at: (x: number, y: number) => {
    const [ox, oy] = toOffset(x, y);
    const p = pointAt(view, ox, oy);
    return [p.cx.toString(), p.cy.toString(), p.prec, view.ls];
  },
  idle: () => !refPending && renderer.idle(view),
  setCycling: (on: boolean) => {
    cycling = on;
  },
  setVram,
  setSystem: (vram: number, ram: number) => {
    sys.vram = vram;
    sys.ram = ram;
    applyLimits();
  },
  limits: () => ({ vram: vramLimit(), max: vramMax(), maxIter, capBy, shrink, sys: { ...sys }, base: Math.round(renderer.baseBytes() / MB) }),
  setFormula: (id: string) => setFormula(FORMULAS.find((f) => f.id === id)!),
  setJulia,
  formulas: FORMULAS.map((f) => f.id),
  info: () => ({ refMs, refOrbitMs, ref: renderer.ref && { M: renderer.ref.M, lenA: renderer.ref.lenA, cap: renderer.ref.cap, escaped: renderer.ref.escaped, levels: renderer.ref.blaLevels }, stats: renderer.stats, last: renderer.lastJob, q: renderer.q, N: renderer.finalN() || itersFor(view.ls), N0: itersFor(view.ls), per: renderer.ref && renderer.ref.perA, interior: renderer.ref && renderer.ref.interior }),
};

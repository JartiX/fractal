import { Renderer, ProbeResult } from './renderer';
import {
  View, Point, cloneView, sameView, fitPrec, pan, zoomAt, pointAt, offsetIn, precFor, rescale,
  parseDecimal, formatDecimal, dblToFixed, LS_MIN, LS_MAX,
} from './view';
import { PALETTES, bakePalette } from './palettes';
import type { RefRequest, RefResult, BlaResult, BlaRequest } from './refWorker';
import { R2, MAX_POWER } from './refWorker';

const HOME: View = { cx: -(3n << 62n), cy: 0n, prec: 64, ls: 0.35 };
/** Starting view: the power-2 set sits left of the origin, higher powers are centred on it. */
function homeView(): View {
  return power === 2 ? cloneView(HOME) : { cx: 0n, cy: 0n, prec: 64, ls: 0.35 };
}
const MAX_ITER = 6_000_000;

const canvas = document.getElementById('view') as HTMLCanvasElement;
const hud = document.getElementById('hud')!;
const help = document.getElementById('help')!;
const toast = document.getElementById('toast')!;
const powerInput = document.getElementById('power') as HTMLInputElement;

let renderer: Renderer;
try {
  renderer = new Renderer(canvas);
} catch (err) {
  document.getElementById('fatal')!.textContent = String((err as Error).message || err);
  throw err;
}

let power = 2; // z -> z^power + c
const view: View = cloneView(HOME);
let iterMult = 1;
let palette = 0;
let cycling = true;

// --- location in the URL -----------------------------------------------------

function loadHash(): void {
  const q = new URLSearchParams(location.hash.slice(1));
  const z = Number(q.get('z'));
  if (q.has('d')) power = Math.max(2, Math.min(MAX_POWER, Number(q.get('d')) | 0));
  if (!q.has('re') || !q.has('im') || !Number.isFinite(z)) {
    Object.assign(view, homeView());
    return;
  }
  view.ls = Math.min(LS_MAX, Math.max(LS_MIN, z));
  view.prec = precFor(view.ls);
  view.cx = parseDecimal(q.get('re')!, view.prec);
  view.cy = parseDecimal(q.get('im')!, view.prec);
  if (q.has('p')) palette = Math.max(0, Math.min(PALETTES.length - 1, Number(q.get('p')) | 0));
  if (q.has('i')) iterMult = Math.max(0.1, Math.min(64, Number(q.get('i')) || 1));
}

function saveHash(): void {
  const digits = Math.ceil(Math.max(0, -view.ls) * 0.30103) + 6;
  const s =
    `re=${formatDecimal(view.cx, view.prec, digits)}&im=${formatDecimal(view.cy, view.prec, digits)}` +
    `&z=${view.ls.toFixed(4)}&p=${palette}` + (power !== 2 ? `&d=${power}` : '') + (iterMult !== 1 ? `&i=${iterMult.toFixed(3)}` : '');
  history.replaceState(null, '', '#' + s);
}

// --- iterations and the reference orbit --------------------------------------

function itersFor(ls: number): number {
  const depth = Math.max(0, -ls);
  return Math.min(MAX_ITER, Math.round(iterMult * (1000 + 80 * Math.pow(depth, 1.5))));
}

const worker = new Worker(new URL('./refWorker.ts', import.meta.url), { type: 'module' });
let refSeq = 0;
// the deepest point seen in the last picture: the best place for a reference orbit
let refHint: { p: Point; iters: number; t: number } | null = null;
let lastHintRef = 0; // when a reference was last moved to a hint
let refPending = false;
let blaPending = false;
let refMs = 0;

worker.onmessage = (ev: MessageEvent<RefResult | BlaResult>) => {
  const m = ev.data;
  if (m.type === 'ref') {
    if (m.id !== refSeq) return;
    refPending = false;
    blaPending = false;
    refMs = m.ms;
    renderer.setRef(m);
  } else {
    blaPending = false;
    if (renderer.ref && renderer.ref.id === m.id) renderer.setBla(m);
  }
};

function aspect(): number {
  return renderer.W / renderer.H;
}

function manageRef(): void {
  if (refPending) return;
  const ref = renderer.ref;
  const N = itersFor(view.ls);
  const off = renderer.refOffset(view);
  let need = !ref || !off;
  let at: Point = view;
  if (refHint) {
    const [hx, hy] = offsetIn(view, refHint.p);
    if (performance.now() - refHint.t < 1500 && Math.abs(hx) < aspect() && Math.abs(hy) < 1) at = refHint.p;
    else refHint = null;
  }
  if (ref && off) {
    const far = Math.hypot(off.dm[0], off.dm[1]) / Math.pow(2, view.ls - Math.floor(view.ls));
    need =
      (!ref.escaped && N > ref.maxIter * 0.85) ||
      ref.prec < -view.ls + 56 ||
      !(far < 1.5) ||
      // the orbit escapes long before the picture is done: pixels outlive it and
      // lose the BLA speed-up, so move the reference somewhere deeper
      (ref.escaped && ref.M <= N && !!refHint && refHint.iters > ref.M * 1.3 && performance.now() - lastHintRef > 700);
  }
  if (need) {
    if (at !== view) {
      // a hint is used once; the next one must come from a picture made with the new orbit
      lastHintRef = performance.now();
      refHint = null;
    }
    const prec = precFor(view.ls - 64);
    const req: RefRequest = {
      type: 'ref',
      id: ++refSeq,
      cx: rescale(at.cx, at.prec, prec),
      cy: rescale(at.cy, at.prec, prec),
      prec,
      maxIter: itersFor(view.ls - 16),
      lcmax: view.ls + Math.log2(1.5 + Math.hypot(aspect(), 1)) + 2,
      power,
    };
    refPending = true;
    worker.postMessage(req);
    return;
  }
  // the BLA table is built for a bound on |dc|; rebuild when the view outgrows it
  if (ref && off && !blaPending && ref.M > 64 && (off.lc > ref.lcmax || off.lc < ref.lcmax - 48)) {
    const req: BlaRequest = { type: 'bla', id: ref.id, lcmax: off.lc + 2 };
    blaPending = true;
    worker.postMessage(req);
  }
}

// --- input ---------------------------------------------------------------------

const keys = new Set<string>();
const pointers = new Map<number, { x: number; y: number }>();
let pendingZoom = 0; // octaves still to apply (positive = in)
let anchor: [number, number] = [0, 0];
let panVel: [number, number] = [0, 0];
let dragMoved = 0;
let lastDrag = 0;
let pinchDist = 0;

const dive = {
  on: false,
  auto: true,
  speed: 0.7, // octaves per second
  target: null as Point | null,
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

function setPower(p: number): void {
  power = p > MAX_POWER ? 2 : p < 2 ? MAX_POWER : p;
  // positions mean nothing across formulas, so start over from the whole set
  Object.assign(view, homeView());
  pendingZoom = 0;
  panVel = [0, 0];
  if (dive.on) setDive(false);
  refHint = null;
  refPending = false;
  refSeq++; // any orbit still being computed is for the old formula
  renderer.reset();
  powerInput.value = String(power);
  say(`Степень ${power}`);
}

function setDive(on: boolean, target: Point | null = null, out = false): void {
  dive.on = on;
  dive.auto = !target && !out;
  dive.target = target;
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
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
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
  if (pointers.size === 1) {
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
  const now = performance.now();
  if (now - lastRight < 400) {
    lastRight = 0;
    setDive(true, null, true);
  } else lastRight = now;
});

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target === powerInput) return;
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
    let speed = dive.speed;
    if (dive.lost > 0) {
      // nothing but the inside of the set in view: back out until structure returns
      speed = -1.2;
      dive.lost -= dt;
    }
    // zoom about the target so it never slides away, and ease it to the centre
    let ax = 0;
    let ay = 0;
    if (dive.target && pointers.size === 0) {
      const [tx, ty] = offsetIn(view, dive.target);
      if (Math.abs(tx) < aspect() * 1.5 && Math.abs(ty) < 1.5) {
        const k = 1 - Math.exp(-dt * 0.9);
        pan(view, tx * k, ty * k);
        ax = tx * (1 - k);
        ay = ty * (1 - k);
      } else if (!dive.auto) setDive(false);
    }
    zoomAt(view, ax, ay, -speed * dt);
    if (view.ls >= LS_MAX && speed < 0) setDive(false);
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
    let hx = 0;
    let hy = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const ox = ((Math.floor(((x + 0.5) / W) * r.w) + 0.5) / r.w * 2 - 1) * r.aspect;
        const oy = (Math.floor(((y + 0.5) / H) * r.h) + 0.5) / r.h * 2 - 1;
        const running = d[i + 1] !== 1;
        const dist = running ? ox * ox + oy * oy : 1e9 - d[i];
        if (dist < bestD) {
          bestD = dist;
          bestIt = d[i + 1] === 2 && !r.capped ? Infinity : d[i];
          hx = ox;
          hy = oy;
        }
      }
    }
    refHint = { p: pointAt(r.view, hx, hy), iters: bestIt, t: now };
  }

  if (!dive.on || !dive.auto) return;
  if (now - lastPick < 350) return;
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
  const avg = new Float32Array(W * H).fill(-1);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      let s = 0;
      let n = 0;
      for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
          const v = val[(y + j) * W + x + i];
          if (v >= 0) {
            s += v;
            n++;
          }
        }
      }
      if (n < 5) continue;
      const a = s / n;
      avg[y * W + x] = a;
      if (a < lo) lo = a;
      if (a > hi) hi = a;
    }
  }
  if (!(hi > lo)) return;
  const cur = dive.target ? offsetIn(r.view, dive.target) : null;
  const diag = Math.hypot(r.aspect, 1);
  let best = -Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const a = avg[y * W + x];
      if (a < 0) continue;
      const ox = (((x + 0.5) / W) * 2 - 1) * r.aspect;
      const oy = ((y + 0.5) / H) * 2 - 1;
      let s = (a - lo) / (hi - lo) - 0.45 * ((ox * ox + oy * oy) / (diag * diag));
      if (cur && Math.hypot(ox - cur[0], oy - cur[1]) < 0.12) s += 0.12; // stick with the current target
      if (s > best) {
        best = s;
        bx = ox;
        by = oy;
      }
    }
  }
  if (best > -Infinity) dive.target = pointAt(r.view, bx, by);
};

// --- HUD -------------------------------------------------------------------------

function fmtZoom(ls: number): string {
  const lg = (HOME.ls - ls) * 0.30103;
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
    `<span>${itersFor(view.ls).toLocaleString('ru-RU')} итераций</span>` +
    `<span>${PALETTES[palette].name}</span>` +
    (power !== 2 ? `<span>степень ${power}</span>` : '') +
    `<span>${fps.toFixed(0)} fps · ${Math.round(s.scale * 100)}%</span>` +
    `<span>${state}</span>` +
    (dive.on ? `<span class="dive">погружение ${dive.speed.toFixed(2)}</span>` : '');
}

// --- main loop -------------------------------------------------------------------

function fitCanvas(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(64, Math.min(3840, Math.round(canvas.clientWidth * dpr)));
  const h = Math.max(64, Math.min(2160, Math.round(canvas.clientHeight * dpr)));
  renderer.resize(w, h);
}

let last = performance.now();
let lastView = cloneView(view);
let stillSince = last;
let lastHud = 0;
let lastSave = 0;
let dirtyHash = false;
let nominal = 16.7;

function frame(t: number): void {
  const dtMs = Math.min(100, Math.max(1, t - last));
  last = t;
  const dt = dtMs / 1000;
  fps += (1000 / dtMs - fps) * 0.05;
  nominal = Math.min(nominal * 1.002, dtMs);
  renderer.targetMs = (window as any).__ms || Math.max(3, Math.min(9, nominal * 0.7));
  renderer.frameTime(dtMs, nominal);

  fitCanvas();
  updateInput(dt);
  fitPrec(view);
  const moved = !sameView(view, lastView);
  if (moved) {
    stillSince = t;
    lastView = cloneView(view);
    dirtyHash = true;
  }
  manageRef();
  if (cycling) renderer.phase = (renderer.phase + dt * 0.03) % 2;
  renderer.frame(view, t - stillSince > 150, itersFor(view.ls));

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
help.addEventListener('click', (e) => {
  if (e.target !== powerInput) document.body.classList.toggle('helpless');
});
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
function bruteNu(px: number, py: number): number {
  const N = itersFor(view.ls);
  const prec = precFor(view.ls - 64);
  const ec = Math.floor(view.ls);
  const sm = Math.pow(2, view.ls - ec);
  const ox = (((px + 0.5) / renderer.W) * 2 - 1) * aspect() * sm;
  const oy = (((py + 0.5) / renderer.H) * 2 - 1) * sm;
  const cx = rescale(view.cx, view.prec, prec) + dblToFixed(ox, ec, prec);
  const cy = rescale(view.cy, view.prec, prec) + dblToFixed(oy, ec, prec);
  const P = BigInt(prec);
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  let zx = 0n;
  let zy = 0n;
  for (let n = 0; n <= N; n++) {
    const x = Number(zx >> SH) * INV;
    const y = Number(zy >> SH) * INV;
    const zz = x * x + y * y;
    if (zz > R2) return Math.max(0, n + 1 + (Math.log2(Math.log2(R2)) - Math.log2(Math.log2(zz))) / Math.log2(power));
    let px2 = zx;
    let py2 = zy;
    for (let i = 1; i < power; i++) {
      const nx = (px2 * zx - py2 * zy) >> P;
      py2 = (px2 * zy + py2 * zx) >> P;
      px2 = nx;
    }
    zx = px2 + cx;
    zy = py2 + cy;
  }
  return -1;
}

(window as any).__fx = {
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
  info: () => ({ refMs, ref: renderer.ref && { M: renderer.ref.M, escaped: renderer.ref.escaped, levels: renderer.ref.blaLevels }, stats: renderer.stats, last: renderer.lastJob, q: renderer.q, N: itersFor(view.ls) }),
};

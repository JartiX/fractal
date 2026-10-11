// Reference orbit worker.
// Computes one arbitrary-precision orbit Z_n for the reference point and the
// bilinear-approximation (BLA) table that lets the shader skip many iterations
// at once. Everything the GPU needs is packed into float32 textures here.

import { Spec, makeBigStep, sx, sy } from './formulas';
import { makeCustom, NEWTON } from './custom';
import { BlaParams, BlaState, BlaLevel, blaRange, merge, emit, concat, LMIN, MAX_LEVELS, EPS_L2, ZERO_E } from './bla';
import type { BlaJob, BlaJobResult } from './blaWorker';

export { LMIN, MAX_LEVELS, EPS_L2 };

export const R2 = 1.0e6; // bailout radius squared, must match the shader
export const TEXW = 2048;
// above this float32 overflows near the bailout radius
export const MAX_POWER = 10;
// two exact orbit points this close (in units of the last bit) are the same point of a cycle
const CYCLE_TOL = 1n << 16n;

export interface RefRequest {
  type: 'ref';
  id: number;
  cx: bigint;
  cy: bigint;
  prec: number;
  maxIter: number;
  lcmax: number;
  spec: Spec;
  vx: bigint; // the view-plane point the orbit stands for (cx, cy is the parameter iterated)
  vy: bigint;
  tag: number; // changes whenever the picture itself must change (e.g. a new Julia c)
  jx: bigint; // Julia parameter (fixed point, `prec` bits); unused otherwise
  jy: bigint;
}
export interface BlaRequest {
  type: 'bla';
  id: number;
  lcmax: number;
}
export interface BlaData {
  blaA: Float32Array | null;
  blaB: Float32Array | null; // folded formulas only: A and B are 2x2 real matrices in two textures
  blaE: Float32Array | null;
  blaRows: number;
  blaOff: Int32Array;
  blaCnt: Int32Array;
  blaLevels: number;
  lcmax: number;
  lrMax: number; // log2 of the largest validity radius in the table
}
export interface RefResult extends BlaData {
  type: 'ref';
  id: number;
  cx: bigint;
  cy: bigint;
  prec: number;
  maxIter: number;
  spec: Spec;
  vx: bigint;
  vy: bigint;
  tag: number;
  M: number; // entries in the orbit texture (all segments)
  lenA: number; // length of the first segment, the one pixels start on
  base: number; // index pixels rebase to: the start of the critical orbit
  cap: number; // most iterations a pixel may run before a truncated segment ends
  escaped: boolean; // the first segment escaped
  perA: number; // the first segment ends having come back onto itself with this period (0: it did not)
  perB: number; // likewise the last segment
  interior: boolean; // the reference point is proven to lie inside (attracting cycle)
  uOff: number; // index of the second reference quantity (custom formulas), 0 if none
  orbit: Float32Array;
  orbitRows: number;
  ms: number;
  msOrbit?: number; // of which the orbit itself (the rest is the BLA table)
}
export interface BlaResult extends BlaData {
  type: 'bla';
  id: number;
}

/** Threads to build BLA tables on, each at the other end of a port (see blaWorker.ts). */
export interface PoolMessage {
  type: 'pool';
  ports: MessagePort[];
}

/** The orbit could not be made: in practice, no memory for one this long. */
export interface FailResult {
  type: 'fail';
  id: number;
}

// --- float helpers -------------------------------------------------------

// (in the middle of a buffer of its own, clear of other threads' data: see bla.ts)
const f64 = new Float64Array(32);
const u32 = new Uint32Array(f64.buffer);
/** e such that x = m * 2^e with m in [0.5, 1) (x normal, non-zero). */
function expOf(x: number): number {
  f64[16] = x;
  return ((u32[33] >>> 20) & 0x7ff) - 1022;
}
const POFF = 1100;
const POW2 = new Float64Array(2 * POFF + 1);
for (let i = 0; i < POW2.length; i++) POW2[i] = Math.pow(2, i - POFF);
function pow2(k: number): number {
  return k < -POFF ? 0 : POW2[(k > POFF ? POFF : k) + POFF];
}

// normalised complex float-exp result registers
let nx = 0;
let ny = 0;
let ne = 0;
function normz(x: number, y: number, e: number): void {
  const ax = x < 0 ? -x : x;
  const ay = y < 0 ? -y : y;
  const m = ax > ay ? ax : ay;
  if (m === 0) {
    nx = 0;
    ny = 0;
    ne = ZERO_E;
    return;
  }
  const k = expOf(m);
  const s = pow2(-k);
  nx = x * s;
  ny = y * s;
  ne = e + k;
}

// one fixed-point component as mantissa * 2^exponent, exact to double precision however small
let fm = 0;
let fx = 0;
function comp(v: bigint, d: number, prec: number): void {
  if (v === 0n) {
    fm = 0;
    fx = ZERO_E;
    return;
  }
  let n = d;
  let s2 = 0;
  if (Math.abs(d) < 1e-6) {
    // the quick conversion has kept only the top bits of 62: redo it from the number's own top
    s2 = Math.max(0, bitLen(v) - 60);
    n = Number(v >> BigInt(s2));
    s2 -= prec;
  }
  const k = expOf(n);
  fm = n * pow2(-k);
  fx = k + s2;
}

function bitLen(v: bigint): number {
  return (v < 0n ? -v : v).toString(16).length * 4;
}

// --- cooperative yielding so a newer request can cancel a running one ------

const chan = new MessageChannel();
let wake: (() => void) | null = null;
chan.port1.onmessage = () => {
  const w = wake;
  wake = null;
  if (w) w();
};
function breathe(): Promise<void> {
  return new Promise((r) => {
    wake = r;
    chan.port2.postMessage(0);
  });
}

// --- state ---------------------------------------------------------------

interface Orbit {
  id: number;
  power: number;
  spec: Spec;
  sre: Uint8Array | null; // exact sign of Re(Z^2) per index (folded formulas)
  M: number;
  zr: Float64Array;
  zi: Float64Array;
  ze: Int32Array;
  /** Where the orbit stopped for want of room, so that a longer one of the same point can go on from there. */
  resume: Resume | null;
}
interface Resume {
  cx: bigint;
  cy: bigint;
  prec: number;
  zx: bigint;
  zy: bigint;
  svx: bigint;
  svy: bigint;
  svfx: number;
  svfy: number;
  svn: number;
  nextSave: number;
}
let cur: Orbit | null = null;
let latestRef = 0;

async function computeRef(req: RefRequest): Promise<void> {
  const t0 = performance.now();
  const { cx, cy, prec, spec } = req;
  const power = spec.kind === 'power' ? spec.power : 2;
  // below this a plain float32 Z would underflow once raised to power - 1
  const TINY = Math.pow(2, -Math.min(56, Math.floor(100 / (power - 1))));
  const cap = req.maxIter + 1;
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  // while the number fits a double's range it converts in one go, without a shifted copy
  const direct = prec <= 960;
  const INVP = Math.pow(2, -prec);
  const step = makeBigStep(spec, prec);
  // Mandelbrot-like: one orbit, of the critical point 0 with c = the reference point.
  // Julia: the orbit of the reference point itself, then the critical orbit, which
  // pixels move onto once they have drifted away from the first.
  const segs: bigint[][] = spec.julia ? [[cx, cy, req.jx, req.jy], [0n, 0n, req.jx, req.jy]] : [[0n, 0n, cx, cy]];
  const orbit = new Float32Array(Math.ceil((cap * segs.length) / TEXW) * TEXW * 4);
  const zr = new Float64Array(cap);
  const zi = new Float64Array(cap);
  const ze = new Int32Array(cap);
  const signRe = spec.kind === 'abs' && spec.absRe;
  const sre = signRe ? new Uint8Array(cap) : null;
  // folded formulas: each component again with its own exponent (see the shader's absStep)
  const comps = spec.kind === 'abs' ? new Float32Array(cap * segs.length * 4) : null;

  // A longer orbit of the point the last one was made for (the picture ran past its
  // end) goes on from where that one stopped: the first part is not iterated again.
  const prev = cur;
  const from =
    prev && prev.resume && !spec.julia && spec.kind === 'power' && prev.spec.kind === 'power' && !prev.spec.julia && prev.power === power &&
    prev.resume.cx === cx && prev.resume.cy === cy && prev.resume.prec === prec && cap > prev.M
      ? prev
      : null;
  const start = from ? from.M - 1 : 0;
  if (from) {
    zr.set(from.zr.subarray(0, start));
    zi.set(from.zi.subarray(0, start));
    ze.set(from.ze.subarray(0, start));
    for (let n = 0, o = 0; n < start; n++, o += 4) {
      orbit[o] = zr[n];
      orbit[o + 1] = zi[n];
      if (ze[n] !== 0) {
        orbit[o + 2] = ze[n];
        orbit[o + 3] = 1;
      }
    }
  }
  let resume: Resume | null = null;

  let total = 0;
  let lenA = 0;
  let escaped = false;
  let capIter = 1e9;
  let perA = 0;
  let perB = 0;
  let tick = performance.now();
  for (let si = 0; si < segs.length; si++) {
    let [zx, zy] = segs[si];
    const kx = segs[si][2];
    const ky = segs[si][3];
    let len = 0;
    let esc = false;
    // Brent's cycle detection in exact arithmetic: an orbit drawn into an attracting
    // cycle comes back onto itself to the last bits, and then need not be stored further
    let svx = zx;
    let svy = zy;
    let svn = 0;
    let nextSave = 1;
    let per = 0;
    // z as doubles, and the saved point likewise: they tell two points apart long
    // before the exact numbers need comparing
    let fzx = direct ? Number(zx) * INVP : Number(zx >> SH) * INV;
    let fzy = direct ? Number(zy) * INVP : Number(zy >> SH) * INV;
    let svfx = fzx;
    let svfy = fzy;
    if (from) {
      const r = from.resume!;
      zx = r.zx;
      zy = r.zy;
      svx = r.svx;
      svy = r.svy;
      svfx = r.svfx;
      svfy = r.svfy;
      svn = r.svn;
      nextSave = r.nextSave;
      fzx = direct ? Number(zx) * INVP : Number(zx >> SH) * INV;
      fzy = direct ? Number(zy) * INVP : Number(zy >> SH) * INV;
    }
    for (let n = start; n < cap; n++) {
      let x = fzx;
      let y = fzy;
      const o = (total + n) * 4;
      if (comps) {
        comp(zx, x, prec);
        comps[o] = fm;
        comps[o + 1] = fx;
        if (Math.abs(x) < 1e-6) x = fm * pow2(fx);
        comp(zy, y, prec);
        comps[o + 2] = fm;
        comps[o + 3] = fx;
        if (Math.abs(y) < 1e-6) y = fm * pow2(fx);
      }
      if (Math.abs(x) < TINY && Math.abs(y) < TINY) {
        // too small for a plain float: store mantissa + exponent
        if (zx === 0n && zy === 0n) {
          orbit[o + 2] = ZERO_E;
          if (si === 0) ze[n] = ZERO_E;
        } else {
          const s2 = Math.max(0, Math.max(bitLen(zx), bitLen(zy)) - 60);
          normz(Number(zx >> BigInt(s2)), Number(zy >> BigInt(s2)), s2 - prec);
          orbit[o] = nx;
          orbit[o + 1] = ny;
          orbit[o + 2] = ne;
          if (si === 0) {
            zr[n] = nx;
            zi[n] = ny;
            ze[n] = ne;
          }
        }
        orbit[o + 3] = 1;
      } else {
        orbit[o] = x;
        orbit[o + 1] = y;
        // exact sign of Re(Z^2): float32 cannot be trusted with it when |x| ~ |y|
        if (signRe) {
          const neg = (zx < 0n ? -zx : zx) < (zy < 0n ? -zy : zy) ? 1 : 0;
          orbit[o + 2] = neg;
          if (si === 0) sre![n] = neg;
        }
        if (si === 0) {
          zr[n] = x;
          zi[n] = y;
        }
      }
      len = n + 1;
      if (x * x + y * y > R2) {
        esc = true;
        break;
      }
      if (per || n === cap - 1) break;
      step(zx, zy, kx, ky);
      zx = sx;
      zy = sy;
      fzx = direct ? Number(zx) * INVP : Number(zx >> SH) * INV;
      fzy = direct ? Number(zy) * INVP : Number(zy >> SH) * INV;
      {
        let hit = false;
        if (Math.abs(fzx - svfx) <= 1e-12 * (1 + Math.abs(fzx)) && Math.abs(fzy - svfy) <= 1e-12 * (1 + Math.abs(fzy))) {
          const ddx = zx - svx;
          const ddy = zy - svy;
          hit = (ddx < 0n ? -ddx : ddx) < CYCLE_TOL && (ddy < 0n ? -ddy : ddy) < CYCLE_TOL;
        }
        if (hit) per = n + 1 - svn; // one more entry, then stop
        else if (n + 1 === nextSave) {
          svx = zx;
          svy = zy;
          svfx = fzx;
          svfy = fzy;
          svn = n + 1;
          nextSave *= 2;
        }
      }
      if ((n & 255) === 0 && performance.now() - tick > 30) {
        await breathe();
        if (req.id !== latestRef) return;
        tick = performance.now();
      }
    }
    if (si === 0) {
      lenA = len;
      escaped = esc;
      perA = per;
    }
    perB = per;
    if (!esc && !per) capIter = Math.min(capIter, len - 1);
    // stopped for want of room, on the last entry made: a longer orbit can take it from here
    if (!esc && !per && segs.length === 1 && spec.kind === 'power') resume = { cx, cy, prec, zx, zy, svx, svy, svfx, svfy, svn, nextSave };
    total += len;
  }
  // attracting, not just periodic: the product of the derivatives round the cycle is below 1
  let interior = false;
  if (perA > 0) {
    let lg = 0;
    for (let n = lenA - 1 - perA; n < lenA - 1; n++) lg += Math.log2(power) + (power - 1) * (ze[n] + Math.log2(Math.hypot(zr[n], zi[n])));
    interior = lg < -0.01;
  }

  cur = { id: req.id, power, spec, sre, M: lenA, zr, zi, ze, resume };
  const msOrbit = performance.now() - t0;
  // a Julia pixel has no dc term, so the table never has to allow for one
  const bla = await buildAny(cur, spec.julia ? -Infinity : req.lcmax);
  if (req.id !== latestRef) return;
  if (spec.julia) bla.lcmax = Infinity;
  let packed: Float32Array;
  let orbitRows: number;
  if (comps) {
    // the component table follows the orbit in the same texture
    orbitRows = Math.ceil((2 * total) / TEXW);
    packed = new Float32Array(orbitRows * TEXW * 4);
    packed.set(orbit.subarray(0, total * 4));
    packed.set(comps.subarray(0, total * 4), total * 4);
  } else {
    orbitRows = Math.ceil(total / TEXW);
    packed = orbit.subarray(0, orbitRows * TEXW * 4);
  }
  const out: RefResult = {
    type: 'ref',
    id: req.id,
    cx,
    cy,
    prec,
    maxIter: req.maxIter,
    spec,
    vx: req.vx,
    vy: req.vy,
    tag: req.tag,
    M: total,
    lenA,
    base: spec.julia ? lenA : 0,
    cap: capIter,
    escaped,
    perA,
    perB,
    interior,
    uOff: comps ? total : 0,
    orbit: packed,
    orbitRows,
    ms: performance.now() - t0,
    msOrbit,
    ...bla,
  };
  const transfer: Transferable[] = [packed.buffer];
  if (bla.blaA) transfer.push(bla.blaA.buffer, bla.blaE!.buffer);
  if (bla.blaB) transfer.push(bla.blaB.buffer);
  (self as unknown as Worker).postMessage(out, transfer);
}

// Custom formulas: one orbit of the reference point, no BLA. A converging map
// ends its orbit on the fixed point; the shader then holds the reference there.
async function computeCustom(req: RefRequest): Promise<void> {
  const t0 = performance.now();
  const { prec, spec } = req;
  let { cx, cy } = req;
  const f = makeCustom(spec, prec);
  const info = f.info;
  const cap = req.maxIter + 1;
  if (spec.custom === NEWTON) {
    // the map has a pole at 0: a reference there would be useless, so step off it (staying in view)
    const thr = Math.pow(2, req.lcmax - 8);
    if (Math.abs(f.toD(cx)) < thr && Math.abs(f.toD(cy)) < thr) cx += 1n << BigInt(Math.max(0, prec + Math.floor(req.lcmax) - 3));
  }
  const c: [bigint, bigint] = [cx, cy];
  // the orbit of the reference point, then (for some maps) the orbit of the critical
  // state, which pixels restart on once they have left the first
  const starts = [f.start(c)];
  if (info.second) starts.push({ z: [0n, 0n], u: [0n, 0n] });
  const zs = new Float32Array(cap * starts.length * 4);
  const us = info.hasU ? new Float32Array(cap * starts.length * 4) : null;
  const TINY = Math.pow(2, -56);
  const HUGE = Math.pow(2, 56);
  let total = 0;
  let lenA = 0;
  let escaped = false;
  let capIter = 1e9;
  let perA = 0;
  let perB = 0;
  let tick = performance.now();
  for (let si = 0; si < starts.length; si++) {
    const s = starts[si];
    let len = 0;
    let esc = false;
    let sv = { z: s.z, u: s.u };
    let svn = 0;
    let nextSave = 1;
    let per = 0;
    const near = (a: bigint, b: bigint) => {
      const d = a - b;
      return (d < 0n ? -d : d) < CYCLE_TOL;
    };
    for (let n = 0; n < cap; n++) {
      const [zx, zy] = s.z;
      const x = f.toD(zx);
      const y = f.toD(zy);
      const o = (total + n) * 4;
      // outside comfortable float range either way: store mantissa + exponent
      if ((Math.abs(x) < TINY && Math.abs(y) < TINY) || Math.abs(x) > HUGE || Math.abs(y) > HUGE) {
        if (zx === 0n && zy === 0n) zs[o + 2] = ZERO_E;
        else {
          const s2 = Math.max(0, Math.max(bitLen(zx), bitLen(zy)) - 60);
          normz(Number(zx >> BigInt(s2)), Number(zy >> BigInt(s2)), s2 - prec);
          zs[o] = nx;
          zs[o + 1] = ny;
          zs[o + 2] = ne;
        }
        zs[o + 3] = 1;
      } else {
        zs[o] = x;
        zs[o + 1] = y;
      }
      if (us) {
        if (f.refUf) us.set(f.refUf(s, c), (total + n) * 4);
        else {
          const u = f.refU(s, c);
          us[(total + n) * 4] = f.toD(u[0]);
          us[(total + n) * 4 + 1] = f.toD(u[1]);
        }
      }
      len = n + 1;
      if (f.escaped ? f.escaped(s, c) : !(x * x + y * y <= info.bail)) {
        esc = true;
        break;
      }
      if (per || n === cap - 1) break;
      f.step(s, c);
      // the whole state has to come back, not just z
      if (near(s.z[0], sv.z[0]) && near(s.z[1], sv.z[1]) && near(s.u[0], sv.u[0]) && near(s.u[1], sv.u[1])) per = n + 1 - svn;
      else if (n + 1 === nextSave) {
        sv = { z: s.z, u: s.u };
        svn = n + 1;
        nextSave *= 2;
      }
      if ((n & 63) === 0 && performance.now() - tick > 30) {
        await breathe();
        if (req.id !== latestRef) return;
        tick = performance.now();
      }
    }
    if (si === 0) {
      lenA = len;
      escaped = esc;
      perA = per;
    }
    perB = per;
    if (!esc && !per) capIter = Math.min(capIter, len - 1);
    total += len;
  }
  // texture layout: every z entry, then the second quantity for the same indices
  const entries = total * (us ? 2 : 1);
  const orbitRows = Math.ceil(entries / TEXW);
  const orbit = new Float32Array(orbitRows * TEXW * 4);
  orbit.set(zs.subarray(0, total * 4));
  if (us) orbit.set(us.subarray(0, total * 4), total * 4);
  cur = null;
  const out: RefResult = {
    type: 'ref',
    id: req.id,
    cx,
    cy,
    prec,
    maxIter: req.maxIter,
    spec,
    vx: cx,
    vy: cy,
    tag: req.tag,
    M: total,
    lenA,
    base: info.second ? lenA : 0,
    cap: capIter,
    escaped,
    perA,
    perB,
    interior: perA > 0,
    uOff: us ? total : 0,
    orbit,
    orbitRows,
    ms: performance.now() - t0,
    blaA: null,
    blaB: null,
    blaE: null,
    blaRows: 0,
    blaOff: new Int32Array(MAX_LEVELS),
    blaCnt: new Int32Array(MAX_LEVELS),
    blaLevels: 0,
    lcmax: 0,
    lrMax: -1e30,
  };
  (self as unknown as Worker).postMessage(out, [orbit.buffer]);
}

// BLA: for reference index m the map d_m -> d_{m+l} is approximated by
//   d' = A*d + B*dc      valid while |d| < r
// Level 0 is a single iteration (A = p*Z^(p-1), B = 1, r ~ eps*|Z|); each higher
// level merges two neighbours. A, B are float-exp, r is kept as log2.
// --- BLA tables, on several threads when the orbit is long --------------------------

// The threads are made by the page and handed over as message ports (see main.ts): a
// worker cannot always start workers of its own - not from inside the desktop app.
let pool: MessagePort[] | null = null;
let jobSeq = 0;
const waiting = new Map<number, { done: (r: BlaJobResult) => void; fail: () => void }>();

function setPool(ports: MessagePort[]): void {
  pool = ports;
  for (const port of ports) {
    port.onmessage = (ev: MessageEvent<BlaJobResult>) => {
      const p = waiting.get(ev.data.id);
      waiting.delete(ev.data.id);
      if (p) p.done(ev.data);
    };
    // a small table to begin with, so that the thread is loaded and warm when a real one comes
    const n = 4096;
    const job: BlaJob = { id: ++jobSeq, P: { kind: 'power', pw: 2, absRe: false, imMode: 0, imSign: 1, lcmax: -40 }, zr: new Float64Array(n).fill(0.7), zi: new Float64Array(n).fill(0.3), ze: new Int32Array(n), sre: null, n, upTo: 10 };
    port.postMessage(job);
  }
}

/** The threads have not answered in time: do without them from now on. */
function dropPool(): void {
  pool = null;
  for (const p of waiting.values()) p.fail();
  waiting.clear();
}

function params(o: Orbit, lcmax: number): BlaParams {
  const s = o.spec;
  return { kind: s.kind === 'abs' ? 'abs' : 'power', pw: o.power, absRe: s.absRe, imMode: s.imMode, imSign: s.imSign, lcmax };
}

async function buildAny(o: Orbit, lcmax: number): Promise<BlaData> {
  const blaOff = new Int32Array(MAX_LEVELS);
  const blaCnt = new Int32Array(MAX_LEVELS);
  const none: BlaData = { blaA: null, blaB: null, blaE: null, blaRows: 0, blaOff, blaCnt, blaLevels: 0, lcmax, lrMax: -1e30 };
  const n0 = o.M - 2;
  if (n0 < 1 << (LMIN + 1)) return none;

  const sizes: number[] = [n0];
  while (sizes[sizes.length - 1] >> 1 >= 1) sizes.push(sizes[sizes.length - 1] >> 1);
  const top = Math.min(sizes.length - 1, LMIN + MAX_LEVELS - 1);
  let total = 0;
  for (let k = LMIN; k <= top; k++) {
    blaOff[k - LMIN] = total;
    blaCnt[k - LMIN] = sizes[k];
    total += sizes[k];
  }
  const rows = Math.ceil(total / TEXW);
  const P = params(o, lcmax);
  const real = P.kind === 'abs';
  const outA = new Float32Array(rows * TEXW * 4);
  const outB = real ? new Float32Array(rows * TEXW * 4) : null;
  const outE = new Float32Array(rows * TEXW * 4);
  let lrMax = -1e30;
  // a stretch's levels go where they belong: entry (first step >> k) of level k
  const place = (levels: BlaLevel[], first: number): void => {
    for (const l of levels) {
      if (l.k > top) continue;
      const at = (blaOff[l.k - LMIN] + (first >> l.k)) * 4;
      outA.set(l.A, at);
      if (outB && l.B) outB.set(l.B, at);
      outE.set(l.E, at);
    }
  };
  // entry j of level 0 is the step at reference index j + 1
  const { zr, zi, ze, sre } = o;
  const workers = n0 >= 200000 ? pool : null;
  let done = false;
  if (workers) {
    // stretches of 2^C steps, several to a thread so that none waits long for the slowest
    const C = Math.max(LMIN, Math.min(17, top, Math.floor(Math.log2(n0 / (8 * workers.length)))));
    const size = 1 << C;
    const count = Math.ceil(n0 / size);
    try {
      const results = await new Promise<BlaJobResult[]>((resolve, reject) => {
        const res: BlaJobResult[] = new Array(count);
        let next = 0;
        let left = count;
        const feed = (w: MessagePort): void => {
          if (next >= count) return;
          const c = next++;
          const first = c * size;
          const n = Math.min(size, n0 - first);
          const job: BlaJob = {
            id: ++jobSeq,
            P,
            zr: zr.slice(first + 1, first + 1 + n),
            zi: zi.slice(first + 1, first + 1 + n),
            ze: ze.slice(first + 1, first + 1 + n),
            sre: sre ? sre.slice(first + 1, first + 1 + n) : null,
            n,
            upTo: C,
          };
          waiting.set(job.id, {
            done: (r) => {
              res[c] = r;
              if (--left === 0) resolve(res);
              else feed(w);
            },
            fail: () => reject(new Error('table thread failed')),
          });
          const transfer: Transferable[] = [job.zr.buffer, job.zi.buffer, job.ze.buffer];
          if (job.sre) transfer.push(job.sre.buffer);
          w.postMessage(job, transfer);
        };
        for (const w of workers) feed(w);
        // a thread that died says nothing: do not wait for it for ever
        setTimeout(() => {
          if (left > 0) dropPool();
        }, 15000 + n0 / 400);
      });
      const tops: BlaState[] = [];
      results.forEach((r, c) => {
        place(r.levels, c * size);
        tops.push(r.top);
        if (r.lrMax > lrMax) lrMax = r.lrMax;
      });
      // the few levels above the stretches, from their tops
      let st = concat(tops, real ? 4 : 2);
      for (let k = C + 1; k <= top; k++) {
        st = merge(P, st);
        const e = emit(P, k, st);
        place([e.level], 0);
        if (e.lrMax > lrMax) lrMax = e.lrMax;
      }
      done = true;
    } catch {
      lrMax = -1e30; // fall through and build it all here
    }
  }
  if (!done) {
    const r = blaRange(P, zr.subarray(1, n0 + 1), zi.subarray(1, n0 + 1), ze.subarray(1, n0 + 1), sre ? sre.subarray(1, n0 + 1) : null, n0, top);
    place(r.levels, 0);
    lrMax = r.lrMax;
  }
  return { blaA: outA, blaB: outB, blaE: outE, blaRows: rows, blaOff, blaCnt, blaLevels: top - LMIN + 1, lcmax, lrMax };
}

self.onmessage = (ev: MessageEvent<RefRequest | BlaRequest | PoolMessage>) => {
  const msg = ev.data;
  if (msg.type === 'pool') {
    setPool(msg.ports);
  } else if (msg.type === 'ref') {
    latestRef = msg.id;
    Promise.resolve()
      .then(() => (msg.spec.kind === 'custom' ? computeCustom(msg) : computeRef(msg)))
      .catch(() => (self as unknown as Worker).postMessage({ type: 'fail', id: msg.id }));
  } else if (cur && cur.id === msg.id) {
    void buildAny(cur, msg.lcmax).then((bla) => {
      const out: BlaResult = { type: 'bla', id: msg.id, ...bla };
      const transfer: Transferable[] = bla.blaA ? [bla.blaA.buffer, bla.blaE!.buffer] : [];
      if (bla.blaB) transfer.push(bla.blaB.buffer);
      (self as unknown as Worker).postMessage(out, transfer);
    });
  }
};

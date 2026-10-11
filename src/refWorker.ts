// Reference orbit worker.
// Computes one arbitrary-precision orbit Z_n for the reference point and the
// bilinear-approximation (BLA) table that lets the shader skip many iterations
// at once. Everything the GPU needs is packed into float32 textures here.

import { Spec, makeBigStep, sx, sy } from './formulas';
import { makeCustom, NEWTON } from './custom';

export const R2 = 1.0e6; // bailout radius squared, must match the shader
export const TEXW = 2048;
export const LMIN = 2; // lowest BLA level kept (skips 2^LMIN iterations)
export const MAX_LEVELS = 24;
// above this float32 overflows near the bailout radius
export const MAX_POWER = 10;
const ZERO_E = -10000000;
export const EPS_L2 = -16; // log2 of the BLA linearisation tolerance
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
}
export interface BlaResult extends BlaData {
  type: 'bla';
  id: number;
}

/** The orbit could not be made: in practice, no memory for one this long. */
export interface FailResult {
  type: 'fail';
  id: number;
}

// --- float helpers -------------------------------------------------------

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
/** e such that x = m * 2^e with m in [0.5, 1) (x normal, non-zero). */
function expOf(x: number): number {
  f64[0] = x;
  return ((u32[1] >>> 20) & 0x7ff) - 1022;
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
    for (let n = 0; n < cap; n++) {
      let x = Number(zx >> SH) * INV;
      let y = Number(zy >> SH) * INV;
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
      {
        const ddx = zx - svx;
        const ddy = zy - svy;
        if ((ddx < 0n ? -ddx : ddx) < CYCLE_TOL && (ddy < 0n ? -ddy : ddy) < CYCLE_TOL) per = n + 1 - svn; // one more entry, then stop
        else if (n + 1 === nextSave) {
          svx = zx;
          svy = zy;
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
    total += len;
  }
  // attracting, not just periodic: the product of the derivatives round the cycle is below 1
  let interior = false;
  if (perA > 0) {
    let lg = 0;
    for (let n = lenA - 1 - perA; n < lenA - 1; n++) lg += Math.log2(power) + (power - 1) * (ze[n] + Math.log2(Math.hypot(zr[n], zi[n])));
    interior = lg < -0.01;
  }

  cur = { id: req.id, power, spec, sre, M: lenA, zr, zi, ze };
  // a Julia pixel has no dc term, so the table never has to allow for one
  const bla = buildAny(cur, spec.julia ? -Infinity : req.lcmax);
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
function buildBla(o: Orbit, lcmax: number): BlaData {
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
  const outA = new Float32Array(rows * TEXW * 4);
  const outE = new Float32Array(rows * TEXW * 4);

  const { zr, zi, ze } = o;
  const pw = o.power;
  const lpw = Math.log2(pw);
  const lrk = Math.log2(Math.min(1, 2 / (pw - 1))); // the neglected term grows with the power
  let lrMax = -1e30;
  let n = n0;
  let ax = new Float64Array(n);
  let ay = new Float64Array(n);
  let bx = new Float64Array(n);
  let by = new Float64Array(n);
  let ae = new Int32Array(n);
  let be = new Int32Array(n);
  let la = new Float64Array(n);
  let lb = new Float64Array(n);
  let lr = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const m = j + 1;
    normz(zr[m], zi[m], ze[m]);
    const zx = nx;
    const zy = ny;
    const lz = nx === 0 && ny === 0 ? -Infinity : ne + Math.log2(Math.hypot(nx, ny));
    // A = p * Z^(p-1)
    const zeN = ne;
    for (let i = 2; i < pw; i++) normz(nx * zx - ny * zy, nx * zy + ny * zx, ne === ZERO_E ? ZERO_E : ne + zeN);
    normz(nx * pw, ny * pw, ne);
    ax[j] = nx;
    ay[j] = ny;
    ae[j] = ne;
    bx[j] = 0.5;
    be[j] = 1;
    la[j] = lpw + (pw - 1) * lz;
    lr[j] = EPS_L2 + lz + lrk;
  }

  for (let k = 1; k <= top; k++) {
    const nk = sizes[k];
    const cax = new Float64Array(nk);
    const cay = new Float64Array(nk);
    const cbx = new Float64Array(nk);
    const cby = new Float64Array(nk);
    const cae = new Int32Array(nk);
    const cbe = new Int32Array(nk);
    const cla = new Float64Array(nk);
    const clb = new Float64Array(nk);
    const clr = new Float64Array(nk);
    for (let j = 0; j < nk; j++) {
      const x = 2 * j;
      const y = x + 1;
      // A = Ay * Ax
      normz(ay_re(ax, ay, x, y), ay_im(ax, ay, x, y), ae[x] + ae[y]);
      cax[j] = nx;
      cay[j] = ny;
      cae[j] = ne;
      cla[j] = la[x] + la[y];
      // B = Ay * Bx + By
      const tx = ax[y] * bx[x] - ay[y] * by[x];
      const ty = ax[y] * by[x] + ay[y] * bx[x];
      const te = ae[y] + be[x];
      const en = te > be[y] ? te : be[y];
      const st = pow2(te - en);
      const sb = pow2(be[y] - en);
      normz(tx * st + bx[y] * sb, ty * st + by[y] * sb, en);
      cbx[j] = nx;
      cby[j] = ny;
      cbe[j] = ne;
      clb[j] = nx === 0 && ny === 0 ? -Infinity : ne + Math.log2(Math.hypot(nx, ny));
      // r = min(rx, (ry - |Bx|*cmax) / |Ax|)
      const t = lb[x] + lcmax;
      const ry = lr[y] > t ? lr[y] + Math.log2(1 - Math.pow(2, t - lr[y])) : -Infinity;
      const r2 = ry - la[x];
      clr[j] = lr[x] < r2 ? lr[x] : r2;
    }
    ax = cax;
    ay = cay;
    bx = cbx;
    by = cby;
    ae = cae;
    be = cbe;
    la = cla;
    lb = clb;
    lr = clr;
    if (k >= LMIN) {
      let p = blaOff[k - LMIN] * 4;
      for (let j = 0; j < nk; j++, p += 4) {
        outA[p] = ax[j];
        outA[p + 1] = ay[j];
        outA[p + 2] = bx[j];
        outA[p + 3] = by[j];
        outE[p] = ae[j];
        outE[p + 1] = be[j];
        const r = lr[j];
        outE[p + 2] = r > -1e30 ? r : -1e30;
        if (r > lrMax) lrMax = r;
      }
    }
  }
  return { blaA: outA, blaB: null, blaE: outE, blaRows: rows, blaOff, blaCnt, blaLevels: top - LMIN + 1, lcmax, lrMax };
}

function buildAny(o: Orbit, lcmax: number): BlaData {
  return o.spec.kind === 'abs' ? buildBlaReal(o, lcmax) : buildBla(o, lcmax);
}

// 2x2 real matrices with a shared exponent, four mantissas at index 4*i
let me = 0; // exponent register of norm4
function norm4(a: Float64Array, i: number, e: number): void {
  const p = 4 * i;
  let m = Math.abs(a[p]);
  for (let k = 1; k < 4; k++) if (Math.abs(a[p + k]) > m) m = Math.abs(a[p + k]);
  if (m === 0) {
    me = ZERO_E;
    return;
  }
  const k = expOf(m);
  const s = pow2(-k);
  a[p] *= s;
  a[p + 1] *= s;
  a[p + 2] *= s;
  a[p + 3] *= s;
  me = e + k;
}
function frob(a: Float64Array, i: number): number {
  const p = 4 * i;
  return Math.sqrt(a[p] * a[p] + a[p + 1] * a[p + 1] + a[p + 2] * a[p + 2] + a[p + 3] * a[p + 3]);
}

// BLA for the folded quadratics. Between folds the map is an ordinary quadratic,
// so one step is d' = J d + dc with J the Jacobian at Z - a real 2x2 matrix, since
// the folds break complex linearity. It holds while d is small against |Z| (the
// d^2 term) and smaller than the distance to the nearest fold line (no sign flips).
function buildBlaReal(o: Orbit, lcmax: number): BlaData {
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
  const outA = new Float32Array(rows * TEXW * 4);
  const outB = new Float32Array(rows * TEXW * 4);
  const outE = new Float32Array(rows * TEXW * 4);

  const { zr, zi, ze, sre } = o;
  const { absRe, imMode, imSign } = o.spec;
  let lrMax = -1e30;
  let n = n0;
  let A = new Float64Array(4 * n);
  let B = new Float64Array(4 * n);
  let ae = new Int32Array(n);
  let be = new Int32Array(n);
  let la = new Float64Array(n);
  let lb = new Float64Array(n);
  let lr = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const m = j + 1;
    const p = 4 * j;
    B[p] = 0.5;
    B[p + 3] = 0.5;
    be[j] = 1;
    lb[j] = 0.5; // log2 of the Frobenius norm of the identity
    if (ze[m] !== 0) {
      // Z too small to be a plain float: the pixel restarts from the origin here anyway
      ae[j] = ZERO_E;
      la[j] = -Infinity;
      lr[j] = -Infinity;
      continue;
    }
    const X = zr[m];
    const Y = zi[m];
    const s1 = absRe && sre![m] ? -1 : 1;
    A[p] = s1 * 2 * X;
    A[p + 1] = -s1 * 2 * Y;
    let g0: number;
    let g1: number;
    let fold = Infinity;
    if (imMode === 0) {
      g0 = Y;
      g1 = X;
    } else if (imMode === 1) {
      const sg = X * Y < 0 ? -1 : 1;
      g0 = sg * Y;
      g1 = sg * X;
      fold = Math.min(Math.abs(X), Math.abs(Y));
    } else if (imMode === 2) {
      g0 = X < 0 ? -Y : Y;
      g1 = Math.abs(X);
      fold = Math.abs(X);
    } else {
      g0 = Math.abs(Y);
      g1 = Y < 0 ? -X : X;
      fold = Math.abs(Y);
    }
    if (absRe) fold = Math.min(fold, Math.abs(Math.abs(X) - Math.abs(Y)) * Math.SQRT1_2);
    A[p + 2] = 2 * imSign * g0;
    A[p + 3] = 2 * imSign * g1;
    norm4(A, j, 0);
    ae[j] = me;
    la[j] = me === ZERO_E ? -Infinity : me + Math.log2(frob(A, j));
    lr[j] = Math.min(EPS_L2 + Math.log2(Math.hypot(X, Y)), Math.log2(fold) - 1);
  }

  for (let k = 1; k <= top; k++) {
    const nk = sizes[k];
    const cA = new Float64Array(4 * nk);
    const cB = new Float64Array(4 * nk);
    const cae = new Int32Array(nk);
    const cbe = new Int32Array(nk);
    const cla = new Float64Array(nk);
    const clb = new Float64Array(nk);
    const clr = new Float64Array(nk);
    for (let j = 0; j < nk; j++) {
      const x = 8 * j; // first of the pair, at 4*(2j)
      const y = x + 4;
      const q = 4 * j;
      const ix = 2 * j;
      const iy = ix + 1;
      // A = Ay * Ax
      cA[q] = A[y] * A[x] + A[y + 1] * A[x + 2];
      cA[q + 1] = A[y] * A[x + 1] + A[y + 1] * A[x + 3];
      cA[q + 2] = A[y + 2] * A[x] + A[y + 3] * A[x + 2];
      cA[q + 3] = A[y + 2] * A[x + 1] + A[y + 3] * A[x + 3];
      norm4(cA, j, ae[ix] + ae[iy]);
      cae[j] = me;
      cla[j] = me === ZERO_E ? -Infinity : me + Math.log2(frob(cA, j));
      // B = Ay * Bx + By
      const te = ae[iy] + be[ix];
      const en = te > be[iy] ? te : be[iy];
      const st = pow2(te - en);
      const sb = pow2(be[iy] - en);
      cB[q] = (A[y] * B[x] + A[y + 1] * B[x + 2]) * st + B[y] * sb;
      cB[q + 1] = (A[y] * B[x + 1] + A[y + 1] * B[x + 3]) * st + B[y + 1] * sb;
      cB[q + 2] = (A[y + 2] * B[x] + A[y + 3] * B[x + 2]) * st + B[y + 2] * sb;
      cB[q + 3] = (A[y + 2] * B[x + 1] + A[y + 3] * B[x + 3]) * st + B[y + 3] * sb;
      norm4(cB, j, en);
      cbe[j] = me;
      clb[j] = me === ZERO_E ? -Infinity : me + Math.log2(frob(cB, j));
      // r = min(rx, (ry - |Bx| cmax) / |Ax|), with norms that can only overestimate
      const t = lb[ix] + lcmax;
      const ry = lr[iy] > t ? lr[iy] + Math.log2(1 - Math.pow(2, t - lr[iy])) : -Infinity;
      const r2 = ry - la[ix];
      clr[j] = lr[ix] < r2 ? lr[ix] : r2;
    }
    A = cA;
    B = cB;
    ae = cae;
    be = cbe;
    la = cla;
    lb = clb;
    lr = clr;
    if (k >= LMIN) {
      let p = blaOff[k - LMIN] * 4;
      for (let j = 0; j < nk; j++, p += 4) {
        const q = 4 * j;
        outA[p] = A[q];
        outA[p + 1] = A[q + 1];
        outA[p + 2] = A[q + 2];
        outA[p + 3] = A[q + 3];
        outB[p] = B[q];
        outB[p + 1] = B[q + 1];
        outB[p + 2] = B[q + 2];
        outB[p + 3] = B[q + 3];
        outE[p] = ae[j];
        outE[p + 1] = be[j];
        const r = lr[j];
        outE[p + 2] = r > -1e30 ? r : -1e30;
        if (r > lrMax) lrMax = r;
      }
    }
  }
  return { blaA: outA, blaB: outB, blaE: outE, blaRows: rows, blaOff, blaCnt, blaLevels: top - LMIN + 1, lcmax, lrMax };
}

function ay_re(ax: Float64Array, ay: Float64Array, x: number, y: number): number {
  return ax[y] * ax[x] - ay[y] * ay[x];
}
function ay_im(ax: Float64Array, ay: Float64Array, x: number, y: number): number {
  return ax[y] * ay[x] + ay[y] * ax[x];
}

self.onmessage = (ev: MessageEvent<RefRequest | BlaRequest>) => {
  const msg = ev.data;
  if (msg.type === 'ref') {
    latestRef = msg.id;
    Promise.resolve()
      .then(() => (msg.spec.kind === 'custom' ? computeCustom(msg) : computeRef(msg)))
      .catch(() => (self as unknown as Worker).postMessage({ type: 'fail', id: msg.id }));
  } else if (cur && cur.id === msg.id) {
    const bla = buildAny(cur, msg.lcmax);
    const out: BlaResult = { type: 'bla', id: msg.id, ...bla };
    const transfer: Transferable[] = bla.blaA ? [bla.blaA.buffer, bla.blaE!.buffer] : [];
    if (bla.blaB) transfer.push(bla.blaB.buffer);
    (self as unknown as Worker).postMessage(out, transfer);
  }
};

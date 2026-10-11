// Bilinear approximation (BLA) tables: the arithmetic, apart from who runs it.
//
// One entry lets a pixel skip 2^k iterations of the reference orbit at once:
// d' = A d + B dc, valid while |d| stays under a radius r. Level 0 is one step;
// an entry of level k merges two neighbours of level k - 1. Entries of different
// stretches of the orbit do not depend on each other, so the orbit is cut into
// stretches of 2^C steps that are built separately - on several threads when the
// orbit is long (see blaWorker.ts) - and only the few levels above C are merged
// from their tops afterwards. Either way the numbers come out the same.

export const LMIN = 2; // lowest BLA level kept (skips 2^LMIN iterations)
export const MAX_LEVELS = 24;
export const EPS_L2 = -16; // log2 of the BLA linearisation tolerance
export const ZERO_E = -10000000;

export interface BlaParams {
  kind: 'power' | 'abs';
  pw: number;
  absRe: boolean;
  imMode: number;
  imSign: number;
  lcmax: number;
}

/**
 * The entries of one level over a stretch. `a` and `b` hold mantissas with a shared
 * exponent each: a complex number per entry (stride 2) for z^p + c, a real 2x2
 * matrix (stride 4) for the folded formulas.
 */
export interface BlaState {
  n: number;
  a: Float64Array;
  b: Float64Array;
  ae: Int32Array;
  be: Int32Array;
  la: Float64Array; // log2 |A|
  lb: Float64Array; // log2 |B|
  lr: Float64Array; // log2 of the radius
}

/** One level of a stretch as it goes into the textures. */
export interface BlaLevel {
  k: number;
  A: Float32Array;
  B: Float32Array | null;
  E: Float32Array;
}

export interface BlaRange {
  levels: BlaLevel[];
  top: BlaState;
  lrMax: number;
}

// The scratch number sits in the middle of a buffer of its own. Several threads each
// have one of these, written millions of times a second: left as eight bytes apiece
// they land side by side in memory, and the threads fight over the one cache line.
const f64 = new Float64Array(32);
const u32 = new Uint32Array(f64.buffer);
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

function blank(n: number, stride: number): BlaState {
  return {
    n,
    a: new Float64Array(stride * n),
    b: new Float64Array(stride * n),
    ae: new Int32Array(n),
    be: new Int32Array(n),
    la: new Float64Array(n),
    lb: new Float64Array(n),
    lr: new Float64Array(n),
  };
}

// Working space for the levels of a stretch, kept from one stretch to the next: the
// levels go back and forth between two of these. (Fresh arrays for every level would
// cost more in memory traffic than the arithmetic, and that part does not speed up
// with more threads.) A state's arrays may be longer than its n.
const work: BlaState[] = [blank(0, 2), blank(0, 2)];
function room(i: number, n: number, stride: number): BlaState {
  let w = work[i];
  if (w.ae.length < n || w.a.length < stride * n) {
    w = blank(Math.max(n, w.ae.length), stride);
    work[i] = w;
  }
  w.n = n;
  return w;
}

/** Level 0 over `n` steps: entry j is the step at index off + j of the arrays given. */
function level0(P: BlaParams, zr: Float64Array, zi: Float64Array, ze: Int32Array, sre: Uint8Array | null, off: number, n: number): BlaState {
  if (P.kind === 'abs') return level0Real(P, zr, zi, ze, sre, off, n);
  const st = room(0, n, 2);
  const { a, b, ae, be, la, lb, lr } = st;
  const pw = P.pw;
  const lpw = Math.log2(pw);
  const lrk = Math.log2(Math.min(1, 2 / (pw - 1))); // the neglected term grows with the power
  for (let j = 0; j < n; j++) {
    normz(zr[off + j], zi[off + j], ze[off + j]);
    const zx = nx;
    const zy = ny;
    const lz = nx === 0 && ny === 0 ? -Infinity : ne + Math.log2(Math.hypot(nx, ny));
    // A = p * Z^(p-1)
    const zeN = ne;
    for (let i = 2; i < pw; i++) normz(nx * zx - ny * zy, nx * zy + ny * zx, ne === ZERO_E ? ZERO_E : ne + zeN);
    normz(nx * pw, ny * pw, ne);
    a[2 * j] = nx;
    a[2 * j + 1] = ny;
    ae[j] = ne;
    b[2 * j] = 0.5;
    b[2 * j + 1] = 0;
    be[j] = 1;
    la[j] = lpw + (pw - 1) * lz;
    lb[j] = 0;
    lr[j] = EPS_L2 + lz + lrk;
  }
  return st;
}

// BLA for the folded quadratics. Between folds the map is an ordinary quadratic,
// so one step is d' = J d + dc with J the Jacobian at Z - a real 2x2 matrix, since
// the folds break complex linearity. It holds while d is small against |Z| (the
// d^2 term) and smaller than the distance to the nearest fold line (no sign flips).
function level0Real(P: BlaParams, zr: Float64Array, zi: Float64Array, ze: Int32Array, sre: Uint8Array | null, off: number, n: number): BlaState {
  const st = room(0, n, 4);
  const { a: A, b: B, ae, be, la, lb, lr } = st;
  const { absRe, imMode, imSign } = P;
  for (let j = 0; j < n; j++) {
    const p = 4 * j;
    B[p] = 0.5;
    B[p + 1] = 0;
    B[p + 2] = 0;
    B[p + 3] = 0.5;
    be[j] = 1;
    lb[j] = 0.5; // log2 of the Frobenius norm of the identity
    if (ze[off + j] !== 0) {
      // Z too small to be a plain float: the pixel restarts from the origin here anyway
      A[p] = 0;
      A[p + 1] = 0;
      A[p + 2] = 0;
      A[p + 3] = 0;
      ae[j] = ZERO_E;
      la[j] = -Infinity;
      lr[j] = -Infinity;
      continue;
    }
    const X = zr[off + j];
    const Y = zi[off + j];
    const s1 = absRe && sre![off + j] ? -1 : 1;
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
  return st;
}

/** The next level: entry j from entries 2j (first) and 2j + 1 (second) of this one. Written into `into` if given. */
export function merge(P: BlaParams, s: BlaState, into?: BlaState): BlaState {
  if (P.kind === 'abs') return mergeReal(P, s, into);
  const nk = s.n >> 1;
  const st = into || blank(nk, 2);
  st.n = nk;
  const { a, b, ae, be, la, lb, lr } = s;
  const lcmax = P.lcmax;
  for (let j = 0; j < nk; j++) {
    const x = 2 * j;
    const y = x + 1;
    const axr = a[2 * x];
    const axi = a[2 * x + 1];
    const ayr = a[2 * y];
    const ayi = a[2 * y + 1];
    const bxr = b[2 * x];
    const bxi = b[2 * x + 1];
    // A = Ay * Ax
    normz(ayr * axr - ayi * axi, ayr * axi + ayi * axr, ae[x] + ae[y]);
    st.a[2 * j] = nx;
    st.a[2 * j + 1] = ny;
    st.ae[j] = ne;
    st.la[j] = la[x] + la[y];
    // B = Ay * Bx + By
    const tx = ayr * bxr - ayi * bxi;
    const ty = ayr * bxi + ayi * bxr;
    const te = ae[y] + be[x];
    const en = te > be[y] ? te : be[y];
    const sa = pow2(te - en);
    const sb = pow2(be[y] - en);
    normz(tx * sa + b[2 * y] * sb, ty * sa + b[2 * y + 1] * sb, en);
    st.b[2 * j] = nx;
    st.b[2 * j + 1] = ny;
    st.be[j] = ne;
    st.lb[j] = nx === 0 && ny === 0 ? -Infinity : ne + Math.log2(Math.hypot(nx, ny));
    // r = min(rx, (ry - |Bx|*cmax) / |Ax|)
    const t = lb[x] + lcmax;
    const ry = lr[y] > t ? lr[y] + Math.log2(1 - Math.pow(2, t - lr[y])) : -Infinity;
    const r2 = ry - la[x];
    st.lr[j] = lr[x] < r2 ? lr[x] : r2;
  }
  return st;
}

function mergeReal(P: BlaParams, s: BlaState, into?: BlaState): BlaState {
  const nk = s.n >> 1;
  const st = into || blank(nk, 4);
  st.n = nk;
  const { a: A, b: B, ae, be, la, lb, lr } = s;
  const { a: cA, b: cB } = st;
  const lcmax = P.lcmax;
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
    st.ae[j] = me;
    st.la[j] = me === ZERO_E ? -Infinity : me + Math.log2(frob(cA, j));
    // B = Ay * Bx + By
    const te = ae[iy] + be[ix];
    const en = te > be[iy] ? te : be[iy];
    const sa = pow2(te - en);
    const sb = pow2(be[iy] - en);
    cB[q] = (A[y] * B[x] + A[y + 1] * B[x + 2]) * sa + B[y] * sb;
    cB[q + 1] = (A[y] * B[x + 1] + A[y + 1] * B[x + 3]) * sa + B[y + 1] * sb;
    cB[q + 2] = (A[y + 2] * B[x] + A[y + 3] * B[x + 2]) * sa + B[y + 2] * sb;
    cB[q + 3] = (A[y + 2] * B[x + 1] + A[y + 3] * B[x + 3]) * sa + B[y + 3] * sb;
    norm4(cB, j, en);
    st.be[j] = me;
    st.lb[j] = me === ZERO_E ? -Infinity : me + Math.log2(frob(cB, j));
    // r = min(rx, (ry - |Bx| cmax) / |Ax|), with norms that can only overestimate
    const t = lb[ix] + lcmax;
    const ry = lr[iy] > t ? lr[iy] + Math.log2(1 - Math.pow(2, t - lr[iy])) : -Infinity;
    const r2 = ry - la[ix];
    st.lr[j] = lr[ix] < r2 ? lr[ix] : r2;
  }
  return st;
}

/** Write a level as float32 texels from entry `at` on: four per entry in A (and B for matrices), exponents and radius in E. */
function emitInto(s: BlaState, l: BlaLevel, at: number): number {
  const n = s.n;
  const { A, B, E } = l;
  let lrMax = -1e30;
  if (B) {
    A.set(s.a.subarray(0, 4 * n), 4 * at);
    B.set(s.b.subarray(0, 4 * n), 4 * at);
  } else {
    for (let j = 0, p = 4 * at; j < n; j++, p += 4) {
      A[p] = s.a[2 * j];
      A[p + 1] = s.a[2 * j + 1];
      A[p + 2] = s.b[2 * j];
      A[p + 3] = s.b[2 * j + 1];
    }
  }
  for (let j = 0, p = 4 * at; j < n; j++, p += 4) {
    E[p] = s.ae[j];
    E[p + 1] = s.be[j];
    const r = s.lr[j];
    E[p + 2] = r > -1e30 ? r : -1e30;
    if (r > lrMax) lrMax = r;
  }
  return lrMax;
}

function newLevel(P: BlaParams, k: number, n: number): BlaLevel {
  return { k, A: new Float32Array(4 * n), B: P.kind === 'abs' ? new Float32Array(4 * n) : null, E: new Float32Array(4 * n) };
}

/** A level as float32 texels. */
export function emit(P: BlaParams, k: number, s: BlaState): { level: BlaLevel; lrMax: number } {
  const level = newLevel(P, k, s.n);
  return { level, lrMax: emitInto(s, level, 0) };
}

// A stretch is worked through in blocks of this many steps (2^BLOCK): a block's
// levels then stay within a core's own cache. Memory is what a table costs most,
// and the bus is shared: threads that all stream through it gain nothing.
const BLOCK = 11;

/**
 * Levels 1..upTo over a stretch of `n` steps whose reference values are the first
 * `n` of the arrays given. Returns the levels from LMIN on, ready for the textures,
 * and the state of level `upTo`, from which the levels above can be merged.
 */
export function blaRange(P: BlaParams, zr: Float64Array, zi: Float64Array, ze: Int32Array, sre: Uint8Array | null, n: number, upTo: number): BlaRange {
  const stride = P.kind === 'abs' ? 4 : 2;
  const levels: BlaLevel[] = [];
  for (let k = LMIN; k <= upTo; k++) levels.push(newLevel(P, k, n >> k));
  let lrMax = -1e30;
  const low = Math.min(upTo, BLOCK);
  const size = 1 << low;
  // the top of each block: one entry of level `low` (none from a block cut short)
  const tops = blank(n >> low, stride);
  for (let first = 0; first < n; first += size) {
    let st = level0(P, zr, zi, ze, sre, first, Math.min(size, n - first));
    for (let k = 1; k <= low; k++) {
      st = merge(P, st, room(k & 1, st.n >> 1, stride));
      if (k >= LMIN) {
        const r = emitInto(st, levels[k - LMIN], first >> k);
        if (r > lrMax) lrMax = r;
      }
    }
    if (st.n > 0) {
      const j = first >> low;
      for (let q = 0; q < stride; q++) {
        tops.a[stride * j + q] = st.a[q];
        tops.b[stride * j + q] = st.b[q];
      }
      tops.ae[j] = st.ae[0];
      tops.be[j] = st.be[0];
      tops.la[j] = st.la[0];
      tops.lb[j] = st.lb[0];
      tops.lr[j] = st.lr[0];
    }
  }
  // the levels above the blocks, from their tops: few entries by now
  let st = tops;
  for (let k = low + 1; k <= upTo; k++) {
    st = merge(P, st);
    const r = emitInto(st, levels[k - LMIN], 0);
    if (r > lrMax) lrMax = r;
  }
  return { levels, top: st, lrMax };
}

/** Several states of one level, end to end. */
export function concat(parts: BlaState[], stride: number): BlaState {
  let n = 0;
  for (const p of parts) n += p.n;
  const st = blank(n, stride);
  let at = 0;
  for (const p of parts) {
    st.a.set(p.a.subarray(0, stride * p.n), stride * at);
    st.b.set(p.b.subarray(0, stride * p.n), stride * at);
    st.ae.set(p.ae.subarray(0, p.n), at);
    st.be.set(p.be.subarray(0, p.n), at);
    st.la.set(p.la.subarray(0, p.n), at);
    st.lb.set(p.lb.subarray(0, p.n), at);
    st.lr.set(p.lr.subarray(0, p.n), at);
    at += p.n;
  }
  return st;
}

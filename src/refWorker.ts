// Reference orbit worker.
// Computes one arbitrary-precision orbit Z_n for the reference point and the
// bilinear-approximation (BLA) table that lets the shader skip many iterations
// at once. Everything the GPU needs is packed into float32 textures here.

export const R2 = 1.0e6; // bailout radius squared, must match the shader
export const TEXW = 2048;
export const LMIN = 2; // lowest BLA level kept (skips 2^LMIN iterations)
export const MAX_LEVELS = 24;
// above this float32 overflows near the bailout radius
export const MAX_POWER = 10;
const ZERO_E = -10000000;
export const EPS_L2 = -16; // log2 of the BLA linearisation tolerance

export interface RefRequest {
  type: 'ref';
  id: number;
  cx: bigint;
  cy: bigint;
  prec: number;
  maxIter: number;
  lcmax: number;
  power: number; // z -> z^power + c
}
export interface BlaRequest {
  type: 'bla';
  id: number;
  lcmax: number;
}
export interface BlaData {
  blaA: Float32Array | null;
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
  power: number;
  M: number;
  escaped: boolean;
  orbit: Float32Array;
  orbitRows: number;
  ms: number;
}
export interface BlaResult extends BlaData {
  type: 'bla';
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
  M: number;
  zr: Float64Array;
  zi: Float64Array;
  ze: Int32Array;
}
let cur: Orbit | null = null;
let latestRef = 0;

async function computeRef(req: RefRequest): Promise<void> {
  const t0 = performance.now();
  const { cx, cy, prec, power } = req;
  // below this a plain float32 Z would underflow once raised to power - 1
  const TINY = Math.pow(2, -Math.min(56, Math.floor(100 / (power - 1))));
  const cap = req.maxIter + 1;
  const P = BigInt(prec);
  const P1 = BigInt(prec - 1);
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  const rowsCap = Math.ceil(cap / TEXW);
  const orbit = new Float32Array(rowsCap * TEXW * 4);
  const zr = new Float64Array(cap);
  const zi = new Float64Array(cap);
  const ze = new Int32Array(cap);

  let zx = 0n;
  let zy = 0n;
  let M = 0;
  let escaped = false;
  let tick = performance.now();
  for (let n = 0; n < cap; n++) {
    const x = Number(zx >> SH) * INV;
    const y = Number(zy >> SH) * INV;
    const o = n * 4;
    if (n === 0 || (Math.abs(x) < TINY && Math.abs(y) < TINY)) {
      // too small for a plain float: store mantissa + exponent
      if (zx === 0n && zy === 0n) {
        orbit[o + 2] = ZERO_E;
        ze[n] = ZERO_E;
      } else {
        const s2 = Math.max(0, Math.max(bitLen(zx), bitLen(zy)) - 60);
        normz(Number(zx >> BigInt(s2)), Number(zy >> BigInt(s2)), s2 - prec);
        orbit[o] = nx;
        orbit[o + 1] = ny;
        orbit[o + 2] = ne;
        zr[n] = nx;
        zi[n] = ny;
        ze[n] = ne;
      }
      orbit[o + 3] = 1;
    } else {
      orbit[o] = x;
      orbit[o + 1] = y;
      zr[n] = x;
      zi[n] = y;
    }
    M = n + 1;
    if (x * x + y * y > R2) {
      escaped = true;
      break;
    }
    if (n === cap - 1) break;
    if (power === 2) {
      const x2 = zx * zx;
      const y2 = zy * zy;
      const xy = zx * zy;
      zx = ((x2 - y2) >> P) + cx;
      zy = (xy >> P1) + cy;
    } else {
      let px = zx;
      let py = zy;
      for (let i = 1; i < power; i++) {
        const nx = (px * zx - py * zy) >> P;
        py = (px * zy + py * zx) >> P;
        px = nx;
      }
      zx = px + cx;
      zy = py + cy;
    }
    if ((n & 255) === 0 && performance.now() - tick > 30) {
      await breathe();
      if (req.id !== latestRef) return;
      tick = performance.now();
    }
  }

  cur = { id: req.id, power, M, zr, zi, ze };
  const bla = buildBla(cur, req.lcmax);
  const orbitRows = Math.ceil(M / TEXW);
  const packed = orbit.subarray(0, orbitRows * TEXW * 4);
  const out: RefResult = {
    type: 'ref',
    id: req.id,
    cx,
    cy,
    prec,
    maxIter: req.maxIter,
    power,
    M,
    escaped,
    orbit: packed,
    orbitRows,
    ms: performance.now() - t0,
    ...bla,
  };
  const transfer: Transferable[] = [orbit.buffer];
  if (bla.blaA) transfer.push(bla.blaA.buffer, bla.blaE!.buffer);
  (self as unknown as Worker).postMessage(out, transfer);
}

// BLA: for reference index m the map d_m -> d_{m+l} is approximated by
//   d' = A*d + B*dc      valid while |d| < r
// Level 0 is a single iteration (A = p*Z^(p-1), B = 1, r ~ eps*|Z|); each higher
// level merges two neighbours. A, B are float-exp, r is kept as log2.
function buildBla(o: Orbit, lcmax: number): BlaData {
  const blaOff = new Int32Array(MAX_LEVELS);
  const blaCnt = new Int32Array(MAX_LEVELS);
  const none: BlaData = { blaA: null, blaE: null, blaRows: 0, blaOff, blaCnt, blaLevels: 0, lcmax, lrMax: -1e30 };
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
  return { blaA: outA, blaE: outE, blaRows: rows, blaOff, blaCnt, blaLevels: top - LMIN + 1, lcmax, lrMax };
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
    void computeRef(msg);
  } else if (cur && cur.id === msg.id) {
    const bla = buildBla(cur, msg.lcmax);
    const out: BlaResult = { type: 'bla', id: msg.id, ...bla };
    const transfer: Transferable[] = bla.blaA ? [bla.blaA.buffer, bla.blaE!.buffer] : [];
    (self as unknown as Worker).postMessage(out, transfer);
  }
};

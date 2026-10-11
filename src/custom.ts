// Exact (BigInt) arithmetic for the formulas that are neither z^p + c nor a
// folded quadratic. Each has its own perturbation step in the shader; this file
// is the ground truth they are derived from, used for the reference orbit and
// for brute-force checks.

import type { Spec } from './formulas';

export const LAMBDA = 0;
export const PHOENIX = 1;
export const MANOWAR = 2;
export const SPIDER = 3;
export const NEWTON = 4;
export const NOVA = 5;
export const MAGNET = 6;
export const TETRATION = 7;

export interface CustomInfo {
  usesC: boolean; // the pixel is the parameter c (otherwise it is the starting point)
  hasU: boolean; // a second per-iteration reference value is stored
  second: boolean; // a second orbit, from the map's critical state, for pixels to restart on
  rebaseCrit: boolean; // the stored orbit is the distance to a critical point: rebase when the pixel is nearer to it
  bail: number; // |z|^2 beyond which the orbit has escaped
  converges: boolean; // orbits can settle on a fixed point
}

const INFO: CustomInfo[] = [
  { second: false, usesC: true, hasU: false, rebaseCrit: true, bail: 1e6, converges: false }, // lambda
  { usesC: false, hasU: true, second: true, rebaseCrit: false, bail: 1e6, converges: false }, // phoenix
  { second: false, usesC: true, hasU: true, rebaseCrit: false, bail: 1e6, converges: false }, // manowar
  { second: false, usesC: true, hasU: true, rebaseCrit: false, bail: 1e6, converges: false }, // spider
  { second: false, usesC: false, hasU: false, rebaseCrit: false, bail: Infinity, converges: true }, // newton: a trip past the pole comes back
  { second: false, usesC: true, hasU: false, rebaseCrit: true, bail: 1e30, converges: true }, // nova
  { second: false, usesC: true, hasU: true, rebaseCrit: true, bail: 1e6, converges: true }, // magnet
  { second: false, usesC: true, hasU: true, rebaseCrit: false, bail: 1e24, converges: true }, // tetration (the second value is the bailout margin)
];
export function customInfo(id: number): CustomInfo {
  return INFO[id];
}

/** Iterations worth spending at a given depth (in octaves). */
export function customIters(id: number, depth: number): number {
  // enough that what is drawn as "never settles" really is so, not just cut short
  if (id === NEWTON) return 500 + 12 * depth;
  if (id === NOVA) return 2500 + 80 * depth;
  if (id === TETRATION) return 800 + 40 * depth;
  return 1000 + 80 * Math.pow(depth, 1.5);
}

type Cx = [bigint, bigint];
export interface State {
  z: Cx;
  u: Cx;
}

export interface CustomMath {
  info: CustomInfo;
  toD(v: bigint): number;
  /** State of the orbit that belongs to the point p (and the parameter it implies). */
  start(p: Cx): State;
  /** Parameter used for the point p. */
  param(p: Cx): Cx;
  step(s: State, c: Cx): void;
  /** The extra reference value stored alongside z for this state. */
  refU(s: State, c: Cx): Cx;
  /** Where the extra value is not a plain complex number: four floats to store instead. */
  refUf?(s: State, c: Cx): number[];
  /** Bailout decided in exact arithmetic (formulas whose colouring steps at the bailout). */
  escaped?(s: State, c: Cx): boolean;
}

const PH_K = 0.5667;

export function makeCustom(spec: Spec, prec: number): CustomMath {
  const id = spec.custom;
  const info = INFO[id];
  const P = BigInt(prec);
  const ONE = 1n << P;
  const SH = BigInt(prec - 62);
  const INV = Math.pow(2, -62);
  const toD = (v: bigint) => Number(v >> SH) * INV;
  const fromD = (v: number): bigint => BigInt(Math.round(v * 4503599627370496)) << BigInt(prec - 52);
  const mul = (a: Cx, b: Cx): Cx => [(a[0] * b[0] - a[1] * b[1]) >> P, (a[0] * b[1] + a[1] * b[0]) >> P];
  const sqr = (a: Cx): Cx => [(a[0] * a[0] - a[1] * a[1]) >> P, (a[0] * a[1]) >> (P - 1n)];
  const div = (a: Cx, b: Cx): Cx => {
    const den = b[0] * b[0] + b[1] * b[1];
    if (den === 0n) return [ONE << 40n, 0n]; // a pole: far away is all that matters
    return [((a[0] * b[0] + a[1] * b[1]) << P) / den, ((a[1] * b[0] - a[0] * b[1]) << P) / den];
  };
  const bits = (v: bigint) => (v < 0n ? -v : v).toString(2).length;

  // exp by scaling down, a Taylor series, and squaring back up
  const exp = (a: Cx): Cx => {
    const xr = toD(a[0]);
    if (xr > 60) return [ONE << 50n, 0n]; // overflows the bailout anyway
    if (xr < -prec) return [0n, 0n];
    const k = Math.max(0, Math.max(bits(a[0]), bits(a[1])) - (prec - 8));
    const r: Cx = [a[0] >> BigInt(k), a[1] >> BigInt(k)];
    let term: Cx = [ONE, 0n];
    const sum: Cx = [ONE, 0n];
    for (let j = 1; j < 2000; j++) {
      term = mul(term, r);
      term = [term[0] / BigInt(j), term[1] / BigInt(j)];
      if (term[0] === 0n && term[1] === 0n) break;
      sum[0] += term[0];
      sum[1] += term[1];
    }
    let out: Cx = sum;
    for (let i = 0; i < k; i++) out = sqr(out);
    return out;
  };
  // ln by Newton's method on exp, started from the double-precision value
  const ln = (c: Cx): Cx => {
    const x = toD(c[0]);
    const y = toD(c[1]);
    if (x === 0 && y === 0) return [0n, 0n];
    let L: Cx = [fromD(0.5 * Math.log(x * x + y * y)), fromD(Math.atan2(y, x))];
    for (let i = 0; i < Math.ceil(Math.log2(prec / 40)) + 2; i++) {
      const t = mul(c, exp([-L[0], -L[1]]));
      L = [L[0] + t[0] - ONE, L[1] + t[1]];
    }
    return L;
  };

  let lnOf: Cx | null = null; // tetration: c whose logarithm is cached
  let lnVal: Cx = [0n, 0n];
  const lnC = (c: Cx): Cx => {
    if (!lnOf || lnOf[0] !== c[0] || lnOf[1] !== c[1]) {
      lnOf = [c[0], c[1]];
      lnVal = ln(c);
    }
    return lnVal;
  };
  const phK = fromD(PH_K);
  const magnetQ = (z: Cx, c: Cx): Cx => {
    const z2 = sqr(z);
    return div([z2[0] + c[0] - ONE, z2[1] + c[1]], [2n * z[0] + c[0] - 2n * ONE, 2n * z[1] + c[1]]);
  };

  // Tetration is judged on the exponent t = z ln c: the next value exp(t) is past
  // the bailout 10^12 exactly when Re t >= ln(10^12).
  let theta = 0n;
  let twoPi = 0n;
  const texp = (s: State, c: Cx): Cx => {
    if (theta === 0n) {
      theta = 12n * ln([10n * ONE, 0n])[0];
      twoPi = 2n * ln([-ONE, 0n])[1];
    }
    return mul(s.z, lnC(c));
  };
  // for the shader: Re t - ln(10^12) as mantissa * 2^exponent, then Re t and Im t mod 2 pi
  const tinfo = (s: State, c: Cx): number[] => {
    const t = texp(s, c);
    const v = t[0] - theta;
    let mant = 0;
    let ex = -1000000;
    if (v !== 0n) {
      const bl = bits(v);
      const sh = Math.max(0, bl - 53);
      mant = Number(v >> BigInt(sh)) / Math.pow(2, bl - sh);
      ex = bl - prec;
    }
    const re = Math.max(-1e30, Math.min(1e30, toD(t[0])));
    return [mant, ex, re, toD(((t[1] % twoPi) + twoPi) % twoPi)];
  };

  return {
    info,
    toD,
    param: (p) => p,
    refUf: id === TETRATION ? tinfo : undefined,
    escaped: id === TETRATION ? (s, c) => texp(s, c)[0] >= theta : undefined,
    start(p) {
      switch (id) {
        case PHOENIX:
          return { z: [-p[1], p[0]], u: [0n, 0n] }; // the classic view is turned on its side
        case MANOWAR:
          return { z: [p[0], p[1]], u: [p[0], p[1]] };
        case SPIDER:
          return { z: [0n, 0n], u: [p[0], p[1]] };
        case NEWTON:
        case TETRATION:
          return { z: [p[0], p[1]], u: [0n, 0n] };
        default:
          return { z: [0n, 0n], u: [0n, 0n] }; // lambda and nova hold the offset from their critical point
      }
    },
    refU(s, c) {
      return id === MAGNET ? magnetQ(s.z, c) : s.u;
    },
    step(s, c) {
      const z = s.z;
      switch (id) {
        case LAMBDA: {
          // z -> c z (1 - z) in y = z - 1/2:  y -> c (1/4 - y^2) - 1/2
          const y2 = sqr(z);
          const m = mul(c, [ONE / 4n - y2[0], -y2[1]]);
          s.z = [m[0] - ONE / 2n, m[1]];
          break;
        }
        case PHOENIX: {
          const z2 = sqr(z);
          s.z = [z2[0] + phK - (s.u[0] >> 1n), z2[1] - (s.u[1] >> 1n)];
          s.u = z;
          break;
        }
        case MANOWAR: {
          const z2 = sqr(z);
          s.z = [z2[0] + s.u[0] + c[0], z2[1] + s.u[1] + c[1]];
          s.u = z;
          break;
        }
        case SPIDER: {
          const z2 = sqr(z);
          s.z = [z2[0] + s.u[0], z2[1] + s.u[1]];
          s.u = [(s.u[0] >> 1n) + s.z[0], (s.u[1] >> 1n) + s.z[1]];
          break;
        }
        case NEWTON: {
          // z -> z - (z^p - 1) / (p z^(p-1))
          let zq: Cx = [ONE, 0n];
          for (let i = 1; i < spec.power; i++) zq = mul(zq, z);
          const zp = mul(zq, z);
          const pw = BigInt(spec.power);
          const d = div([zp[0] - ONE, zp[1]], [zq[0] * pw, zq[1] * pw]);
          s.z = [z[0] - d[0], z[1] - d[1]];
          break;
        }
        case NOVA: {
          // Newton for z^3 = 1 plus c, in y = z - 1:  y -> y^2 (3 + 2y) / (3 (1 + y)^2) + c
          const b2 = sqr([ONE + z[0], z[1]]);
          const d = div(mul(sqr(z), [3n * ONE + 2n * z[0], 2n * z[1]]), [3n * b2[0], 3n * b2[1]]);
          s.z = [d[0] + c[0], d[1] + c[1]];
          break;
        }
        case MAGNET:
          s.z = sqr(magnetQ(z, c));
          break;
        default:
          s.z = exp(mul(z, lnC(c))); // tetration: z -> c^z
      }
    },
  };
}

const LLR = Math.log2(Math.log2(1e6));

/**
 * Smooth iteration value of the pixel at p, by exact iteration; -1 when it
 * neither escapes nor settles within N. Mirrors the shader's tests step for step.
 * `rootOnly` (Newton) returns just the index of the root reached.
 */
export function customBrute(spec: Spec, prec: number, p: Cx, N: number, rootOnly = false): number {
  const id = spec.custom;
  const f = makeCustom(spec, prec);
  const c = p;
  const s = f.start(p);
  let prevDd = 1;
  for (let n = 0; ; n++) {
    const x = f.toD(s.z[0]);
    const y = f.toD(s.z[1]);
    const zz = x * x + y * y;
    if (id === NEWTON) {
      // z^p - 1 in doubles
      let fx = 1;
      let fy = 0;
      for (let i = 0; i < spec.power; i++) [fx, fy] = [fx * x - fy * y, fx * y + fy * x];
      const ff = (fx - 1) * (fx - 1) + fy * fy;
      if (ff < 1e-5) {
        let k = Math.floor((Math.atan2(y, x) / (2 * Math.PI)) * spec.power + 0.5);
        if (k < 0) k += spec.power;
        if (rootOnly) return k;
        return (n - Math.log2(Math.log(Math.max(ff, 1e-11)) / Math.log(1e-5))) * 7 + k * 97;
      }
    } else if (id === TETRATION) {
      if (f.escaped!(s, c)) return 3 * (n + 1) + 3; // the next value is past the bailout
    } else if (id !== NOVA) {
      if (zz > 1e6) return Math.max(0, n + 1 + LLR - Math.log2(Math.log2(zz)));
      if (id === MAGNET) {
        const d1 = (x - 1) * (x - 1) + y * y;
        if (d1 < 1e-4) return 60 + 2 * (n - Math.log2(Math.log(Math.max(d1, 1e-9)) / Math.log(1e-4)));
      }
    }
    if (n >= N) return -1;
    const before = s.z;
    f.step(s, c);
    if (id === NOVA) {
      const dx = f.toD(s.z[0] - before[0]);
      const dy = f.toD(s.z[1] - before[1]);
      const dd = dx * dx + dy * dy;
      if (dd < 1e-8) {
        const frac = Math.min(1, Math.max(0, Math.log(1e-8 / prevDd) / Math.log(dd / prevDd)));
        return (n + (Number.isFinite(frac) ? frac : 0)) * 5;
      }
      prevDd = dd;
    }
  }
}

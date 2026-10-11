// View state: the centre is arbitrary-precision fixed point (BigInt), the scale is
// kept as log2 so it never underflows. All offsets handed to the GPU are small
// doubles measured relative to the scale.

export interface Point {
  cx: bigint;
  cy: bigint;
  prec: number; // fractional bits of cx / cy
}

export interface View extends Point {
  ls: number; // log2 of the view half-height
}

export const LS_MAX = 2.5;
export const LS_MIN = -9000;

/** v * 2^e2 as fixed point with `prec` fractional bits. */
export function dblToFixed(v: number, e2: number, prec: number): bigint {
  if (v === 0 || !Number.isFinite(v)) return 0n;
  const m = BigInt(Math.round(v * 4503599627370496)); // 2^52
  const sh = e2 + prec - 52;
  return sh >= 0 ? m << BigInt(sh) : m >> BigInt(-sh);
}

/** d / 2^prec / 2^e2 as a double. */
export function fixedToDbl(d: bigint, prec: number, e2: number): number {
  const t = prec + e2;
  if (t > 64) return Number(d >> BigInt(t - 64)) / 18446744073709551616;
  if (t >= 0) return Number(d) / Math.pow(2, t);
  return Number(d) * Math.pow(2, -t);
}

export function rescale(x: bigint, from: number, to: number): bigint {
  return to >= from ? x << BigInt(to - from) : x >> BigInt(from - to);
}

export function precFor(ls: number): number {
  return Math.max(64, Math.ceil((Math.ceil(-ls) + 80) / 32) * 32);
}

export function fitPrec(v: View): void {
  const p = precFor(v.ls);
  if (p > v.prec || p < v.prec - 96) {
    v.cx = rescale(v.cx, v.prec, p);
    v.cy = rescale(v.cy, v.prec, p);
    v.prec = p;
  }
}

export function cloneView(v: View): View {
  return { cx: v.cx, cy: v.cy, prec: v.prec, ls: v.ls };
}

export function sameView(a: View, b: View): boolean {
  if (a.ls !== b.ls) return false;
  const p = Math.max(a.prec, b.prec);
  return (
    rescale(a.cx, a.prec, p) === rescale(b.cx, b.prec, p) &&
    rescale(a.cy, a.prec, p) === rescale(b.cy, b.prec, p)
  );
}

/** (a - b) / 2^e2 */
export function delta(a: Point, b: Point, e2: number): [number, number] {
  const p = Math.max(a.prec, b.prec);
  const dx = rescale(a.cx, a.prec, p) - rescale(b.cx, b.prec, p);
  const dy = rescale(a.cy, a.prec, p) - rescale(b.cy, b.prec, p);
  return [fixedToDbl(dx, p, e2), fixedToDbl(dy, p, e2)];
}

/** Offset of point `a` from the centre of `v`, in view half-heights. */
export function offsetIn(v: View, a: Point): [number, number] {
  const ec = Math.floor(v.ls);
  const sm = Math.pow(2, v.ls - ec);
  const [dx, dy] = delta(a, v, ec);
  return [dx / sm, dy / sm];
}

/** Point at offset (dx, dy) half-heights from the centre of `v`. */
export function pointAt(v: View, dx: number, dy: number): Point {
  const ec = Math.floor(v.ls);
  const sm = Math.pow(2, v.ls - ec);
  return {
    cx: v.cx + dblToFixed(dx * sm, ec, v.prec),
    cy: v.cy + dblToFixed(dy * sm, ec, v.prec),
    prec: v.prec,
  };
}

export function pan(v: View, dx: number, dy: number): void {
  const p = pointAt(v, dx, dy);
  v.cx = p.cx;
  v.cy = p.cy;
}

/** Zoom by `dls` octaves keeping the point at offset (ax, ay) half-heights fixed on screen. */
export function zoomAt(v: View, ax: number, ay: number, dls: number): void {
  const nls = Math.min(LS_MAX, Math.max(LS_MIN, v.ls + dls));
  dls = nls - v.ls;
  if (dls === 0) return;
  if (dls < 0) {
    // going deeper: grow precision before moving the centre
    v.ls = nls;
    fitPrec(v);
    v.ls = nls - dls;
  }
  const f = Math.pow(2, dls);
  pan(v, ax * (1 - f), ay * (1 - f));
  v.ls = nls;
  fitPrec(v);
}

export function parseDecimal(s: string, prec: number): bigint {
  const m = /^\s*([+-]?)(\d*)\.?(\d*)\s*$/.exec(s);
  if (!m || (m[2] === '' && m[3] === '')) return 0n;
  const digits = m[3].length;
  const val = BigInt((m[2] || '0') + m[3]);
  const r = (val << BigInt(prec)) / 10n ** BigInt(digits);
  return m[1] === '-' ? -r : r;
}

export function formatDecimal(x: bigint, prec: number, digits: number): string {
  const neg = x < 0n;
  const a = neg ? -x : x;
  const scaled = (a * 10n ** BigInt(digits)) >> BigInt(prec);
  const s = scaled.toString().padStart(digits + 1, '0');
  const out = s.slice(0, s.length - digits) + '.' + s.slice(s.length - digits);
  return (neg ? '-' : '') + out;
}

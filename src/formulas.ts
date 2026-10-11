// The catalogue of fractals and the exact (BigInt) form of each iteration.
//
// Three families, all drawn by perturbation against an arbitrary-precision orbit:
//   power  z -> z^p + c, with BLA skipping: the fastest at depth.
//   abs    quadratic maps with folds (abs / conjugate).
//   custom everything else; each formula has its own perturbation step (custom.ts).

export type Kind = 'power' | 'abs' | 'custom';

/** Everything the reference worker and the shader need to know about the map. */
export interface Spec {
  kind: Kind;
  power: number;
  absRe: boolean; // |Re(z^2)|
  imMode: 0 | 1 | 2 | 3; // Im(z^2)/2 taken as: x*y, |x*y|, |x|*y, x*|y|
  imSign: 1 | -1;
  julia: boolean; // c is fixed and the pixel is the starting z
  custom: number; // which custom formula (kind 'custom')
  param: number; // the pixel is mapped before use as c: 0 as is, 1 the lambda map c(2 - c)/4
}

export interface Formula {
  id: string;
  name: string;
  desc: string;
  group: string;
  kind: Kind;
  absRe?: boolean;
  imMode?: 0 | 1 | 2 | 3;
  imSign?: 1 | -1;
  julia?: boolean;
  custom?: number;
  param?: number;
  hasPower?: boolean; // the power setting applies
  power?: number; // default power
  home: [number, number, number]; // centre re, im and log2 of the half-height
}

const DEEP = 'Бесконечная глубина';
const FOLD = 'Со складками';
const PLAIN = 'Особые формулы';

export const FORMULAS: Formula[] = [
  { id: 'mandelbrot', name: 'Мандельброт', desc: 'z² + c — классика; степень меняется клавишей M', group: DEEP, kind: 'power', hasPower: true, power: 2, home: [-0.75, 0, 0.35] },
  { id: 'julia', name: 'Жюлиа', desc: 'то же правило, но c закреплено; свой узор для каждой точки', group: DEEP, kind: 'power', julia: true, hasPower: true, power: 2, home: [0, 0, 0.6] },

  { id: 'ship', name: 'Горящий корабль', desc: 'модули под квадратом рождают мачты и паруса; отдалите, чтобы увидеть целиком', group: FOLD, kind: 'abs', imMode: 1, imSign: -1, home: [-1.762, 0.028, -4.45] },
  { id: 'tricorn', name: 'Трикорн', desc: 'сопряжённый Мандельброт с тройной симметрией', group: FOLD, kind: 'abs', imMode: 0, imSign: -1, home: [-0.25, 0, 0.75] },
  { id: 'celtic', name: 'Кельтский', desc: 'модуль действительной части — узлы и петли', group: FOLD, kind: 'abs', absRe: true, imMode: 0, imSign: 1, home: [-0.75, 0, 0.62] },
  { id: 'buffalo', name: 'Буффало', desc: 'модули обеих частей — рогатая голова', group: FOLD, kind: 'abs', absRe: true, imMode: 1, imSign: -1, home: [-0.5, 0.45, 0.45] },
  { id: 'perp', name: 'Перпендикулярный Мандельброт', desc: 'складка по одной оси, острые шпили', group: FOLD, kind: 'abs', imMode: 2, imSign: -1, home: [-0.65, 0, 0.45] },

  { id: 'newton', name: 'Ньютон', desc: 'бассейны корней zⁿ = 1; степень меняется клавишей M', group: PLAIN, kind: 'custom', custom: 4, hasPower: true, power: 3, home: [0, 0, 0.4] },
  { id: 'nova', name: 'Нова', desc: 'метод Ньютона со сдвигом — множество из завитков', group: PLAIN, kind: 'custom', custom: 5, home: [-0.45, 0, 0.0] },
  { id: 'magnet', name: 'Магнит', desc: 'формула из физики фазовых переходов', group: PLAIN, kind: 'custom', custom: 6, home: [1.4, 0, 0.9] },
  { id: 'phoenix', name: 'Феникс', desc: 'каждая итерация помнит предыдущую', group: PLAIN, kind: 'custom', custom: 1, home: [0, 0, 0.25] },
  { id: 'lambda', name: 'Лямбда', desc: 'логистическое отображение c·z·(1 − z): два Мандельброта спиной к спине', group: PLAIN, kind: 'power', param: 1, power: 2, home: [1, 0, 1.0] },
  { id: 'spider', name: 'Паук', desc: 'c меняется вместе с z на каждом шаге', group: PLAIN, kind: 'custom', custom: 3, home: [-0.6, 0, 0.5] },
  { id: 'manowar', name: 'Мановар', desc: 'z² плюс предыдущее значение', group: PLAIN, kind: 'custom', custom: 2, home: [-0.03, 0, -2.3] },
  { id: 'tetration', name: 'Тетрация', desc: 'степенная башня c^c^c…', group: PLAIN, kind: 'custom', custom: 7, home: [0.8, 0, 1.7] },
];

export const JULIA_PRESETS: { name: string; re: number; im: number }[] = [
  { name: 'Спирали', re: -0.8, im: 0.156 },
  { name: 'Кролик Дуади', re: -0.123, im: 0.745 },
  { name: 'Дендрит', re: 0, im: 1 },
  { name: 'Молнии', re: -0.70176, im: -0.3842 },
  { name: 'Галактики', re: 0.285, im: 0.01 },
  { name: 'Диск Зигеля', re: -0.391, im: -0.587 },
  { name: 'Кружево', re: -0.7269, im: 0.1889 },
  { name: 'Сан-Марко', re: -0.75, im: 0 },
  { name: 'Самолёт', re: -1.755, im: 0 },
  { name: 'Морские коньки', re: -0.74543, im: 0.11301 },
];

export function specOf(f: Formula, power: number, julia: boolean): Spec {
  return {
    kind: f.kind,
    power: f.kind === 'power' || f.hasPower ? power : 2,
    absRe: !!f.absRe,
    imMode: f.imMode ?? 0,
    imSign: f.imSign ?? 1,
    julia: f.kind !== 'custom' && !f.param && (julia || !!f.julia),
    custom: f.custom ?? 0,
    param: f.param ?? 0,
  };
}

export function specKey(s: Spec): string {
  return s.kind === 'custom'
    ? `c${s.custom}`
    : `${s.kind}|${s.absRe ? 1 : 0}|${s.imMode}|${s.imSign}|${s.julia ? 1 : 0}|${s.param}`;
}

// Result registers of bigStep, to avoid allocating a pair per iteration.
export let sx = 0n;
export let sy = 0n;

/** One exact iteration z -> f(z) + c in fixed point with `prec` fractional bits; result in sx, sy. */
export function makeBigStep(s: Spec, prec: number): (zx: bigint, zy: bigint, cx: bigint, cy: bigint) => void {
  const P = BigInt(prec);
  const P1 = BigInt(prec - 1);
  if (s.kind === 'abs') {
    const { absRe, imMode, imSign } = s;
    return (zx, zy, cx, cy) => {
      let re = (zx * zx - zy * zy) >> P;
      if (absRe && re < 0n) re = -re;
      const ax = zx < 0n ? -zx : zx;
      const ay = zy < 0n ? -zy : zy;
      let im = imMode === 0 ? zx * zy : imMode === 1 ? ax * ay : imMode === 2 ? ax * zy : zx * ay;
      im >>= P1;
      sx = re + cx;
      sy = (imSign < 0 ? -im : im) + cy;
    };
  }
  if (s.power === 2) {
    return (zx, zy, cx, cy) => {
      const x2 = zx * zx;
      const y2 = zy * zy;
      const xy = zx * zy;
      sx = ((x2 - y2) >> P) + cx;
      sy = (xy >> P1) + cy;
    };
  }
  const power = s.power;
  return (zx, zy, cx, cy) => {
    let px = zx;
    let py = zy;
    for (let i = 1; i < power; i++) {
      const nx = (px * zx - py * zy) >> P;
      py = (px * zy + py * zx) >> P;
      px = nx;
    }
    sx = px + cx;
    sy = py + cy;
  };
}

/**
 * The parameter the iteration really uses for the view-plane point p (fixed point,
 * `prec` bits). Lambda: c z (1 - z) is z^2 + c' in other coordinates, c' = c (2 - c) / 4.
 */
export function paramOf(s: Spec, px: bigint, py: bigint, prec: number): [bigint, bigint] {
  if (s.param !== 1) return [px, py];
  const P = BigInt(prec);
  const ax = (2n << P) - px;
  const ay = -py;
  return [((px * ax - py * ay) >> P) >> 2n, ((px * ay + py * ax) >> P) >> 2n];
}

/** log2 of the largest parameter offset when the view-plane offsets reach 2^lc around (cre, cim). */
export function paramLc(s: Spec, lc: number, cre: number, cim: number): number {
  if (s.param !== 1) return lc;
  // dc' = dc (1 - c)/2 - dc^2/4
  return lc + Math.log2(Math.hypot(1 - cre, cim) / 2 + Math.pow(2, lc) / 4);
}

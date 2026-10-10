// Colour palettes. Each is a list of [position 0..1, r, g, b] stops in sRGB;
// the texture is sampled with mirrored repeat so cycling never shows a seam.
//
// A palette only reads well on a fractal if brightness swings as well as hue:
// neighbouring bands of equal brightness melt together however different the
// hues are. So every palette here runs through dark and light stops in turn.

export interface Palette {
  name: string;
  stops: [number, number, number, number][];
}

const c = (t: number, r: number, g: number, b: number): [number, number, number, number] => [t, r, g, b];

export const PALETTES: Palette[] = [
  {
    name: 'Глубина',
    stops: [c(0, 0, 7, 100), c(0.16, 32, 107, 203), c(0.42, 237, 255, 255), c(0.6425, 255, 170, 0), c(0.8575, 60, 10, 20), c(1, 0, 2, 30)],
  },
  {
    name: 'Классический',
    stops: [c(0, 0, 0, 50), c(0.17, 0, 70, 230), c(0.34, 0, 200, 220), c(0.5, 60, 220, 90), c(0.66, 250, 235, 40), c(0.83, 235, 50, 20), c(1, 50, 0, 10)],
  },
  {
    name: 'Огненный',
    stops: [c(0, 6, 0, 0), c(0.2, 110, 0, 0), c(0.4, 225, 35, 0), c(0.58, 255, 140, 0), c(0.76, 255, 225, 90), c(0.9, 255, 255, 235), c(1, 120, 150, 255)],
  },
  { name: 'Океанский', stops: [c(0, 0, 0, 77), c(0.2, 0, 0, 255), c(0.5, 0, 255, 255), c(0.8, 0, 255, 204), c(1, 255, 255, 255)] },
  {
    name: 'Лесной',
    stops: [c(0, 4, 18, 10), c(0.22, 18, 92, 40), c(0.42, 125, 185, 55), c(0.6, 238, 228, 150), c(0.8, 150, 98, 38), c(1, 34, 24, 14)],
  },
  {
    name: 'Фиолетовый',
    stops: [c(0, 10, 0, 30), c(0.22, 78, 20, 160), c(0.42, 200, 60, 220), c(0.6, 255, 190, 240), c(0.76, 255, 242, 205), c(0.9, 240, 120, 90), c(1, 50, 0, 70)],
  },
  {
    name: 'Золотой закат',
    stops: [c(0, 18, 10, 50), c(0.2, 118, 28, 92), c(0.4, 232, 70, 50), c(0.58, 255, 172, 40), c(0.76, 255, 236, 172), c(1, 52, 86, 150)],
  },
  {
    name: 'Неоновый',
    stops: [c(0, 0, 0, 0), c(0.14, 0, 255, 160), c(0.28, 0, 28, 58), c(0.42, 255, 0, 200), c(0.56, 30, 0, 60), c(0.7, 0, 200, 255), c(0.85, 8, 8, 40), c(1, 255, 240, 0)],
  },
  {
    name: 'Пастельный',
    stops: [c(0, 62, 72, 112), c(0.2, 160, 190, 235), c(0.4, 246, 226, 236), c(0.57, 250, 188, 168), c(0.74, 198, 158, 210), c(0.89, 150, 212, 200), c(1, 84, 104, 134)],
  },
  {
    name: 'Лавовый',
    stops: [c(0, 0, 0, 0), c(0.3, 38, 6, 6), c(0.5, 160, 18, 0), c(0.64, 255, 92, 0), c(0.78, 255, 205, 70), c(0.9, 118, 28, 10), c(1, 18, 10, 16)],
  },
  { name: 'Ледяной', stops: [c(0, 0, 0, 128), c(0.4, 0, 128, 255), c(0.7, 255, 255, 255), c(1, 204, 255, 255)] },
  {
    name: 'Психоделический',
    stops: [c(0, 255, 0, 110), c(0.14, 30, 0, 60), c(0.28, 0, 230, 255), c(0.42, 255, 240, 0), c(0.56, 0, 58, 40), c(0.7, 120, 255, 0), c(0.84, 90, 0, 200), c(1, 255, 120, 0)],
  },
  { name: 'Монохромный', stops: [c(0, 0, 0, 0), c(0.33, 77, 77, 77), c(0.66, 153, 153, 153), c(1, 255, 255, 255)] },
  {
    // full hue circle, with every other stop darkened so the bands stay apart
    name: 'Радужный спектр',
    stops: Array.from({ length: 13 }, (_, i) => {
      const [r, g, b] = hsv((i / 12) % 1, i % 2 ? 0.9 : 0.75, i % 2 ? 0.42 : 1);
      return c(i / 12, r, g, b);
    }),
  },
];

function hsv(h: number, s: number, v: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 6) % 6;
    return Math.round(255 * (v - v * s * Math.max(0, Math.min(k, 4 - k, 1))));
  };
  return [f(5), f(3), f(1)];
}

export const PAL_SIZE = 1024;

export function bakePalette(p: Palette): Uint8Array {
  const out = new Uint8Array(PAL_SIZE * 4);
  const s = p.stops;
  let k = 0;
  for (let i = 0; i < PAL_SIZE; i++) {
    const t = i / (PAL_SIZE - 1);
    while (k < s.length - 2 && t > s[k + 1][0]) k++;
    const a = s[k];
    const b = s[k + 1];
    let f = Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0])));
    f = f * f * (3 - 2 * f);
    out[i * 4] = a[1] + (b[1] - a[1]) * f;
    out[i * 4 + 1] = a[2] + (b[2] - a[2]) * f;
    out[i * 4 + 2] = a[3] + (b[3] - a[3]) * f;
    out[i * 4 + 3] = 255;
  }
  return out;
}

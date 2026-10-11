// GLSL ES 3.00 sources.
import type { Spec } from './formulas';

export const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Perturbation iteration with rebasing and BLA skipping.
//
// Per pixel we track d = z - Z (distance to the reference orbit) as
//   d = w * 2^e          w: float32 pair kept near 1, e: integer exponent
// so d can be far below float32 range. The pixel's offset from the reference
// point is dc = cm * 2^uEc. State lives in two float textures and is advanced
// uSteps steps per pass, so one frame never does more than a bounded amount
// of work.
//
// S0 = (w.x, w.y, e, m)        m: index into the reference orbit
// S1 = (iter | nu, status)     status: 0 running, 1 escaped, 2 hit max iterations
// The formula is compiled in through #defines, one program per fractal family,
// so the plain Mandelbrot path pays nothing for the others.
//   KIND 0: z^p + c      KIND 1: quadratic with folds (ABS_RE, IM_MODE, IM_SIGN)
//   JULIA 1: c is fixed, the pixel offset is the starting d
export function iterateSrc(s: Spec): string {
  const defs =
    `#define KIND ${s.kind === 'abs' ? 1 : 0}
#define ABS_RE ${s.absRe ? 1 : 0}
#define IM_MODE ${s.imMode}
` +
    `#define PARAM ${s.param}
#define IM_SIGN ${s.imSign < 0 ? '-1.0' : '1.0'}
#define JULIA ${s.julia ? 1 : 0}
`;
  return ITERATE.replace('//DEFINES', defs);
}

const ITERATE = `#version 300 es
//DEFINES
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;

uniform sampler2D uS0;
uniform sampler2D uS1;
uniform sampler2D uOrbit;
uniform sampler2D uBlaA;
uniform sampler2D uBlaB; // folded formulas: the B matrices (uBlaA then holds the A matrices)
uniform sampler2D uBlaE;
uniform int uInit;
uniform int uRefLen;
uniform int uEndA; // last index of the segment pixels start on
uniform int uBase; // index of the critical orbit's start, where pixels rebase to
uniform int uUOff; // folded formulas: where the per-component table starts in the orbit texture
uniform int uPerA; // period of the first reference segment once it has settled on a cycle (0 if not)
uniform int uPerB; // likewise for the last segment
uniform sampler2DArray uAccTex; // finished picture, for skipping what is certainly inside
uniform int uSkipIn; // 1: an extra sample of a finished picture, 2: a picture overlapping an earlier one
uniform int uAccLayer;
uniform vec4 uSrcX; // where this job lies in that earlier picture: ratio, offset.xy
uniform vec2 uSrcSize;
uniform ivec2 uAccMax;
uniform int uMaxIter;
uniform int uSteps;
uniform int uEc;
uniform int uBlaLevels;
uniform int uBlaOff[24];
uniform int uBlaCnt[24];
uniform vec2 uRes;
uniform vec2 uJitter;
uniform vec2 uDm;
uniform float uAspect;
uniform float uSm;
uniform float uBlaBias; // log2 widening of the BLA validity radius
uniform int uPow; // z -> z^uPow + c
uniform vec2 uKp; // (1 - C)/2 of the reference as mantissa * 2^uKe (lambda map)
uniform int uKe;
uniform float uEps2; // squared BLA tolerance: no BLA can apply once |d|^2 >= uEps2 * |Z|^2
uniform float uLP; // log2(p): per-iteration terms of log2 |dz'/dz| = log2(p) + (p-1) log2 |z|
uniform float uPH; // (p-1)/2, for use with log2 |z|^2

layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;

const float R2 = 1.0e6;
const float LLR = 4.3169906; // log2(log2(R2))
const int ZERO_E = -10000000;
const int LMIN = 2;
// A pixel is inside for certain once its orbit has contracted this much in all
// (log2 of the derivative with respect to its first point), and is still at it.
// The orbit of a point outside never contracts overall, however close it passes
// to the critical point: what it loses there it has gained on the way.
const float CONTRACTED = -50.0;
const float STILL = 20.0;

// exact 2^k; 0 when below float range
float p2(int k) {
  if (k < -126) return 0.0;
  return uintBitsToFloat(uint(min(k, 127) + 127) << 23u);
}
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 csqr(vec2 a) { return vec2(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y); }

void renorm(inout vec2 w, inout int e) {
  float m = max(abs(w.x), abs(w.y));
  if (m > 1048576.0 || (m < 9.5367431640625e-7 && m > 0.0)) {
    int k = clamp(int(floor(log2(m))), -120, 120);
    w *= p2(-k);
    e += k;
  }
}

// a such that the next d is a*d + dc:  ((Z+d)^p - Z^p) / d = sum of z^j Z^(p-1-j), z = Z+d
vec2 stepFactor(vec2 Z, vec2 d) {
  vec2 z = Z + d;
  vec2 S = z + Z;
  vec2 zp = Z;
  for (int k = 3; k <= uPow; k++) {
    zp = cmul(zp, Z);
    S = cmul(S, z) + zp;
  }
  return S;
}
#if KIND == 1
// (|c + D| - |c|) / 2^e for c = cm * 2^ce and D = d * 2^e: how a fold changes a
// perturbation. c and D may both lie far below float range (a deep zoom on an
// axis keeps one component of the orbit tiny for a long time), so they meet in
// units of 2^e, never as plain floats.
float diffabs(float cm, int ce, float d, int e) {
  if (cm == 0.0) return abs(d);
  if (ce - e > 40) return cm > 0.0 ? d : -d;
  float cw = cm * p2(ce - e);
  float t = cw + d;
  if (cw >= 0.0) return t >= 0.0 ? d : -d - 2.0 * cw;
  return t > 0.0 ? d + 2.0 * cw : -d;
}
// perturbed step of the folded quadratic at reference Z, d = w * 2^e.
// U = (X mantissa, X exponent, Y mantissa, Y exponent) of the same Z.
vec2 absStep(vec4 Z, vec4 U, vec2 w, int e) {
  float s = p2(e);
  float X = Z.x;
  float Y = Z.y;
  int ex = int(U.y);
  int ey = int(U.w);
  float re = 2.0 * (X * w.x - Y * w.y) + s * (w.x * w.x - w.y * w.y);
#if ABS_RE
  // Re(Z^2) = X^2 - Y^2: its sign is stored exactly, its size is taken from the mantissas
  int eh = max(ex, ey);
  float xa = U.x * p2(ex - eh);
  float ya = U.z * p2(ey - eh);
  float c2 = abs(xa * xa - ya * ya);
  re = diffabs(Z.z > 0.5 ? -c2 : c2, 2 * eh, re, e);
#endif
  float dxy = X * w.y + w.x * Y + s * w.x * w.y;
#if IM_MODE == 0
  float im = dxy;
#elif IM_MODE == 1
  float im = diffabs(U.x * U.z, ex + ey, dxy, e);
#elif IM_MODE == 2
  float im = diffabs(U.x, ex, w.x, e) * (Y + w.y * s) + abs(X) * w.y;
#else
  float im = diffabs(U.z, ey, w.y, e) * (X + w.x * s) + abs(Y) * w.x;
#endif
  return vec2(re, 2.0 * IM_SIGN * im);
}
// the map itself, applied to w (used where the reference is exactly 0)
vec2 absMap(vec2 w) {
  float re = w.x * w.x - w.y * w.y;
#if ABS_RE
  re = abs(re);
#endif
#if IM_MODE == 0
  float im = w.x * w.y;
#elif IM_MODE == 1
  float im = abs(w.x * w.y);
#elif IM_MODE == 2
  float im = abs(w.x) * w.y;
#else
  float im = w.x * abs(w.y);
#endif
  return vec2(re, 2.0 * IM_SIGN * im);
}
#endif
vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }

// log2(|z|^2) that survives |z|^2 overflowing float (high powers overshoot the bailout hugely)
float l2norm(vec2 z) {
  float m = max(abs(z.x), abs(z.y));
  vec2 q = z / m;
  return 2.0 * log2(m) + log2(dot(q, q));
}

vec4 orbit(int m) { return texelFetch(uOrbit, ivec2(m & 2047, m >> 11), 0); }
// Extra (jittered) samples exist to smooth edges and dust. Where the first sample
// shows a pixel and all its neighbours inside, or all escaped at practically the
// same count, another sample would say the same: reuse the value, skip the work.
// Returns -1 for "inside", a count for "same as before", -2 for "iterate it".
float settled(ivec2 px) {
  float lo = 1.0e30;
  float hi = -1.0e30;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float v = texelFetch(uAccTex, ivec3(clamp(px + ivec2(i, j), ivec2(0), uAccMax), uAccLayer), 0).r;
      if (v < -1.5) return -2.0;
      lo = min(lo, v);
      hi = max(hi, v);
    }
  }
  if (hi < 0.0) return -1.0;
  if (lo >= 0.0 && hi - lo < 0.25) return texelFetch(uAccTex, ivec3(px, uAccLayer), 0).r;
  return -2.0;
}
// While the view moves, a new picture overlaps the last one almost entirely.
// Where that one is smooth around a spot - every sample there known, and no bend
// between them worth a colour step - the spot's value is read off it instead of
// being iterated again.
// Returns -1 for "inside", -1.25 for "taken to be inside", a count, or -2 for "iterate it".
vec4 srow(ivec2 c, int j) {
  return vec4(
    texelFetch(uAccTex, ivec3(c.x - 1, c.y + j, uAccLayer), 0).r, texelFetch(uAccTex, ivec3(c.x, c.y + j, uAccLayer), 0).r,
    texelFetch(uAccTex, ivec3(c.x + 1, c.y + j, uAccLayer), 0).r, texelFetch(uAccTex, ivec3(c.x + 2, c.y + j, uAccLayer), 0).r);
}
float reuse() {
  vec2 p = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  vec2 q = vec2(p.x * uAspect, p.y) * uSrcX.x + uSrcX.yz;
  vec2 t = (vec2(q.x / uAspect, q.y) * 0.5 + 0.5) * uSrcSize - 0.5;
  vec2 b = floor(t);
  vec2 f = t - b;
  ivec2 c = ivec2(b);
  if (c.x < 1 || c.y < 1 || c.x > int(uSrcSize.x) - 3 || c.y > int(uSrcSize.y) - 3) return -2.0;
  vec4 r0 = srow(c, -1);
  vec4 r1 = srow(c, 0);
  vec4 r2 = srow(c, 1);
  vec4 r3 = srow(c, 2);
  vec4 mn = min(min(r0, r1), min(r2, r3));
  vec4 mx = max(max(r0, r1), max(r2, r3));
  float lo = min(min(mn.x, mn.y), min(mn.z, mn.w));
  float hi = max(max(mx.x, mx.y), max(mx.z, mx.w));
  if (hi < 0.0 && lo > -1.5) return hi == -1.0 && lo == -1.0 ? -1.0 : -1.25;
  if (lo < 0.0) return -2.0; // something unknown, or the edge of the set
  // how far the samples around the spot stray from straight lines, either way
  vec4 bx = abs(vec4(r1.x - 2.0 * r1.y + r1.z, r1.y - 2.0 * r1.z + r1.w, r2.x - 2.0 * r2.y + r2.z, r2.y - 2.0 * r2.z + r2.w));
  vec4 by = abs(vec4(r0.y - 2.0 * r1.y + r2.y, r1.y - 2.0 * r2.y + r3.y, r0.z - 2.0 * r1.z + r2.z, r1.z - 2.0 * r2.z + r3.z));
  vec4 bm = max(bx, by);
  if (max(max(bm.x, bm.y), max(bm.z, bm.w)) > 0.05 * pow(lo + 1.0, 0.3)) return -2.0;
  return mix(mix(r1.y, r1.z, f.x), mix(r2.y, r2.z, f.x), f.y);
}
// the stored orbit ends where it has come back onto itself: step back one period
int wrap(int m) {
  if (m == uEndA && uPerA > 0) return m - uPerA;
  if (m == uRefLen - 1 && uPerB > 0) return m - uPerB;
  return m;
}

int ctz(int x) {
  return int(floor(log2(float(x & (-x))) + 0.5));
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec2 uv = (gl_FragCoord.xy + uJitter) / uRes * 2.0 - 1.0;
  vec2 cm = uDm + uv * vec2(uAspect, 1.0) * uSm;
  int ec = uEc; // exponent of the pixel's parameter offset: dc = cadd * 2^ec
#if JULIA
  vec2 cadd = vec2(0.0);
#else
  vec2 cadd = cm;
#endif
#if PARAM == 1
  // the lambda map c' = c (2 - c) / 4, as an offset: dc' = dc (1 - C)/2 - dc^2/4
  {
    int e1 = uEc + uKe;
    int e2 = 2 * uEc - 2;
    ec = max(e1, e2);
    cadd = cmul(cm, uKp) * p2(e1 - ec) - csqr(cm) * p2(e2 - ec);
    renorm(cadd, ec);
  }
#endif
  vec2 w;
  int e, m, it;
  float L = 0.0; // log2 of the orbit's derivative so far
  float Ls = 0.0; // L at the last power-of-two iteration: a contraction has to go on to count
  if (uInit == 1) {
    if (uSkipIn != 0) {
      float known = uSkipIn == 1 ? settled(px) : reuse();
      if (known > -1.5) {
        o0 = vec4(0.0);
        o1 = known == -1.0 ? vec4(0.0, 2.0, 0.0, 0.0) : known < 0.0 ? vec4(0.0, 3.0, 0.0, 0.0) : vec4(known, 1.0, 0.0, 0.0);
        return;
      }
    }
#if JULIA
    w = cm;
#else
    w = vec2(0.0);
#endif
    e = ec;
    renorm(w, e);
    m = 0;
    it = 0;
  } else {
    vec4 s0 = texelFetch(uS0, px, 0);
    vec4 s1 = texelFetch(uS1, px, 0);
    if (s1.y != 0.0) {
      o0 = s0;
      o1 = s1;
      return;
    }
    w = s0.xy;
    e = int(s0.z);
    m = int(s0.w);
    it = int(s1.x);
    L = s1.z;
    Ls = s1.w;
    // out of iterations for now; picks up again if the limit is raised
    if (it >= uMaxIter) {
      o0 = s0;
      o1 = s1;
      return;
    }
  }

  int top = LMIN + uBlaLevels - 1;
  float status = 0.0;
  float nu = 0.0;

  vec2 c0 = cadd * p2(ec);
  float invLogPow = 1.0 / log2(float(uPow));
  bool bla = uBlaLevels > 0;

  for (int i = 0; i < uSteps; i++) {
    vec4 Z = orbit(m);

    // Tight loop for the common case late in an orbit: d is an ordinary float
    // and too large for any BLA, so iterate plainly without exponent upkeep.
    if (Z.w == 0.0 && e > -20 && e < 20) {
      w *= p2(e);
      e = 0;
      float pz = 1.0; // product of |z|^2 over the steps not yet added to L
      float pn = 0.0;
      for (; i < uSteps; i++) {
        vec2 z = Z.xy + w;
        float zz = dot(z, z);
        float ww = dot(w, w);
        if (zz > R2) {
          nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(z))) * invLogPow);
          status = 1.0;
          break;
        }
        if (it >= uMaxIter) break;
        if (zz < ww || m == uRefLen - 1 || m == uEndA || ww < 1.0e-18 || (bla && ww < uEps2 * dot(Z.xy, Z.xy))) break;
        pz *= zz;
        pn += 1.0;
#if KIND == 1
        w = absStep(Z, orbit(uUOff + m), w, 0) + c0;
#else
        w = cmul(stepFactor(Z.xy, w), w) + c0;
#endif
        m = wrap(m + 1);
        it++;
        if (pz < 1.0e-12 || pz > 1.0e12 || (it & (it - 1)) == 0) {
          L += uPH * log2(pz) + uLP * pn;
          pz = 1.0;
          pn = 0.0;
          if (L < CONTRACTED && L < Ls - STILL) {
            status = 2.0;
            break;
          }
          if ((it & (it - 1)) == 0) Ls = L;
        }
        Z = orbit(m);
        if (Z.w != 0.0) break;
      }
      L += uPH * log2(pz) + uLP * pn;
      if (status != 0.0 || i >= uSteps) break;
    }

    // escape test, and rebase onto the start of the reference when the pixel
    // comes closer to the origin than to the reference orbit
    float lz2; // log2 |z|^2
    if (Z.w == 0.0) {
      vec2 d = w * p2(e);
      vec2 z = Z.xy + d;
      float zz = dot(z, z);
      lz2 = log2(max(zz, 1.0e-37));
      if (zz > R2) {
        nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(z))) * invLogPow);
        status = 1.0;
        break;
      }
      // Out of iterations: stop before the rebase below. At the end of a truncated
      // orbit that rebase squeezes the pixel into plain floats, and a longer orbit
      // of the same point, when it comes, would find nothing left to carry on with.
      if (it >= uMaxIter) break;
      if (zz < dot(d, d) || m == uRefLen - 1 || m == uEndA) {
        w = z;
        e = 0;
        renorm(w, e);
        m = uBase;
        Z = vec4(0.0, 0.0, float(ZERO_E), 1.0);
      }
    } else {
      // Z is tiny or exactly zero here, but the pixel itself may be far from it
      int Ze = int(Z.z);
      int ez = max(Ze, e);
      vec2 dm = w * p2(e - ez);
      vec2 zm = Z.xy * p2(Ze - ez) + dm;
      vec2 zf = zm * p2(ez);
      float zz = dot(zf, zf);
      lz2 = 2.0 * float(ez) + log2(max(dot(zm, zm), 1.0e-37));
      if (zz > R2) {
        nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(zf))) * invLogPow);
        status = 1.0;
        break;
      }
      if (it >= uMaxIter) break;
#if KIND == 1
      // folds cannot be perturbed around a vanishing Z: always restart from the origin
      if (m != uBase) {
#else
      if (m != uBase && (dot(zm, zm) < dot(dm, dm) || m == uRefLen - 1 || m == uEndA)) {
#endif
        w = zm;
        e = ez;
        renorm(w, e);
        m = uBase;
        Z = vec4(0.0, 0.0, float(ZERO_E), 1.0);
      }
    }

    if (it >= uMaxIter) break;
    int it0 = it;

    bool stepped = false;
    float ww = dot(w, w);
    bool tryBla = bla && m > 0 && m <= uEndA;
    if (tryBla && Z.w == 0.0 && e > -40) tryBla = e < 40 && ww * p2(2 * e) < uEps2 * dot(Z.xy, Z.xy);
    if (tryBla) {
      float ld = ww == 0.0 ? -1.0e30 : float(e) + 0.5 * log2(ww);
      int j0 = m - 1;
      int lvl = j0 == 0 ? top : min(top, ctz(j0));
      for (int k = lvl; k >= LMIN; k--) {
        int idx = k - LMIN;
        int j = j0 >> k;
        if (j >= uBlaCnt[idx]) continue;
        int t = uBlaOff[idx] + j;
        ivec2 tc = ivec2(t & 2047, t >> 11);
        vec4 E = texelFetch(uBlaE, tc, 0);
        if (ld < E.z + uBlaBias) {
#if KIND == 1
          vec4 Am = texelFetch(uBlaA, tc, 0);
          vec4 Bm = texelFetch(uBlaB, tc, 0);
          int ea = int(E.x) + e;
          int eb = int(E.y) + ec;
          int en = max(ea, eb);
          w = vec2(dot(Am.xy, w), dot(Am.zw, w)) * p2(ea - en) + vec2(dot(Bm.xy, cadd), dot(Bm.zw, cadd)) * p2(eb - en);
          e = en;
          renorm(w, e);
          m = wrap(m + (1 << k));
          it += 1 << k;
          // half the log of |det A|: what the plain steps would have added one by one
          L += float(int(E.x)) + 0.5 * log2(max(abs(Am.x * Am.w - Am.y * Am.z), 1.0e-37));
#else
          vec4 AB = texelFetch(uBlaA, tc, 0);
          int ea = int(E.x) + e;
          int eb = int(E.y) + ec;
          int en = max(ea, eb);
          w = cmul(AB.xy, w) * p2(ea - en) + cmul(AB.zw, cadd) * p2(eb - en);
          e = en;
          renorm(w, e);
          m = wrap(m + (1 << k));
          it += 1 << k;
          L += float(int(E.x)) + 0.5 * log2(dot(AB.xy, AB.xy)); // |A| is the derivative along the skipped stretch
#endif
          stepped = true;
          break;
        }
      }
    }

    if (!stepped) {
#if KIND == 1
      if (Z.w == 0.0) {
        w = absStep(Z, orbit(uUOff + m), w, e) + cadd * p2(min(ec - e, 100));
      } else {
        // the reference is exactly 0 here: apply the map to d itself
        int en = max(2 * e, ec);
        w = absMap(w) * p2(2 * e - en) + cadd * p2(ec - en);
        e = en;
      }
#else
      // d' = (Z+d)^p - Z^p + dc
      if (Z.w == 0.0 && ec - e <= 40) {
        w = cmul(stepFactor(Z.xy, w * p2(e)), w) + cadd * p2(ec - e);
      } else {
        // Z or d is out of float range: sum of C(p,k) Z^(p-k) d^k with the larger
        // of the two factored out, so every intermediate value stays near 1
        float mw = max(abs(w.x), abs(w.y));
        if (mw == 0.0) {
          w = cadd;
          e = ec;
        } else {
          int kw = clamp(int(floor(log2(mw))), -120, 120);
          w *= p2(-kw);
          e += kw;
          int Ze = int(Z.z);
          bool zero = Z.x == 0.0 && Z.y == 0.0;
          vec2 acc = vec2(0.0);
          vec2 rp = vec2(1.0, 0.0);
          vec2 T;
          int et;
          if (zero || e >= Ze) {
            vec2 u = zero ? vec2(0.0) : cdiv(Z.xy, w) * p2(Ze - e);
            float bin = 1.0;
            for (int k = uPow; k >= 1; k--) {
              acc += bin * rp;
              rp = cmul(rp, u);
              bin = bin * float(k) / float(uPow - k + 1);
            }
            vec2 wp = w;
            for (int j = 1; j < uPow; j++) wp = cmul(wp, w);
            T = cmul(wp, acc);
            et = uPow * e;
          } else {
            vec2 v = cdiv(w, Z.xy) * p2(e - Ze);
            float bin = float(uPow);
            for (int k = 1; k <= uPow; k++) {
              acc += bin * rp;
              rp = cmul(rp, v);
              bin = bin * float(uPow - k) / float(k + 1);
            }
            vec2 zp = vec2(1.0, 0.0);
            for (int j = 1; j < uPow; j++) zp = cmul(zp, Z.xy);
            T = cmul(cmul(zp, w), acc);
            et = (uPow - 1) * Ze + e;
          }
          int en = max(et, ec);
          w = T * p2(et - en) + cadd * p2(ec - en);
          e = en;
        }
      }
#endif
      renorm(w, e);
      m = wrap(m + 1);
      it++;
#if JULIA
      L += uLP + uPH * lz2;
#else
      if (it0 > 0) L += uLP + uPH * lz2; // the very first step starts on the critical point itself
#endif
    }
    if (L < CONTRACTED && L < Ls - STILL) {
      status = 2.0;
      break;
    }
    if ((it0 ^ it) > it0) Ls = L; // passed a power of two
  }

  // a finished pixel keeps the iteration it finished at, for the "is anything still finishing" test
  o0 = status != 0.0 ? vec4(float(it), 0.0, 0.0, 0.0) : vec4(w, float(e), float(m));
  o1 = vec4(status == 1.0 ? nu : float(it), status, L, Ls);
}`;

// Collapses the state of a job into one float per pixel of its layer:
//   v >= 0      smooth iteration count of an escaped pixel
//   v == -1     inside the set, proven
//   v == -1.25  taken to be inside: a picture at rest went quiet with it unfinished
//               (status 3 carries that over from an earlier picture)
//   v <= -2     not known yet: -(iterations reached so far) - 2
// A limit merely reached says nothing: such a pixel stays unknown.
export const RESOLVE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uS1;
uniform int uFinal; // nothing more will be learnt: whatever has not finished is inside
out vec4 o;
void main() {
  vec4 s = texelFetch(uS1, ivec2(gl_FragCoord.xy), 0);
  float v = s.y == 1.0 ? s.x : s.y == 2.0 ? -1.0 : s.y == 3.0 || uFinal == 1 ? -1.25 : -(s.x + 2.0);
  o = vec4(v, 0.0, 0.0, 1.0);
}`;

// Occlusion-query probes over the job state.
// COUNT passes where a pixel is still running and has iterations left under the limit.
export const COUNT = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uS1;
uniform float uN;
out vec4 o;
void main() {
  vec4 s = texelFetch(uS1, ivec2(gl_FragCoord.xy), 0);
  if (s.y != 0.0 || s.x >= uN) discard;
  o = vec4(1.0);
}`;

// EVENTS counts, for each 16x16 block of the job, the pixels that finished after
// iteration uT, and those still neither escaped nor proven inside. Read back, it
// tells whether late finishers are stray specks on a boundary, dust that sheds
// pixels at any count whatever, or whole patches coming in together (finishing
// within a few iterations of each other), and how much is left that could come out.
export const EVENTS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uS0;
uniform sampler2D uS1;
uniform float uT;
uniform ivec2 uSize;
out vec4 o;
void main() {
  ivec2 b = ivec2(gl_FragCoord.xy) * 16;
  float n = 0.0;
  float u = 0.0;
  float lo = 1.0e30;
  float hi = 0.0;
  for (int j = 0; j < 16; j++) {
    for (int i = 0; i < 16; i++) {
      ivec2 p = b + ivec2(i, j);
      if (p.x >= uSize.x || p.y >= uSize.y) continue;
      float st = texelFetch(uS1, p, 0).y;
      if (st == 0.0) u += 1.0;
      else if (st == 1.0) {
        float at = texelFetch(uS0, p, 0).x;
        if (at > uT) {
          n += 1.0;
          lo = min(lo, at);
          hi = max(hi, at);
        }
      }
    }
  }
  // late finishers, still unknown, and between which iterations the late ones finished
  o = vec4(n, u, lo, hi);
}`;

// Coarse point-sampled copy of the job state for CPU-side statistics.
export const PROBE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uS1;
uniform vec2 uSize;
uniform vec2 uGrid;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uGrid;
  o = texelFetch(uS1, ivec2(uv * uSize), 0);
}`;

/** Layers the picture is composed of. */
export const MAXL = 6;

// Final image. The picture is a short stack of layers, each a finished or
// half-finished rendering of some view, kept in fractal coordinates. A screen
// pixel takes the best layer that knows its spot and falls through to the next
// where one does not: a spot that was ever computed is never drawn empty, and a
// layer still being computed sharpens the picture as its pixels come in. What no
// layer knows is left for FILL.
export const DISPLAY = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
#define MAXL ${MAXL}

uniform sampler2DArray uStack;
uniform int uCount; // layers to draw, best first
uniform vec4 uLX[MAXL]; // ratio, offset.xy, slot in the stack
uniform vec4 uLS[MAXL]; // size.xy, opacity, width of the soft edge in texels (0: none)
uniform sampler2DArray uAA; // extra jittered samples of the first layer
uniform int uAACount;
uniform int uDirect; // 1: every spot is known, draw straight to the screen; 0: colour and coverage for FILL
uniform sampler2D uPal;
uniform vec2 uRes;
uniform float uAspect;
uniform float uPhase;
uniform float uDensity;
out vec4 o;

vec3 shade(float nu) {
  if (nu < 0.0) return vec3(0.0);
  return textureLod(uPal, vec2(pow(nu, 0.7) * uDensity + uPhase, 0.5), 0.0).rgb;
}

vec2 toTex(vec2 pa, vec4 x, vec2 size) {
  vec2 q = pa * x.x + x.yz;
  return (vec2(q.x / uAspect, q.y) * 0.5 + 0.5) * size - 0.5;
}

// cubic B-spline weights for the four texels around a sample position
vec4 bsw(float f) {
  float f2 = f * f;
  float f3 = f2 * f;
  return vec4(1.0 - 3.0 * f + 3.0 * f2 - f3, 4.0 - 6.0 * f2 + 3.0 * f3, 1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3, f3) / 6.0;
}

void main() {
  vec2 p = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  vec2 pa = vec2(p.x * uAspect, p.y);
  vec3 col = vec3(0.0); // what the layers know, front to back
  float A = 0.0;
  for (int l = 0; l < uCount; l++) {
    vec4 X = uLX[l];
    vec4 S = uLS[l];
    vec2 t = toTex(pa, X, S.xy);
    if (t.x < -0.5 || t.y < -0.5 || t.x > S.x - 0.5 || t.y > S.y - 0.5) continue;
    ivec2 hi = ivec2(S.xy) - 1;
    int slot = int(X.w);
    float a = S.z;
    if (S.w > 0.0) a *= clamp(min(min(t.x + 0.5, S.x - 0.5 - t.x), min(t.y + 0.5, S.y - 0.5 - t.y)) / S.w, 0.0, 1.0);
    ivec2 n = clamp(ivec2(floor(t + 0.5)), ivec2(0), hi);
    vec3 mean = vec3(0.0);
    if (l == 0 && uAACount > 0) {
      // several samples of this very view: their plain average is the picture.
      // (A sample that ran out of its own, lower limit says nothing and is left out.)
      mean = shade(texelFetch(uStack, ivec3(n, slot), 0).r);
      float cnt = 1.0;
      for (int i = 0; i < uAACount; i++) {
        float v = texelFetch(uAA, ivec3(n, i), 0).r;
        if (v > -1.5) {
          mean += shade(v);
          cnt += 1.0;
        }
      }
      mean /= cnt;
      if (uAACount >= 3) {
        col += (1.0 - A) * a * mean;
        A += (1.0 - A) * a;
        if (A > 0.995) break;
        continue;
      }
    }
    // One sample per pixel is pure noise wherever detail is finer than a pixel, so a
    // layer is read through a cubic B-spline: it stands in for the averaging more
    // samples will bring, and keeps a magnified layer smooth.
    vec2 b = floor(t);
    vec2 f = t - b;
    ivec2 c = ivec2(b);
    vec4 wx = bsw(f.x);
    vec4 wy = bsw(f.y);
    vec3 cs = vec3(0.0);
    float k = 0.0;
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        float w = wx[i] * wy[j];
        if (w < 1.0e-5) continue;
        float v = texelFetch(uStack, ivec3(clamp(c + ivec2(i - 1, j - 1), ivec2(0), hi), slot), 0).r;
        if (v > -1.5) {
          cs += w * shade(v);
          k += w;
        }
      }
    }
    if (l == 0 && uAACount > 0) cs = mix(cs, mean * k, float(uAACount) / 3.0);
    col += (1.0 - A) * a * cs;
    A += (1.0 - A) * a * k;
    if (A > 0.995) break;
  }
  if (uDirect == 0) {
    o = vec4(col, A);
    return;
  }
  // palette is stored linear; encode for display and dither away banding
  col = pow(col, vec3(1.0 / 2.2));
  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  o = vec4(col + (n - 0.5) / 255.0, 1.0);
}`;

// Spots that no layer knows yet take on the colours around them, nearest first
// (push-pull): the composed picture is halved again and again with the unknown
// left out (DOWN), then each size fills its gaps from the next coarser one (UP),
// and the screen takes what is still missing from the finest of those (FILL).
// Such a spot stays calm until it is computed - no guess flashing through the
// palette, no hole - and is held a little darker, as what has not come out yet
// lies deeper than what has.
// All of them carry rgb = colour times coverage, a = coverage.
export const DOWN = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uSrc;
uniform vec2 uSize;
out vec4 o;
void main() {
  o = texture(uSrc, gl_FragCoord.xy / uSize);
}`;

export const UP = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uOwn;
uniform sampler2D uUp; // the next coarser size, already filled
uniform vec2 uSize;
uniform int uTop; // the coarsest size has nothing to fill from
out vec4 o;
void main() {
  vec4 own = texelFetch(uOwn, ivec2(gl_FragCoord.xy), 0);
  vec4 up = uTop == 1 ? vec4(0.0) : texture(uUp, gl_FragCoord.xy / uSize);
  float has = up.a > 1.0e-4 ? 1.0 : 0.0;
  vec3 c = has > 0.0 ? up.rgb / up.a : vec3(0.0);
  o = vec4(own.rgb + (1.0 - own.a) * c, own.a + (1.0 - own.a) * has);
}`;

export const FILL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uC;
uniform sampler2D uUp;
uniform vec2 uRes;
out vec4 o;
void main() {
  vec4 c = texelFetch(uC, ivec2(gl_FragCoord.xy), 0);
  vec3 col = c.rgb;
  if (c.a < 0.995) {
    vec4 up = texture(uUp, gl_FragCoord.xy / uRes);
    if (up.a > 1.0e-4) col += (1.0 - c.a) * 0.6 * up.rgb / up.a;
  }
  col = pow(max(col, vec3(0.0)), vec3(1.0 / 2.2));
  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  o = vec4(col + (n - 0.5) / 255.0, 1.0);
}`;

// Perturbation for the formulas that are neither z^p + c nor a folded quadratic.
// Same state and scaling as ITERATE (d = w * 2^e against a reference orbit), but
// each formula brings its own step, and there is no BLA table.
//   FORMULA     which map (see custom.ts)
//   REBASE_CRIT the orbit texture holds the distance to a critical point, so a
//               pixel that comes nearer to it than to the reference restarts there
//   HAS_U       a second quantity per pixel (wu, same scale as w) and per reference index
//   USES_C      the pixel is the parameter; otherwise it is the starting point
export function customSrc(formula: number, info: { rebaseCrit: boolean; hasU: boolean; usesC: boolean }): string {
  const defs =
    `#define FORMULA ${formula}\n#define REBASE_CRIT ${info.rebaseCrit ? 1 : 0}\n` +
    `#define HAS_U ${info.hasU ? 1 : 0}\n#define USES_C ${info.usesC ? 1 : 0}`;
  return CUSTOM.replace('//DEFINES', defs);
}

const CUSTOM = `#version 300 es
//DEFINES
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;

uniform sampler2D uS0;
uniform sampler2D uS1;
uniform sampler2D uOrbit;
uniform int uInit;
uniform int uRefLen;
uniform int uMaxIter;
uniform int uSteps;
uniform int uEc;
uniform int uPow;
uniform int uEndA; // last index of the segment pixels start on
uniform int uBase; // index pixels restart from
uniform int uUOff; // where the second reference quantity starts in the orbit texture
uniform int uPerA; // period of the first reference segment once it has settled on a cycle (0 if not)
uniform int uPerB; // likewise for the last segment
uniform sampler2DArray uAccTex; // finished picture, for skipping what is certainly inside
uniform int uSkipIn; // 1: an extra sample of a finished picture, 2: a picture overlapping an earlier one
uniform int uAccLayer;
uniform vec4 uSrcX; // where this job lies in that earlier picture: ratio, offset.xy
uniform vec2 uSrcSize;
uniform ivec2 uAccMax;
uniform vec2 uRes;
uniform vec2 uJitter;
uniform vec2 uDm;
uniform float uAspect;
uniform float uSm;
uniform vec4 uCref; // reference parameter as two floats per component (hi, lo)
uniform vec4 uLref; // its logarithm, likewise (tetration)

layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;

const float R2 = 1.0e6;
const float LLR = 4.3169906; // log2(log2(R2))

float p2(int k) {
  if (k < -126) return 0.0;
  return uintBitsToFloat(uint(min(k, 127) + 127) << 23u);
}
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 csqr(vec2 a) { return vec2(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y); }
vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
vec2 cexp(vec2 a) { return exp(a.x) * vec2(cos(a.y), sin(a.y)); }
vec2 clog(vec2 a) { return vec2(0.5 * log(dot(a, a)), atan(a.y, a.x)); }
// product with a constant known to twice float precision
vec2 cmulhl(vec4 c, vec2 v) { return cmul(c.xy, v) + cmul(c.zw, v); }
const vec2 ONE = vec2(1.0, 0.0);

void renorm(inout vec2 w, inout vec2 wu, inout int e) {
  float m = max(abs(w.x), abs(w.y));
  if (m == 0.0) m = max(abs(wu.x), abs(wu.y));
  if (m > 1048576.0 || (m < 9.5367431640625e-7 && m > 0.0)) {
    int k = clamp(int(floor(log2(m))), -120, 120);
    float f = p2(-k);
    w *= f;
    wu *= f;
    e += k;
  }
}

vec4 orbit(int m) { return texelFetch(uOrbit, ivec2(m & 2047, m >> 11), 0); }
// Extra (jittered) samples exist to smooth edges and dust. Where the first sample
// shows a pixel and all its neighbours inside, or all escaped at practically the
// same count, another sample would say the same: reuse the value, skip the work.
// Returns -1 for "inside", a count for "same as before", -2 for "iterate it".
float settled(ivec2 px) {
  float lo = 1.0e30;
  float hi = -1.0e30;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float v = texelFetch(uAccTex, ivec3(clamp(px + ivec2(i, j), ivec2(0), uAccMax), uAccLayer), 0).r;
      if (v < -1.5) return -2.0;
      lo = min(lo, v);
      hi = max(hi, v);
    }
  }
  if (hi < 0.0) return -1.0;
  if (lo >= 0.0 && hi - lo < 0.25) return texelFetch(uAccTex, ivec3(px, uAccLayer), 0).r;
  return -2.0;
}
// While the view moves, a new picture overlaps the last one almost entirely.
// Where that one is smooth around a spot - every sample there known, and no bend
// between them worth a colour step - the spot's value is read off it instead of
// being iterated again.
// Returns -1 for "inside", -1.25 for "taken to be inside", a count, or -2 for "iterate it".
vec4 srow(ivec2 c, int j) {
  return vec4(
    texelFetch(uAccTex, ivec3(c.x - 1, c.y + j, uAccLayer), 0).r, texelFetch(uAccTex, ivec3(c.x, c.y + j, uAccLayer), 0).r,
    texelFetch(uAccTex, ivec3(c.x + 1, c.y + j, uAccLayer), 0).r, texelFetch(uAccTex, ivec3(c.x + 2, c.y + j, uAccLayer), 0).r);
}
float reuse() {
  vec2 p = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  vec2 q = vec2(p.x * uAspect, p.y) * uSrcX.x + uSrcX.yz;
  vec2 t = (vec2(q.x / uAspect, q.y) * 0.5 + 0.5) * uSrcSize - 0.5;
  vec2 b = floor(t);
  vec2 f = t - b;
  ivec2 c = ivec2(b);
  if (c.x < 1 || c.y < 1 || c.x > int(uSrcSize.x) - 3 || c.y > int(uSrcSize.y) - 3) return -2.0;
  vec4 r0 = srow(c, -1);
  vec4 r1 = srow(c, 0);
  vec4 r2 = srow(c, 1);
  vec4 r3 = srow(c, 2);
  vec4 mn = min(min(r0, r1), min(r2, r3));
  vec4 mx = max(max(r0, r1), max(r2, r3));
  float lo = min(min(mn.x, mn.y), min(mn.z, mn.w));
  float hi = max(max(mx.x, mx.y), max(mx.z, mx.w));
  if (hi < 0.0 && lo > -1.5) return hi == -1.0 && lo == -1.0 ? -1.0 : -1.25;
  if (lo < 0.0) return -2.0; // something unknown, or the edge of the set
  // how far the samples around the spot stray from straight lines, either way
  vec4 bx = abs(vec4(r1.x - 2.0 * r1.y + r1.z, r1.y - 2.0 * r1.z + r1.w, r2.x - 2.0 * r2.y + r2.z, r2.y - 2.0 * r2.z + r2.w));
  vec4 by = abs(vec4(r0.y - 2.0 * r1.y + r2.y, r1.y - 2.0 * r2.y + r3.y, r0.z - 2.0 * r1.z + r2.z, r1.z - 2.0 * r2.z + r3.z));
  vec4 bm = max(bx, by);
  if (max(max(bm.x, bm.y), max(bm.z, bm.w)) > 0.05 * pow(lo + 1.0, 0.3)) return -2.0;
  return mix(mix(r1.y, r1.z, f.x), mix(r2.y, r2.z, f.x), f.y);
}
// the stored orbit ends where it has come back onto itself: step back one period
int wrap(int m) {
  if (m == uEndA && uPerA > 0) return m - uPerA;
  if (m == uRefLen - 1 && uPerB > 0) return m - uPerB;
  return m;
}
// a reference value as an ordinary float (values kept as mantissa + exponent may underflow to 0)
vec2 zval(vec4 Z) { return Z.w == 0.0 ? Z.xy : Z.xy * p2(int(Z.z)); }

#if FORMULA == 7
vec2 cexpm1(vec2 a) {
  if (dot(a, a) > 0.01) return cexp(a) - ONE;
  vec2 a2 = csqr(a);
  return a + 0.5 * a2 + cmul(a, a2) / 6.0 + csqr(a2) / 24.0;
}
// Tetration: the next value z' = exp(R) cis(ph) is small. Then c^z' is close to 1
// and c^(c^z') close to c: two steps on, the orbit is back beside the reference's
// start, displaced by  dc + c (exp(l (exp(z' l) - 1)) - 1).  Going there puts the
// pixel back on the reference with all its digits; iterated by itself in floats it
// would keep only the few that "1 + something tiny" leaves.
void backToStart(float R, float ph, vec2 cm, inout vec2 w, inout int e, inout int m, inout int it) {
  float Rc = max(R, -6.0e5);
  int ez = int(floor(Rc / 0.69314718));
  vec2 zt = exp(Rc - float(ez) * 0.69314718) * vec2(cos(ph), sin(ph));
  vec2 C = uCref.xy + uCref.zw;
  vec2 Lf = uLref.xy + uLref.zw;
  vec2 back; // c (exp(l (exp(z' l) - 1)) - 1), as back * 2^eb
  int eb = ez;
  if (ez < -100) back = cmul(cmul(C, csqr(Lf)), zt); // far too small for anything but the first order
  else {
    back = cmul(C, cexpm1(cmul(Lf, cexpm1(cmul(zt * p2(ez), Lf)))));
    eb = 0;
  }
  int en = max(uEc, eb);
  w = cm * p2(uEc - en) + back * p2(eb - en);
  e = en;
  m = 0;
  it += 2; // the caller counts the third
}
#endif

float escNu(int it, vec2 z) {
  float m = max(abs(z.x), abs(z.y));
  if (!(m < 1.0e30)) return float(it); // overflowed: no fraction to recover
  vec2 q = z / m;
  return max(0.0, float(it) + 1.0 + LLR - log2(2.0 * log2(m) + log2(dot(q, q))));
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec2 uv = (gl_FragCoord.xy + uJitter) / uRes * 2.0 - 1.0;
  vec2 cm = uDm + uv * vec2(uAspect, 1.0) * uSm;
#if USES_C
  vec2 cadd = cm;
#else
  vec2 cadd = vec2(0.0);
#endif
  vec2 Cf = uCref.xy + uCref.zw;

  vec2 w;
  vec2 wu = vec2(0.0);
  int e, m, it;
  if (uInit == 1) {
    if (uSkipIn != 0) {
      float known = uSkipIn == 1 ? settled(px) : reuse();
      if (known > -1.5) {
        o0 = vec4(0.0);
        o1 = known == -1.0 ? vec4(0.0, 2.0, 0.0, 0.0) : known < 0.0 ? vec4(0.0, 3.0, 0.0, 0.0) : vec4(known, 1.0, 0.0, 0.0);
        return;
      }
    }
    e = uEc;
    m = 0;
    it = 0;
#if FORMULA == 1
    w = vec2(-cm.y, cm.x);
#elif FORMULA == 2
    w = cm;
    wu = cm;
#elif FORMULA == 3
    w = vec2(0.0);
    wu = cm;
#elif FORMULA == 4 || FORMULA == 7
    w = cm;
#else
    w = vec2(0.0);
#endif
#if FORMULA == 5
    wu = vec2(1.0, 0.0); // previous step size, for smooth colouring
#else
    renorm(w, wu, e);
#endif
  } else {
    vec4 s0 = texelFetch(uS0, px, 0);
    vec4 s1 = texelFetch(uS1, px, 0);
    if (s1.y != 0.0) {
      o0 = s0;
      o1 = s1;
      return;
    }
    w = s0.xy;
    e = int(s0.z);
    m = int(s0.w);
    it = int(s1.x);
    wu = s1.zw;
    // out of iterations for now; picks up again if the limit is raised
    if (it >= uMaxIter) {
      o0 = s0;
      o1 = s1;
      return;
    }
  }

  int last = uRefLen - 1;
  float status = 0.0;
  float nu = 0.0;
  vec2 ZB = zval(orbit(uBase)); // where pixels restart
#if FORMULA == 5
  vec2 cpix = Cf + cm * p2(uEc); // the pixel's own c, for when it goes its own way
#endif

#if FORMULA == 7
  // log(1 + dc/C): exact while it fits a float, first order (kept scaled) when it does not
  vec2 dlm = cdiv(cm, Cf);
  vec2 dlx = dlm * p2(uEc);
  vec2 dl = dot(dlx, dlx) < 1.0e-6 ? dlx - 0.5 * csqr(dlx) : clog(ONE + dlx);
  vec2 lpix = clog(Cf + cm * p2(uEc)); // the pixel's own log c, for when it goes its own way
#endif

  for (int i = 0; i < uSteps; i++) {
#if FORMULA == 4
    // Newton's map has a pole at 0 and throws whatever comes near it far away. A
    // pixel that parted from the reference cannot be kept as "reference + d" then:
    // one of them may be astronomically larger than the other. So the pixel is a
    // perturbation only while it is still close to the reference (in ratio); after
    // that it is iterated on its own, as z = w * 2^e (m < 0 marks this).
    bool own = m < 0;
    vec2 Zm = vec2(0.0); // reference as a mantissa times 2^kz
    int kz = 0;
    vec2 zm = w; // the pixel in the same units
    if (!own) {
      vec4 Zt = orbit(m);
      if (Zt.w != 0.0) {
        Zm = Zt.xy;
        kz = int(Zt.z);
      } else {
        kz = int(floor(log2(max(abs(Zt.x), abs(Zt.y))))) + 1;
        Zm = Zt.xy * p2(-kz);
      }
      vec2 dm = w * p2(e - kz);
      zm = Zm + dm;
      if (kz < -1000000 || dot(dm, dm) > 2.44140625e-4 * dot(zm, zm)) {
        if (kz < -1000000) zm = w; else e = kz;
        w = zm;
        m = -1;
        own = true;
      }
    }
    if (own) {
      // keep w near 1 so its powers stay in range
      float mw = max(abs(w.x), abs(w.y));
      if (mw > 0.0) {
        int k = clamp(int(floor(log2(mw))), -120, 120);
        w *= p2(-k);
        e += k;
      }
      zm = w;
      kz = e;
    }
    if (kz > -40 && kz < 40) {
      vec2 z = zm * p2(kz);
      vec2 zq = ONE;
      for (int k = 1; k < uPow; k++) zq = cmul(zq, z);
      vec2 f = cmul(zq, z) - ONE;
      float ff = dot(f, f);
      if (ff < 1.0e-5) {
        float k = floor(atan(z.y, z.x) / 6.2831853 * float(uPow) + 0.5);
        if (k < 0.0) k += float(uPow);
        nu = (float(it) - log2(log(max(ff, 1.0e-11)) / log(1.0e-5))) * 7.0 + k * 97.0;
        status = 1.0;
        break;
      }
    }
    if (it >= uMaxIter) break;
    float pw = float(uPow);
    int q = uPow - 1;
    if (own) {
      // N(z) = z (p-1)/p + 1 / (p z^q), each term with its own exponent
      vec2 wq = ONE;
      for (int k = 1; k < uPow; k++) wq = cmul(wq, w);
      int e2 = -q * e;
      int en = max(e, e2);
      w = w * ((pw - 1.0) / pw) * p2(e - en) + cdiv(ONE, pw * wq) * p2(e2 - en);
      e = en;
    } else {
      // d' = d ((p-1)/p - S / (p z^q Z^q)),  S = sum of z^j Z^(q-1-j); in units of 2^kz
      vec2 Zq = ONE;
      vec2 zq = ONE;
      for (int k = 1; k < uPow; k++) {
        Zq = cmul(Zq, Zm);
        zq = cmul(zq, zm);
      }
      vec2 S = ONE;
      vec2 zp = ONE;
      for (int k = 2; k < uPow; k++) {
        zp = cmul(zp, Zm);
        S = cmul(S, zm) + zp;
      }
      vec2 big = cdiv(S, pw * cmul(zq, Zq)); // carries 2^(-(q+1) kz)
      int e2 = e - (q + 1) * kz;
      int en = max(e, e2);
      w = w * ((pw - 1.0) / pw) * p2(e - en) - cmul(big, w) * p2(e2 - en);
      e = en;
      m = min(wrap(m + 1), last);
    }
    renorm(w, wu, e);
    it++;
#elif FORMULA == 7
    // Tetration: z' = exp(t), t = z ln c. Everything is decided on the exponent t,
    // kept for the reference exactly and for the pixel as t = T + a:
    //   - the next value is past the bailout when Re t >= ln(10^12);
    //   - while a is small the pixel stays a perturbation, d' = Z' (exp(a) - 1);
    //   - once it is not, the pixel has parted from the reference by a large ratio
    //     and is iterated by itself from z' = exp(T + a), in plain floats (m < 0).
    // Nothing is ever formed as a difference of two huge numbers.
    if (m >= 0) {
      float s = p2(e);
      vec2 d = w * s;
      vec2 z = zval(orbit(m)) + d;
      vec2 dlw = uEc < -20 ? dlm * p2(uEc - e) : dl * p2(clamp(-e, -126, 126));
      vec2 aw = cmulhl(uLref, w) + cmul(z, dlw);
      vec2 a = aw * s;
      vec4 Um = orbit(uUOff + m); // Re T - ln(10^12) as mantissa and exponent, then Re T and Im T mod 2 pi
      int Ue = int(Um.y);
      int en = max(Ue, e);
      if (Um.x * p2(Ue - en) + aw.x * p2(e - en) >= 0.0) {
        nu = float(it + 1) * 3.0 + 3.0;
        status = 1.0;
        break;
      }
      if (it >= uMaxIter) break;
      if (dot(a, a) < 2.44140625e-4 && dot(d, d) <= 2.44140625e-4 * dot(z, z) && !(m == last && uPerB == 0)) {
        int nx = min(wrap(m + 1), last);
        vec2 a2 = csqr(a);
        vec2 g = ONE + 0.5 * a + a2 / 6.0 + cmul(a, a2) / 24.0; // (exp(a) - 1) / a
        w = cmul(zval(orbit(nx)), cmul(aw, g));
        renorm(w, wu, e);
        m = nx;
      } else {
        float R = Um.z + a.x;
        float ph = Um.w + a.y;
        if (R < -4.2) {
          backToStart(R, ph, cm, w, e, m, it);
          renorm(w, wu, e);
        } else {
          w = exp(R) * vec2(cos(ph), sin(ph));
          e = 0;
          m = -1;
        }
      }
    } else {
      vec2 t = cmul(w, lpix);
      if (t.x >= 27.6310211) {
        nu = float(it + 1) * 3.0 + 3.0;
        status = 1.0;
        break;
      }
      if (it >= uMaxIter) break;
      if (t.x < -4.2) {
        backToStart(t.x, t.y, cm, w, e, m, it);
        renorm(w, wu, e);
      } else {
        w = cexp(t);
      }
    }
    it++;
#else
    vec4 Zt = orbit(max(m, 0));
    vec2 Z = zval(Zt);
    float s = p2(e);
    vec2 d = w * s;
    vec2 z = Z + d;
    float zz = dot(z, z);
#if FORMULA == 5
    // Nova's Newton step has a pole at z = 0 (y = -1) that throws orbits far away.
    // As with Newton: a pixel that has parted from the reference (in ratio) is
    // iterated by itself from then on, as z = w * 2^e, so that nothing is ever a
    // difference of two huge numbers (m < 0 marks this).
    if (m >= 0) {
      vec2 zf = ONE + z;
      if (dot(d, d) > 2.44140625e-4 * dot(zf, zf)) {
        w = zf;
        e = 0;
        m = -1;
      }
    }
    if (m < 0) {
      float mw = max(abs(w.x), abs(w.y));
      if (mw > 0.0) {
        int k = clamp(int(floor(log2(mw))), -120, 120);
        w *= p2(-k);
        e += k;
      }
      if (it >= uMaxIter) break;
      vec2 zo = w * p2(e);
      // z' = (2/3) z + 1 / (3 z^2) + c, each term with its own exponent
      int e2 = -2 * e;
      int en = max(max(e, e2), 0);
      w = w * (2.0 / 3.0) * p2(e - en) + cdiv(ONE, 3.0 * csqr(w)) * p2(e2 - en) + cpix * p2(-en);
      e = en;
      vec2 stp = w * p2(e) - zo;
      float dd = dot(stp, stp);
      if (dd < 1.0e-8) {
        float frac = clamp(log(1.0e-8 / wu.x) / log(dd / wu.x), 0.0, 1.0);
        nu = (float(it) + (frac == frac ? frac : 0.0)) * 5.0;
        status = 1.0;
        break;
      }
      wu.x = dd;
      it++;
      continue;
    }
#endif

    // ---- has the pixel finished?
#if FORMULA == 4
    vec2 zq = ONE; // z^(p-1)
    for (int k = 1; k < uPow; k++) zq = cmul(zq, z);
    vec2 f = cmul(zq, z) - ONE;
    float ff = dot(f, f);
    if (ff < 1.0e-5) {
      float k = floor(atan(z.y, z.x) / 6.2831853 * float(uPow) + 0.5);
      if (k < 0.0) k += float(uPow);
      nu = (float(it) - log2(log(max(ff, 1.0e-11)) / log(1.0e-5))) * 7.0 + k * 97.0;
      status = 1.0;
      break;
    }
#elif FORMULA == 7
    if (!(zz < 1.0e24)) {
      nu = float(it) * 3.0 + 3.0;
      status = 1.0;
      break;
    }
#elif FORMULA != 5
    if (!(zz <= R2)) {
      nu = escNu(it, z);
      status = 1.0;
      break;
    }
#endif
#if FORMULA == 6
    vec2 z1 = z - ONE;
    float d1 = dot(z1, z1);
    if (d1 < 1.0e-4) {
      // 1 is superattracting: the same smooth count as for Newton
      nu = 60.0 + 2.0 * (float(it) - log2(log(max(d1, 1.0e-9)) / log(1.0e-4)));
      status = 1.0;
      break;
    }
#endif
    if (it >= uMaxIter) break;

    // ---- keep d small: restart from the critical point when the pixel is nearer to it
    bool atZero = false; // the reference value here is exactly 0
#if REBASE_CRIT
    if (Zt.w != 0.0) {
      if (m != 0) {
        int Ze = int(Zt.z);
        int ez = max(Ze, e);
        w = Zt.xy * p2(Ze - ez) + w * p2(e - ez);
        e = ez;
        renorm(w, wu, e);
        m = 0;
      }
      atZero = true;
      Z = vec2(0.0);
      s = p2(e);
      d = w * s;
      z = d;
    } else if (zz < dot(d, d)) {
      w = z;
      e = 0;
      renorm(w, wu, e);
      m = 0;
      atZero = true;
      Z = vec2(0.0);
      s = p2(e);
      d = w * s;
      z = d;
    }
#endif

    // ---- restart from the base orbit, carrying the whole value: always when this
    // segment ran out by escaping, and for two-component maps also once the pixel
    // has drifted well away and sits nearer the base. (While d is still tiny that
    // would throw away the pixel's own digits in the second component.)
    bool restart = ((m == last && uPerB == 0) || (m == uEndA && uPerA == 0)) && last > 0;
    vec2 zb = z - ZB;
#if HAS_U && FORMULA != 6
    if (m != uBase && dot(d, d) > 1.0e-6 && dot(zb, zb) < dot(d, d)) restart = true;
#endif
    if (restart) {
      w = zb;
#if HAS_U && FORMULA != 6
      wu = (orbit(uUOff + m).xy + wu * s) - orbit(uUOff + uBase).xy;
#endif
      e = 0;
      renorm(w, wu, e);
      m = uBase;
      Zt = orbit(m);
      Z = zval(Zt);
      s = p2(e);
      d = w * s;
      z = Z + d;
#if REBASE_CRIT
      atZero = Zt.w != 0.0;
#endif
    }
    int mn = min(wrap(m + 1), last);
    vec2 cs = cadd * p2(min(uEc - e, 100));

    // ---- one step of d
#if FORMULA == 0
    // lambda in y = z - 1/2:  d' = -C (2Y + d) d + dc (1/4 - y^2)
    if (atZero) {
      int en = max(2 * e, uEc);
      w = -cmulhl(uCref, csqr(w)) * p2(2 * e - en) + cmul(cadd, vec2(0.25, 0.0) - csqr(d)) * p2(uEc - en);
      e = en;
    } else {
      w = -cmulhl(uCref, cmul(2.0 * Z + d, w)) + cmul(cs, vec2(0.25, 0.0) - csqr(z));
    }
#elif FORMULA == 1
    vec2 wn = 2.0 * cmul(Z, w) + s * csqr(w) - 0.5 * wu;
    wu = w;
    w = wn;
#elif FORMULA == 2
    vec2 wn = 2.0 * cmul(Z, w) + s * csqr(w) + wu + cs;
    wu = w;
    w = wn;
#elif FORMULA == 3
    w = 2.0 * cmul(Z, w) + s * csqr(w) + wu;
    wu = 0.5 * wu + w;
#elif FORMULA == 4
    // Newton: d' = d ((p-1)/p - S / (p z^q Z^q)),  q = p - 1,  S = sum of z^j Z^(q-1-j)
    vec2 Zq = ONE;
    for (int k = 1; k < uPow; k++) Zq = cmul(Zq, Z);
    vec2 S = ONE;
    vec2 zp = ONE;
    for (int k = 2; k < uPow; k++) {
      zp = cmul(zp, Z);
      S = cmul(S, z) + zp;
    }
    float pw = float(uPow);
    w = cmul(vec2((pw - 1.0) / pw, 0.0) - cdiv(S, pw * cmul(zq, Zq)), w);
#elif FORMULA == 5
    // Nova in y = z - 1:  g(y) = y^2 (3 + 2y) / (3 (1 + y)^2) + c
    float eOld = float(e);
    vec2 wOld = w;
    if (atZero) {
      int en = max(2 * e, uEc);
      vec2 hh = cdiv(3.0 * ONE + 2.0 * d, 3.0 * csqr(ONE + d));
      w = cmul(csqr(w), hh) * p2(2 * e - en) + cadd * p2(uEc - en);
      e = en;
    } else {
      vec2 a = ONE + Z;
      vec2 b = ONE + z;
      vec2 h = cdiv(3.0 * ONE + 2.0 * z, 3.0 * csqr(b));
      vec2 t = cmul(2.0 * Z + d, h) - cdiv(cmul(csqr(Z), a + b + 2.0 * cmul(a, b)), 3.0 * cmul(csqr(a), csqr(b)));
      w = cmul(t, w) + cs;
    }
    // settled when the step itself has become tiny
    vec2 stp = (zval(orbit(mn)) - Z) + w * p2(e) - wOld * p2(int(eOld));
    float dd = dot(stp, stp);
    if (dd < 1.0e-8) {
      float frac = clamp(log(1.0e-8 / wu.x) / log(dd / wu.x), 0.0, 1.0);
      nu = (float(it) + (frac == frac ? frac : 0.0)) * 5.0;
      status = 1.0;
      break;
    }
    wu.x = dd;
#elif FORMULA == 6
    // Magnet: q = (z^2 + c - 1) / (2z + c - 2), z' = q^2
    vec2 D = 2.0 * Z + Cf - vec2(2.0, 0.0);
    vec2 Q = orbit(uUOff + m).xy;
    vec2 Nn = cmul(Q, D);
    vec2 dn = 2.0 * cmul(Z, w) + s * csqr(w) + cs;
    vec2 dd = 2.0 * w + cs;
    vec2 dq = cdiv(cmul(dn, D) - cmul(Nn, dd), cmul(D, D + dd * s));
    w = 2.0 * cmul(Q, dq) + s * csqr(dq);
#else
    // tetration: d' = Z' (exp(d L + z dL) - 1)
    vec2 dlw = uEc < -20 ? dlm * p2(uEc - e) : dl * p2(clamp(-e, -126, 126));
    vec2 aw = cmulhl(uLref, w) + cmul(z, dlw);
    vec2 a = aw * s;
    if (a.x > 60.0) {
      nu = float(it + 1) * 3.0 + 3.0;
      status = 1.0;
      break;
    }
    vec2 g = dot(a, a) < 1.0e-4 ? ONE + 0.5 * a + csqr(a) / 6.0 : cdiv(cexp(a) - ONE, a);
    w = cmul(zval(orbit(mn)), cmul(aw, g));
#endif
#if FORMULA == 5
    // wu holds a step size here, not a scaled quantity: keep it out of the rescaling
    vec2 keep = wu;
    renorm(w, wu, e);
    wu = keep;
#else
    renorm(w, wu, e);
#endif
    m = mn;
    it++;
#endif
  }

  o0 = status != 0.0 ? vec4(float(it), 0.0, 0.0, 0.0) : vec4(w, float(e), float(m));
  o1 = vec4(status == 1.0 ? nu : float(it), status, wu);
}`;

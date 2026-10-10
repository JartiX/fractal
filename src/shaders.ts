// GLSL ES 3.00 sources.

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
export const ITERATE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uS0;
uniform sampler2D uS1;
uniform sampler2D uOrbit;
uniform sampler2D uBlaA;
uniform sampler2D uBlaE;
uniform int uInit;
uniform int uRefLen;
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
uniform float uEps2; // squared BLA tolerance: no BLA can apply once |d|^2 >= uEps2 * |Z|^2

layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;

const float R2 = 1.0e6;
const float LLR = 4.3169906; // log2(log2(R2))
const int ZERO_E = -10000000;
const int LMIN = 2;

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
vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }

// log2(|z|^2) that survives |z|^2 overflowing float (high powers overshoot the bailout hugely)
float l2norm(vec2 z) {
  float m = max(abs(z.x), abs(z.y));
  vec2 q = z / m;
  return 2.0 * log2(m) + log2(dot(q, q));
}

vec4 orbit(int m) { return texelFetch(uOrbit, ivec2(m & 2047, m >> 11), 0); }

int ctz(int x) {
  return int(floor(log2(float(x & (-x))) + 0.5));
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec2 w;
  int e, m, it;
  if (uInit == 1) {
    w = vec2(0.0);
    e = uEc;
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
  }

  vec2 uv = (gl_FragCoord.xy + uJitter) / uRes * 2.0 - 1.0;
  vec2 cm = uDm + uv * vec2(uAspect, 1.0) * uSm;
  int top = LMIN + uBlaLevels - 1;
  float status = 0.0;
  float nu = 0.0;

  vec2 c0 = cm * p2(uEc);
  float invLogPow = 1.0 / log2(float(uPow));
  bool bla = uBlaLevels > 0;

  for (int i = 0; i < uSteps; i++) {
    vec4 Z = orbit(m);

    // Tight loop for the common case late in an orbit: d is an ordinary float
    // and too large for any BLA, so iterate plainly without exponent upkeep.
    if (Z.w == 0.0 && e > -20 && e < 20) {
      w *= p2(e);
      e = 0;
      for (; i < uSteps; i++) {
        vec2 z = Z.xy + w;
        float zz = dot(z, z);
        float ww = dot(w, w);
        if (zz > R2) {
          nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(z))) * invLogPow);
          status = 1.0;
          break;
        }
        if (it >= uMaxIter) {
          status = 2.0;
          break;
        }
        if (zz < ww || m == uRefLen - 1 || ww < 1.0e-18 || (bla && ww < uEps2 * dot(Z.xy, Z.xy))) break;
        w = cmul(stepFactor(Z.xy, w), w) + c0;
        m++;
        it++;
        Z = orbit(m);
        if (Z.w != 0.0) break;
      }
      if (status != 0.0 || i >= uSteps) break;
    }

    // escape test, and rebase onto the start of the reference when the pixel
    // comes closer to the origin than to the reference orbit
    if (Z.w == 0.0) {
      vec2 d = w * p2(e);
      vec2 z = Z.xy + d;
      float zz = dot(z, z);
      if (zz > R2) {
        nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(z))) * invLogPow);
        status = 1.0;
        break;
      }
      if (zz < dot(d, d) || m == uRefLen - 1) {
        w = z;
        e = 0;
        renorm(w, e);
        m = 0;
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
      if (zz > R2) {
        nu = max(0.0, float(it) + 1.0 + (LLR - log2(l2norm(zf))) * invLogPow);
        status = 1.0;
        break;
      }
      if (m != 0 && (dot(zm, zm) < dot(dm, dm) || m == uRefLen - 1)) {
        w = zm;
        e = ez;
        renorm(w, e);
        m = 0;
        Z = vec4(0.0, 0.0, float(ZERO_E), 1.0);
      }
    }

    if (it >= uMaxIter) {
      status = 2.0;
      break;
    }

    bool stepped = false;
    float ww = dot(w, w);
    bool tryBla = bla && m > 0;
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
          vec4 AB = texelFetch(uBlaA, tc, 0);
          int ea = int(E.x) + e;
          int eb = int(E.y) + uEc;
          int en = max(ea, eb);
          w = cmul(AB.xy, w) * p2(ea - en) + cmul(AB.zw, cm) * p2(eb - en);
          e = en;
          renorm(w, e);
          m += 1 << k;
          it += 1 << k;
          stepped = true;
          break;
        }
      }
    }

    if (!stepped) {
      // d' = (Z+d)^p - Z^p + dc
      if (Z.w == 0.0 && uEc - e <= 40) {
        w = cmul(stepFactor(Z.xy, w * p2(e)), w) + cm * p2(uEc - e);
      } else {
        // Z or d is out of float range: sum of C(p,k) Z^(p-k) d^k with the larger
        // of the two factored out, so every intermediate value stays near 1
        float mw = max(abs(w.x), abs(w.y));
        if (mw == 0.0) {
          w = cm;
          e = uEc;
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
          int en = max(et, uEc);
          w = T * p2(et - en) + cm * p2(uEc - en);
          e = en;
        }
      }
      renorm(w, e);
      m++;
      it++;
    }
  }

  o0 = vec4(w, float(e), float(m));
  o1 = vec4(status == 1.0 ? nu : float(it), status, 0.0, 0.0);
}`;

// Collapses a finished (or abandoned) job into one float per pixel:
//   v >= 0   smooth iteration count of an escaped pixel
//   v == -1  inside the set
//   v <= -2  not finished: -(iterations reached so far) - 2
// A pixel the job did not finish keeps what the previous picture knew about
// that spot, so stopping a job early never punches black holes in the image.
export const RESOLVE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
uniform sampler2D uS1;
uniform sampler2DArray uOld;
uniform int uOldOn;
uniform int uOldInside; // the old picture's "inside" is still trustworthy
uniform int uGuessInside; // pixels had stopped finishing: what is left is most likely inside
uniform int uCapped; // job stopped short of the wanted iterations: its "inside" is unknown
uniform vec4 uOldX; // ratio, offset.xy, layer aspect
uniform vec2 uOldSize;
uniform vec2 uRes;
uniform float uAspect;
out vec4 o;
void main() {
  vec4 s = texelFetch(uS1, ivec2(gl_FragCoord.xy), 0);
  float v;
  if (s.y == 1.0) v = s.x;
  else if (s.y == 2.0 && uCapped == 0) v = -1.0;
  else {
    float est = s.x;
    v = uGuessInside == 1 ? -1.0 : -(est + 2.0);
    if (uOldOn == 1) {
      vec2 p = gl_FragCoord.xy / uRes * 2.0 - 1.0;
      vec2 q = vec2(p.x * uAspect, p.y) * uOldX.x + uOldX.yz;
      vec2 t = (vec2(q.x / uOldX.w, q.y) * 0.5 + 0.5) * uOldSize - 0.5;
      if (t.x > -1.0 && t.y > -1.0 && t.x < uOldSize.x && t.y < uOldSize.y) {
        ivec2 c = clamp(ivec2(floor(t + 0.5)), ivec2(0), ivec2(uOldSize) - 1);
        float old = texelFetch(uOld, ivec3(c, 0), 0).r;
        if (old >= est) v = old; // escaped later than we have got: trust it
        else if (old < 0.0 && old > -1.5) { if (uOldInside == 1) v = -1.0; } // known to be inside
        else if (old < -1.5 && uGuessInside == 0) v = -(max(est, -old - 2.0) + 2.0);
      }
    }
  }
  o = vec4(v, 0.0, 0.0, 1.0);
}`;

// Passes only where a pixel is still running; used with an occlusion query.
export const COUNT = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uS1;
out vec4 o;
void main() {
  if (texelFetch(uS1, ivec2(gl_FragCoord.xy), 0).y != 0.0) discard;
  o = vec4(1.0);
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

// Final image. Layers are reprojected into the current view:
//   acc  - the last finished image (several jittered samples once the view rests)
//   prev - the image before it, cross-faded out so a new frame never pops in
//   B    - the job in progress, shown only at rest; unfinished pixels fall back to acc
export const DISPLAY = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;

uniform sampler2DArray uAcc;
uniform int uAccCount;
uniform vec2 uAccSize;
uniform vec4 uAccX; // ratio, offset.xy, layer aspect
uniform sampler2DArray uPrev;
uniform int uPrevOn;
uniform vec2 uPrevSize;
uniform vec4 uPrevX;
uniform float uFade; // 0 = previous image, 1 = current
uniform sampler2D uB;
uniform int uBOn;
uniform int uBCapped;
uniform vec2 uBSize;
uniform vec4 uBX;
uniform sampler2D uPal;
uniform vec2 uRes;
uniform float uAspect;
uniform float uPhase;
uniform float uDensity;
out vec4 o;

const vec3 INSIDE = vec3(0.0);

vec3 shade(float nu) {
  if (nu < -1.5) nu = -nu - 2.0; // unfinished: colour of the iteration reached so far
  if (nu < 0.0) return INSIDE;
  return textureLod(uPal, vec2(pow(nu, 0.7) * uDensity + uPhase, 0.5), 0.0).rgb;
}

vec2 toTex(vec2 p, vec4 x, vec2 size) {
  vec2 q = vec2(p.x * uAspect, p.y) * x.x + x.yz;
  return (vec2(q.x / x.w, q.y) * 0.5 + 0.5) * size - 0.5;
}

bool inside(vec2 t, vec2 size) {
  return t.x > -1.0 && t.y > -1.0 && t.x < size.x && t.y < size.y;
}

vec4 layerColor(sampler2DArray tex, int count, vec2 size, vec4 x, vec2 p) {
  if (count == 0) return vec4(0.0);
  vec2 t = toTex(p, x, size);
  if (!inside(t, size)) return vec4(0.0);
  ivec2 hi = ivec2(size) - 1;
  if (count == 1) {
    vec2 b = floor(t);
    vec2 f = t - b;
    ivec2 c = ivec2(b);
    vec3 c00 = shade(texelFetch(tex, ivec3(clamp(c, ivec2(0), hi), 0), 0).r);
    vec3 c10 = shade(texelFetch(tex, ivec3(clamp(c + ivec2(1, 0), ivec2(0), hi), 0), 0).r);
    vec3 c01 = shade(texelFetch(tex, ivec3(clamp(c + ivec2(0, 1), ivec2(0), hi), 0), 0).r);
    vec3 c11 = shade(texelFetch(tex, ivec3(clamp(c + ivec2(1, 1), ivec2(0), hi), 0), 0).r);
    return vec4(mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y), 1.0);
  }
  ivec2 c = clamp(ivec2(floor(t + 0.5)), ivec2(0), hi);
  vec3 sum = vec3(0.0);
  for (int i = 0; i < count; i++) sum += shade(texelFetch(tex, ivec3(c, i), 0).r);
  return vec4(sum / float(count), 1.0);
}

vec3 bTap(ivec2 c, ivec2 hi, vec4 fallback) {
  vec2 s = texelFetch(uB, clamp(c, ivec2(0), hi), 0).xy;
  if (s.y == 1.0) return shade(s.x);
  if (s.y == 2.0 && uBCapped == 0) return INSIDE;
  return fallback.a > 0.5 ? fallback.rgb : shade(s.x);
}

void main() {
  vec2 p = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  vec4 a = layerColor(uAcc, uAccCount, uAccSize, uAccX, p);
  if (uPrevOn == 1) {
    vec4 pr = layerColor(uPrev, 1, uPrevSize, uPrevX, p);
    if (pr.a > 0.5) a = a.a > 0.5 ? vec4(mix(pr.rgb, a.rgb, uFade), 1.0) : pr;
  }
  vec3 col = a.rgb;
  if (uBOn == 1) {
    vec2 t = toTex(p, uBX, uBSize);
    if (inside(t, uBSize)) {
      ivec2 hi = ivec2(uBSize) - 1;
      vec2 b = floor(t);
      vec2 f = t - b;
      ivec2 c = ivec2(b);
      vec3 c00 = bTap(c, hi, a);
      vec3 c10 = bTap(c + ivec2(1, 0), hi, a);
      vec3 c01 = bTap(c + ivec2(0, 1), hi, a);
      vec3 c11 = bTap(c + ivec2(1, 1), hi, a);
      col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    }
  }
  // palette is stored linear; encode for display and dither away banding
  col = pow(col, vec3(1.0 / 2.2));
  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  o = vec4(col + (n - 0.5) / 255.0, 1.0);
}`;

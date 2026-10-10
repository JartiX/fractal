import { VERT, ITERATE, RESOLVE, COUNT, PROBE, DISPLAY } from './shaders';
import { View, Point, cloneView, sameView, delta, offsetIn } from './view';
import { PAL_SIZE } from './palettes';
import type { RefResult, BlaData } from './refWorker';
import { TEXW, EPS_L2 } from './refWorker';

const PROBE_W = 64;
const PROBE_H = 36;

export interface RefGpu extends Point {
  id: number;
  M: number;
  escaped: boolean;
  maxIter: number;
  orbitTex: WebGLTexture;
  blaA: WebGLTexture | null;
  blaE: WebGLTexture | null;
  blaOff: Int32Array;
  blaCnt: Int32Array;
  blaLevels: number;
  lcmax: number;
  lrMax: number;
  power: number;
}

interface Job {
  id: number;
  view: View;
  w: number;
  h: number;
  aa: boolean;
  motion: boolean;
  jitter: [number, number];
  refId: number;
  N: number;
  capped: boolean; // the reference is too short for the iterations this view wants
  ec: number;
  sm: number;
  dm: [number, number];
  useBla: boolean;
  front: number;
  passes: number;
  steps: number;
  K: number;
  noneActive: boolean;
  good: boolean;
  fracs: number[];
  t0: number;
}

export interface ProbeResult {
  view: View;
  aspect: number;
  w: number;
  h: number;
  data: Float32Array; // PROBE_W * PROBE_H * 4: (nu | iter, status, -, -)
  active: number; // fraction of pixels still running
  capped: boolean;
}

interface Target {
  fbo: WebGLFramebuffer;
  s0: WebGLTexture;
  s1: WebGLTexture;
}

type Uniforms = Record<string, WebGLUniformLocation | null>;
interface Program {
  p: WebGLProgram;
  u: Uniforms;
}

function halton(i: number, b: number): number {
  let f = 1;
  let r = 0;
  while (i > 0) {
    f /= b;
    r += f * (i % b);
    i = Math.floor(i / b);
  }
  return r;
}

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  W = 0;
  H = 0;
  ref: RefGpu | null = null;
  onProbe: ((r: ProbeResult) => void) | null = null;

  // tuning / state exposed for the HUD
  q = 1; // resolution factor used while the view is moving
  maxSamples = 8;
  targetMs = 8;
  phase = 0;
  density = 0.03;
  noBla = false; // diagnostics
  biasMotion = 1;
  biasStatic = -3;
  lastJob = { passes: 0, steps: 0, ms: 0, w: 0, complete: false, fracs: [] as number[] };
  stats = { steps: 0, passes: 0, samples: 0, scale: 1, busy: false, bla: false, gpuMs: 0 };

  private prog!: { iterate: Program; resolve: Program; count: Program; probe: Program; display: Program };
  private vao!: WebGLVertexArrayObject;
  private targets: Target[] = [];
  private accTexs: WebGLTexture[] = []; // current picture and the previous one
  private accCur = 0;
  private prev: { view: View; w: number; h: number } | null = null;
  private fadeStart = 0;
  fadeMs = 90;
  private accFbo!: WebGLFramebuffer;
  private acc = { view: null as View | null, w: 0, h: 0, count: 0, complete: false, N: 0 };
  private palTex!: WebGLTexture;
  private dummyTex!: WebGLTexture;
  private probeFbo!: WebGLFramebuffer;
  private probeTex!: WebGLTexture;
  private pbo!: WebGLBuffer;
  private probePending: { sync: WebGLSync; job: Job } | null = null;
  private probeBuf = new Float32Array(PROBE_W * PROBE_H * 4);
  private timerExt: any = null;
  private timers: { q: WebGLQuery; job: Job; first: boolean; K: number }[] = [];
  private counts: { q: WebGLQuery; job: Job }[] = [];
  private job: Job | null = null;
  private jobSeq = 0;
  private c0 = 2e-7; // estimated ms per (step * pixel) with every pixel running

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 недоступен');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('Нет поддержки рендера в float-текстуры');
    this.gl = gl;
    this.timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    this.init();
  }

  // --- setup ---------------------------------------------------------------

  private compile(fs: string, names: string[]): Program {
    const gl = this.gl;
    const mk = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader error');
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, mk(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link error');
    const u: Uniforms = {};
    for (const n of names) u[n] = gl.getUniformLocation(p, n);
    return { p, u };
  }

  private init(): void {
    const gl = this.gl;
    this.prog = {
      iterate: this.compile(ITERATE, [
        'uS0', 'uS1', 'uOrbit', 'uBlaA', 'uBlaE', 'uInit', 'uRefLen', 'uMaxIter', 'uSteps', 'uEc',
        'uBlaLevels', 'uBlaOff', 'uBlaCnt', 'uRes', 'uJitter', 'uDm', 'uAspect', 'uSm', 'uBlaBias', 'uEps2', 'uPow',
      ]),
      resolve: this.compile(RESOLVE, ['uS1', 'uOld', 'uOldOn', 'uOldInside', 'uGuessInside', 'uCapped', 'uOldX', 'uOldSize', 'uRes', 'uAspect']),
      count: this.compile(COUNT, ['uS1']),
      probe: this.compile(PROBE, ['uS1', 'uSize', 'uGrid']),
      display: this.compile(DISPLAY, [
        'uAcc', 'uAccCount', 'uAccSize', 'uAccX', 'uPrev', 'uPrevOn', 'uPrevSize', 'uPrevX', 'uFade', 'uB', 'uBOn', 'uBCapped', 'uBSize', 'uBX', 'uPal', 'uRes',
        'uAspect', 'uPhase', 'uDensity',
      ]),
    };
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    this.palTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.dummyTex = this.floatTex(1, 1, new Float32Array(4));

    this.probeTex = this.floatTex(PROBE_W, PROBE_H, null);
    this.probeFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.probeTex, 0);
    this.pbo = gl.createBuffer()!;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, this.probeBuf.byteLength, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.accFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private floatTex(w: number, h: number, data: Float32Array | null): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  setPalette(pixels: Uint8Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, PAL_SIZE, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  }

  resize(w: number, h: number): void {
    if (w === this.W && h === this.H) return;
    const gl = this.gl;
    this.W = w;
    this.H = h;
    this.canvas.width = w;
    this.canvas.height = h;
    for (const t of this.targets) {
      gl.deleteFramebuffer(t.fbo);
      gl.deleteTexture(t.s0);
      gl.deleteTexture(t.s1);
    }
    this.targets = [];
    for (let i = 0; i < 2; i++) {
      const s0 = this.floatTex(w, h, null);
      const s1 = this.floatTex(w, h, null);
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s0, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, s1, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      this.targets.push({ fbo, s0, s1 });
    }
    this.maxSamples = w * h > 5_000_000 ? 4 : 8;
    for (const t of this.accTexs) gl.deleteTexture(t);
    this.accTexs = [];
    for (let i = 0; i < 2; i++) {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R32F, w, h, this.maxSamples);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.accTexs.push(t);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.acc.count = 0;
    this.prev = null;
    this.job = null;
  }

  /** Forget everything drawn and the reference orbit (the formula changed). */
  reset(): void {
    const gl = this.gl;
    const old = this.ref;
    if (old) {
      gl.deleteTexture(old.orbitTex);
      if (old.blaA) gl.deleteTexture(old.blaA);
      if (old.blaE) gl.deleteTexture(old.blaE);
    }
    this.ref = null;
    this.job = null;
    this.acc.count = 0;
    this.prev = null;
  }

  // --- reference data --------------------------------------------------------

  setRef(r: RefResult): void {
    const gl = this.gl;
    const old = this.ref;
    const orbitTex = this.floatTex(TEXW, r.orbitRows, r.orbit);
    this.ref = {
      id: r.id, cx: r.cx, cy: r.cy, prec: r.prec, M: r.M, escaped: r.escaped, maxIter: r.maxIter,
      orbitTex, blaA: null, blaE: null, blaOff: r.blaOff, blaCnt: r.blaCnt, blaLevels: 0, lcmax: r.lcmax, lrMax: r.lrMax, power: r.power,
    };
    this.setBla(r);
    if (old) {
      gl.deleteTexture(old.orbitTex);
      if (old.blaA) gl.deleteTexture(old.blaA);
      if (old.blaE) gl.deleteTexture(old.blaE);
    }
  }

  setBla(b: BlaData): void {
    const gl = this.gl;
    const ref = this.ref;
    if (!ref) return;
    if (ref.blaA) gl.deleteTexture(ref.blaA);
    if (ref.blaE) gl.deleteTexture(ref.blaE);
    ref.blaA = b.blaA ? this.floatTex(TEXW, b.blaRows, b.blaA) : null;
    ref.blaE = b.blaE ? this.floatTex(TEXW, b.blaRows, b.blaE) : null;
    ref.blaOff = b.blaOff;
    ref.blaCnt = b.blaCnt;
    ref.blaLevels = b.blaA ? b.blaLevels : 0;
    ref.lcmax = b.lcmax;
    ref.lrMax = b.lrMax;
    // a running job keeps sampling the textures it started with, so restart it
    if (this.job) {
      if (!this.job.aa && this.job.passes > 0) this.promote(this.job, false);
      this.job = null;
    }
  }

  /** log2 of the largest |dc| a job at this view would use, and the offset to the reference. */
  refOffset(v: View): { dm: [number, number]; lc: number } | null {
    const ref = this.ref;
    if (!ref) return null;
    const ec = Math.floor(v.ls);
    const sm = Math.pow(2, v.ls - ec);
    const dm = delta(v, ref, ec);
    const aspect = this.W / this.H;
    const lc = ec + Math.log2(Math.hypot(dm[0], dm[1]) + Math.hypot(aspect, 1) * sm);
    return { dm, lc };
  }

  // --- frame -----------------------------------------------------------------

  /** True when the picture for `now` is finished and nothing is left to refine. */
  idle(now: View): boolean {
    const a = this.acc;
    return !this.job && a.count >= this.maxSamples && !!a.view && sameView(a.view, now);
  }

  frame(now: View, isStatic: boolean, N: number): void {
    this.pollQueries();
    this.pollProbe();
    const ref = this.ref;
    let job = this.job;

    if (job && ref) {
      const same = sameView(job.view, now);
      const complete = job.noneActive || job.steps >= job.N + 2;
      if (job.refId !== ref.id) {
        if (!job.aa && job.passes > 0) this.promote(job, false);
        job = null;
      } else if (job.aa) {
        if (!same) job = null;
        else if (complete) {
          this.resolve(job, this.acc.count);
          this.acc.count++;
          job = null;
        }
      } else if (complete) {
        this.tuneScale(job, false);
        this.promote(job, true);
        job = null;
      } else if (!same) {
        const [dx, dy] = offsetIn(job.view, now);
        const hard = Math.abs(now.ls - job.view.ls) > 0.6 || Math.hypot(dx, dy) > 0.5;
        if (job.good || hard) {
          this.tuneScale(job, !job.good);
          this.promote(job, false);
          job = null;
        }
      }
    }
    if (!job && ref) job = this.startJob(now, isStatic, N, ref);
    if (job && ref) this.pass(job, ref);
    this.job = job;

    this.stats.busy = !!job;
    this.stats.samples = this.acc.count;
    this.stats.scale = job && !job.aa ? job.w / this.W : this.acc.count ? this.acc.w / this.W : 1;
    this.stats.bla = !!job && job.useBla;
    this.display(now);
  }

  private tuneScale(job: Job, forced: boolean): void {
    if (!job.motion) return;
    // aim for a new picture roughly every 60-110 ms while moving
    const ms = performance.now() - job.t0;
    if (forced || ms > 110) this.q = Math.max(0.3, this.q * 0.88);
    else if (ms < 60) this.q = Math.min(1, this.q * 1.1);
  }

  private startJob(now: View, isStatic: boolean, N: number, ref: RefGpu): Job | null {
    const off = this.refOffset(now)!;
    const dmLen = Math.hypot(off.dm[0], off.dm[1]);
    if (!Number.isFinite(dmLen) || dmLen > 3000 || ref.prec < -now.ls + 40) return null;

    const acc = this.acc;
    const full = acc.w === this.W && acc.h === this.H;
    const exact = acc.count > 0 && acc.complete && full && sameView(acc.view!, now);
    let aa = false;
    let q = isStatic ? 1 : this.q;
    let jitter: [number, number] = [0, 0];
    if (exact) {
      if (!isStatic || acc.count >= this.maxSamples) return null;
      aa = true;
      q = 1;
      jitter = [halton(acc.count, 2) - 0.5, halton(acc.count, 3) - 0.5];
    }
    const w = Math.max(32, Math.round(this.W * q));
    const h = Math.max(32, Math.round(this.H * q));
    const ec = Math.floor(now.ls);
    const maxIter = ref.escaped ? N : Math.min(N, ref.M - 1);
    return {
      id: ++this.jobSeq,
      view: cloneView(now),
      w,
      h,
      aa,
      motion: !isStatic,
      jitter,
      refId: ref.id,
      N: maxIter,
      capped: maxIter < N,
      ec,
      sm: Math.pow(2, now.ls - ec),
      dm: off.dm,
      useBla: !this.noBla && ref.blaLevels > 0 && off.lc <= ref.lcmax,
      front: 0,
      passes: 0,
      steps: 0,
      K: Math.max(8, Math.min(4096, Math.round(this.targetMs / (this.c0 * w * h)))),
      noneActive: false,
      good: false,
      fracs: [],
      t0: performance.now(),
    };
  }

  private bindTex(unit: number, target: number, tex: WebGLTexture | null): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
  }

  private pass(job: Job, ref: RefGpu): void {
    const gl = this.gl;
    const src = this.targets[job.front];
    const dst = this.targets[1 - job.front];
    const { p, u } = this.prog.iterate;
    const K = Math.max(1, Math.round(job.K));

    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
    gl.viewport(0, 0, job.w, job.h);
    gl.useProgram(p);
    this.bindTex(0, gl.TEXTURE_2D, src.s0);
    this.bindTex(1, gl.TEXTURE_2D, src.s1);
    this.bindTex(2, gl.TEXTURE_2D, ref.orbitTex);
    this.bindTex(3, gl.TEXTURE_2D, ref.blaA || this.dummyTex);
    this.bindTex(4, gl.TEXTURE_2D, ref.blaE || this.dummyTex);
    gl.uniform1i(u.uS0, 0);
    gl.uniform1i(u.uS1, 1);
    gl.uniform1i(u.uOrbit, 2);
    gl.uniform1i(u.uBlaA, 3);
    gl.uniform1i(u.uBlaE, 4);
    gl.uniform1i(u.uInit, job.passes === 0 ? 1 : 0);
    gl.uniform1i(u.uRefLen, ref.M);
    gl.uniform1i(u.uMaxIter, job.N);
    gl.uniform1i(u.uSteps, K);
    gl.uniform1i(u.uEc, job.ec);
    gl.uniform1i(u.uBlaLevels, job.useBla ? ref.blaLevels : 0);
    gl.uniform1iv(u.uBlaOff, ref.blaOff);
    gl.uniform1iv(u.uBlaCnt, ref.blaCnt);
    gl.uniform2f(u.uRes, job.w, job.h);
    gl.uniform2f(u.uJitter, job.jitter[0], job.jitter[1]);
    gl.uniform2f(u.uDm, job.dm[0], job.dm[1]);
    gl.uniform1f(u.uAspect, this.W / this.H);
    gl.uniform1f(u.uSm, job.sm);
    gl.uniform1i(u.uPow, ref.power);
    // looser linearisation while moving (faster), tighter for the resting picture
    const bias = job.motion ? this.biasMotion : this.biasStatic;
    gl.uniform1f(u.uBlaBias, bias);
    gl.uniform1f(u.uEps2, Math.pow(2, 2 * (EPS_L2 + bias)));

    const ext = this.timerExt;
    let tq: WebGLQuery | null = null;
    if (ext && this.timers.length < 4) {
      tq = gl.createQuery()!;
      gl.beginQuery(ext.TIME_ELAPSED_EXT, tq);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (tq) {
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      this.timers.push({ q: tq, job, first: job.passes === 0, K });
    }
    job.front = 1 - job.front;
    job.passes++;
    job.steps += K;
    this.stats.steps = K;
    this.stats.passes = job.passes;
    if (!ext) job.K = Math.min(32768, job.K * 1.25);

    // is anything still running?
    if (this.counts.length < 4) {
      const cq = gl.createQuery()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.colorMask(false, false, false, false);
      gl.useProgram(this.prog.count.p);
      this.bindTex(0, gl.TEXTURE_2D, dst.s1);
      gl.uniform1i(this.prog.count.u.uS1, 0);
      gl.beginQuery(gl.ANY_SAMPLES_PASSED, cq);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.endQuery(gl.ANY_SAMPLES_PASSED);
      gl.colorMask(true, true, true, true);
      this.counts.push({ q: cq, job });
    }

    if (!this.probePending && !job.aa) {
      const pr = this.prog.probe;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeFbo);
      gl.viewport(0, 0, PROBE_W, PROBE_H);
      gl.useProgram(pr.p);
      this.bindTex(0, gl.TEXTURE_2D, dst.s1);
      gl.uniform1i(pr.u.uS1, 0);
      gl.uniform2f(pr.u.uSize, job.w, job.h);
      gl.uniform2f(pr.u.uGrid, PROBE_W, PROBE_H);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.readPixels(0, 0, PROBE_W, PROBE_H, gl.RGBA, gl.FLOAT, 0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
      gl.flush();
      this.probePending = { sync, job };
    }
  }

  /** Report an over-long frame when no GPU timer is available. */
  frameTime(dt: number, nominal: number): void {
    if (this.timerExt) return;
    this.c0 *= dt > nominal * 1.6 ? 1.3 : 0.985;
  }

  private pollQueries(): void {
    const gl = this.gl;
    const ext = this.timerExt;
    while (this.timers.length) {
      const t = this.timers[0];
      if (!gl.getQueryParameter(t.q, gl.QUERY_RESULT_AVAILABLE)) break;
      this.timers.shift();
      const ms = gl.getQueryParameter(t.q, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(t.q);
      if (gl.getParameter(ext.GPU_DISJOINT_EXT) || !(ms > 0)) continue;
      this.stats.gpuMs = ms;
      if (t.first) {
        const c = ms / (t.K * t.job.w * t.job.h);
        this.c0 = this.c0 * 0.5 + c * 0.5;
      }
      if (t.job === this.job) {
        const want = (t.K * this.targetMs) / Math.max(ms, 0.05);
        t.job.K = Math.max(8, Math.min(32768, Math.min(want, t.job.K * 2)));
      }
    }
    while (this.counts.length) {
      const c = this.counts[0];
      if (!gl.getQueryParameter(c.q, gl.QUERY_RESULT_AVAILABLE)) break;
      this.counts.shift();
      if (!gl.getQueryParameter(c.q, gl.QUERY_RESULT)) c.job.noneActive = true;
      gl.deleteQuery(c.q);
    }
  }

  private pollProbe(): void {
    const gl = this.gl;
    const pp = this.probePending;
    if (!pp) return;
    if (gl.getSyncParameter(pp.sync, gl.SYNC_STATUS) !== gl.SIGNALED) return;
    this.probePending = null;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.probeBuf);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.deleteSync(pp.sync);
    const d = this.probeBuf;
    const job = pp.job;
    let active = 0;
    for (let i = 1; i < d.length; i += 4) if (d[i] === 0) active++;
    const f = active / (PROBE_W * PROBE_H);
    job.fracs.push(f);
    job.fracs.push(job.steps);
    // "good enough" to hand over while moving: nearly everything is done, or a
    // quarter more work has stopped finishing pixels (the rest is deep or inside)
    if (f < 0.02) job.good = true;
    else if (f < 0.6) {
      for (let i = job.fracs.length - 4; i >= 0; i -= 2) {
        if (job.fracs[i + 1] <= job.steps * 0.8) {
          if (job.fracs[i] - f < 0.01) job.good = true;
          break;
        }
      }
    }
    if (this.onProbe) this.onProbe({ view: job.view, aspect: this.W / this.H, w: job.w, h: job.h, data: d, active: f, capped: job.capped });
  }

  /** Writes the job into `layer` of the current picture; `inherit` fills unfinished pixels from the other one. */
  private resolve(job: Job, layer: number, inherit = false): void {
    const gl = this.gl;
    const { p, u } = this.prog.resolve;
    const acc = this.acc;
    const ox = inherit && acc.count && acc.view ? this.layerXform(job.view, acc.view) : null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.accTexs[this.accCur], 0, layer);
    gl.viewport(0, 0, job.w, job.h);
    gl.useProgram(p);
    this.bindTex(0, gl.TEXTURE_2D, this.targets[job.front].s1);
    this.bindTex(1, gl.TEXTURE_2D_ARRAY, this.accTexs[1 - this.accCur]);
    gl.uniform1i(u.uS1, 0);
    gl.uniform1i(u.uOld, 1);
    gl.uniform1i(u.uOldOn, ox ? 1 : 0);
    gl.uniform1i(u.uOldInside, acc.N >= job.N * 0.9 ? 1 : 0);
    gl.uniform1i(u.uCapped, job.capped ? 1 : 0);
    gl.uniform1i(u.uGuessInside, job.good && !job.capped ? 1 : 0);
    if (ox) {
      gl.uniform4f(u.uOldX, ox[0], ox[1], ox[2], ox[3]);
      gl.uniform2f(u.uOldSize, acc.w, acc.h);
    }
    gl.uniform2f(u.uRes, job.w, job.h);
    gl.uniform1f(u.uAspect, this.W / this.H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private promote(job: Job, complete: boolean): void {
    this.lastJob = { passes: job.passes, steps: job.steps, ms: performance.now() - job.t0, w: job.w, complete, fracs: job.fracs };
    this.prev = this.acc.count && this.acc.view ? { view: this.acc.view, w: this.acc.w, h: this.acc.h } : null;
    this.fadeStart = performance.now();
    this.accCur = 1 - this.accCur;
    this.resolve(job, 0, !complete || job.capped);
    this.acc = { view: job.view, w: job.w, h: job.h, count: 1, complete: complete && !job.capped, N: job.N };
  }

  private layerXform(now: View, lv: View): [number, number, number, number] | null {
    const ratio = Math.pow(2, now.ls - lv.ls);
    const [dx, dy] = offsetIn(lv, now);
    if (!(ratio < 256 && ratio > 1 / 256) || !(Math.hypot(dx, dy) < 256)) return null;
    return [ratio, dx, dy, this.W / this.H];
  }

  private display(now: View): void {
    const gl = this.gl;
    const { p, u } = this.prog.display;
    const job = this.job;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(p);

    const acc = this.acc;
    const ax = acc.count && acc.view ? this.layerXform(now, acc.view) : null;
    this.bindTex(0, gl.TEXTURE_2D_ARRAY, this.accTexs[this.accCur]);
    gl.uniform1i(u.uAcc, 0);
    gl.uniform1i(u.uAccCount, ax ? acc.count : 0);
    gl.uniform2f(u.uAccSize, acc.w || 1, acc.h || 1);
    if (ax) gl.uniform4f(u.uAccX, ax[0], ax[1], ax[2], ax[3]);

    const fade = Math.min(1, (performance.now() - this.fadeStart) / this.fadeMs);
    const px = this.prev && fade < 1 ? this.layerXform(now, this.prev.view) : null;
    this.bindTex(3, gl.TEXTURE_2D_ARRAY, this.accTexs[1 - this.accCur]);
    gl.uniform1i(u.uPrev, 3);
    gl.uniform1i(u.uPrevOn, px ? 1 : 0);
    gl.uniform1f(u.uFade, fade * fade * (3 - 2 * fade));
    if (px) {
      gl.uniform2f(u.uPrevSize, this.prev!.w, this.prev!.h);
      gl.uniform4f(u.uPrevX, px[0], px[1], px[2], px[3]);
    }

    // Filling a moving picture in pixel by pixel reads as boiling noise, so the job
    // in progress is only shown at rest (or when there is nothing else to show).
    const showB = job && !job.aa && job.passes > 0 && (!job.motion || !ax);
    const bx = showB ? this.layerXform(now, job!.view) : null;
    this.bindTex(1, gl.TEXTURE_2D, bx ? this.targets[job!.front].s1 : this.dummyTex);
    gl.uniform1i(u.uB, 1);
    gl.uniform1i(u.uBOn, bx ? 1 : 0);
    gl.uniform1i(u.uBCapped, job && job.capped ? 1 : 0);
    if (bx) {
      gl.uniform2f(u.uBSize, job!.w, job!.h);
      gl.uniform4f(u.uBX, bx[0], bx[1], bx[2], bx[3]);
    }
    this.bindTex(2, gl.TEXTURE_2D, this.palTex);
    gl.uniform1i(u.uPal, 2);
    gl.uniform2f(u.uRes, this.W, this.H);
    gl.uniform1f(u.uAspect, this.W / this.H);
    gl.uniform1f(u.uPhase, this.phase);
    gl.uniform1f(u.uDensity, this.density);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Synchronous read of the finished image at job pixel coordinates (tests only). */
  readNu(points: [number, number][]): number[] {
    const gl = this.gl;
    const out: number[] = [];
    const buf = new Float32Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.accTexs[this.accCur], 0, 0);
    for (const [x, y] of points) {
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, buf);
      out.push(buf[0]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}

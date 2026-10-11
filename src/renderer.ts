import { VERT, iterateSrc, customSrc, RESOLVE, COUNT, EVENTS, PROBE, DISPLAY, DOWN, UP, FILL, MAXL } from './shaders';
import { customInfo } from './custom';
import { View, Point, cloneView, sameView, delta, offsetIn, fixedToDbl, pan, zoomAt } from './view';
import { Spec, specKey, paramLc } from './formulas';
import { PAL_SIZE } from './palettes';
import type { RefResult, BlaData } from './refWorker';
import { TEXW, EPS_L2 } from './refWorker';

// The iteration limit grows by this factor for as long as patches of pixels are
// still finishing in the upper half of the iterations done; once that half has
// gone quiet, what is left is taken to be inside.
const GROW = 1.5;
const EV_W = 256; // room for the 16x16-block event map of a 4096x2304 job
const EV_H = 144;
// This many late finishers in one block make a patch still coming in. A thin rim
// along a boundary (a couple of pixels wide) stays below it: chasing the last
// pixel of a rim costs ten times the iterations and shows nothing new.
const EV_PATCH = 48;
const PROBE_W = 64;
const PROBE_H = 36;

export interface RefGpu extends Point {
  id: number;
  M: number;
  lenA: number;
  base: number;
  cap: number;
  spec: Spec;
  tag: number;
  perA: number;
  perB: number;
  interior: boolean;
  uOff: number;
  cref: [number, number, number, number]; // reference parameter split into float pairs (hi, lo)
  lref: [number, number, number, number]; // its logarithm, likewise
  kp: [number, number]; // (1 - C)/2 as a mantissa pair
  ke: number; // and its exponent
  cre: number; // the view-plane point as plain numbers
  cim: number;
  escaped: boolean;
  maxIter: number;
  orbitTex: WebGLTexture;
  blaA: WebGLTexture | null;
  blaB: WebGLTexture | null;
  blaE: WebGLTexture | null;
  blaOff: Int32Array;
  blaCnt: Int32Array;
  blaLevels: number;
  lcmax: number;
  lrMax: number;
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
  ref: RefGpu; // the orbit it runs on, which need not be the newest
  N: number; // current iteration limit; grows while pixels keep finishing
  N0: number; // limit the job started with
  base: number; // limit the picture was expected to need, whatever a job it replaces had got to
  Nprev: number; // limit before the latest extension
  epoch: number; // bumped whenever N changes, to tell stale query results apart
  stage: number; // 0 iterating, 1 asking whether to extend, 2 waiting for a longer reference orbit
  final: boolean; // nothing more to learn: ready to become the finished picture
  wantN: number; // limit it would like but the reference is too short for
  lastActive: number; // fraction of pixels unfinished at the last probe
  lastBusy: number; // highest limit at which patches were still finishing
  lateN: number; // how far pixels were seen to finish in earnest (what the next picture and the extra samples need)
  ec: number;
  sm: number;
  dm: [number, number];
  lc: number; // log2 of the largest |dc| in the job
  useBla: boolean;
  front: number;
  passes: number;
  steps: number;
  hsteps: number; // steps taken since N last changed
  K: number;
  noBudget: boolean; // every unfinished pixel has used up the current limit
  tag: number;
  layer: Layer | null; // where it is drawn (for extra samples: the picture they refine)
  dirty: boolean; // has moved on since its layer was last written
  src: Layer | null; // an earlier picture whose smooth parts it takes over instead of iterating
  exact: boolean; // ...which lies on this picture's own pixel grid: all it knows is taken over
  share: number; // share of its pixels that it has to work out itself (0: not known beforehand)
  carry: number; // an extra sample: 1 if it starts from the same sample of the picture before
  strict: boolean; // made to the resting tolerance
  busyAt: number; // when patches were last seen finishing
  roundAt: number; // when the limit was last raised
  quiet: boolean; // gone quiet: what is unfinished is drawn as inside, though the job looks a little further
  doubts: number; // rounds added only because a patch stayed unknown
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
}

/** One picture of the stack: a rendering of `view`, finished or not, in one slice of the stack texture. */
interface Layer {
  slot: number;
  view: View;
  w: number;
  h: number;
  N: number; // iteration limit it was taken to
  Naa: number; // limit that serves extra samples of it
  cost: number; // ms it took to make, or would take to make anew
  tag: number;
  complete: boolean; // a finished picture: nothing in it is still unknown
  known: number; // share of its pixels that are known, roughly
  done: boolean; // its job ran its course (whatever is unknown in it will stay so)
  strict: boolean; // made to the resting tolerance throughout: fit to be copied into a resting picture
  born: number; // when it first came on screen (it fades in)
  seq: number;
}

interface Half {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
  w: number;
  h: number;
}

/** The extra samples of the picture at rest: whose they are, how many, and where the first ones can be taken from. */
interface Extra {
  layer: Layer | null;
  count: number;
  /** samples of the picture this one was dragged from (in the other texture), shifted by whole pixels */
  from: { count: number; kx: number; ky: number } | null;
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
  /** The camera is driven by the program (a dive), not by hand: it can be paced by the pictures. */
  auto = false;
  /**
   * How far the picture lags the camera: octaves by which the best layer that covers
   * the whole screen is coarser than the screen (1: none covers it). A dive is slowed by it.
   */
  behind = 0;
  maxSamples = 8;
  targetMs = 8;
  phase = 0;
  density = 0.03;
  noBla = false; // diagnostics
  noGrid = false; // diagnostics: never take a dragged view's pixels over from the picture before
  biasMotion = 1;
  biasStatic = -3;
  lastJob = { passes: 0, steps: 0, ms: 0, w: 0, complete: false, fracs: [] as number[] };
  stats = { steps: 0, passes: 0, samples: 0, scale: 1, busy: false, bla: false, gpuMs: 0, restMs: 0, layers: 0, N: 0, deepening: false, held: 0 };

  private prog!: { iterate: Program; resolve: Program; count: Program; events: Program; probe: Program; display: Program; down: Program; up: Program; fill: Program };
  private iterPrograms = new Map<string, Program>();
  spec: Spec | null = null;
  private vao!: WebGLVertexArrayObject;
  private targets: Target[] = [];
  private stackTex!: WebGLTexture; // the layers: one float per pixel each (see RESOLVE)
  // extra jittered samples of the finished picture at rest; the other one holds those of
  // the picture it was dragged from, which lie on the same sub-pixel positions
  private aaTexs: WebGLTexture[] = [];
  private aaCur = 0;
  private layers: Layer[] = [];
  private layerSeq = 0;
  private showTag = -1; // which formula / parameter the layers on screen belong to
  private holdUntil = 0; // after a change of fractal: leave the last frame on screen until then, or until there is a picture
  private exact: Layer | null = null; // the finished full-size picture of the view on screen
  private aa: Extra = { layer: null, count: 0, from: null };
  // steps a pass of an extra sample can take within the frame's time, as the sample before showed
  // (apart for samples that start from an earlier picture's): each one is much like the last
  private aaK: [number, number] = [0, 0];
  private aaKFor: Layer | null = null;
  private vel = { ls: 0, x: 0, y: 0 }; // how the view is moving, per millisecond
  private lastView: View | null = null;
  private lastT = 0;
  private leadMs = 150; // how long a picture takes while moving: how far ahead the next one aims
  fadeMs = 90;
  private accFbo!: WebGLFramebuffer;
  private aaTarget = 8; // samples wanted for the picture on screen
  private compTex: WebGLTexture | null = null; // the composed picture: colour and coverage (see FILL)
  private compFbo!: WebGLFramebuffer;
  private pull: Half[] = []; // it halved again and again
  private push: Half[] = []; // and each of those with its gaps filled from the coarser ones
  private lx = new Float32Array(MAXL * 4);
  private lsz = new Float32Array(MAXL * 4);
  private palTex!: WebGLTexture;
  private dummyTex!: WebGLTexture;
  private probeFbo!: WebGLFramebuffer;
  private probeTex!: WebGLTexture;
  private pbo!: WebGLBuffer;
  private probePending: { sync: WebGLSync; job: Job } | null = null;
  private passSyncs: { sync: WebGLSync; at: number }[] = []; // passes still on the GPU, as far as is known
  private probeBuf = new Float32Array(PROBE_W * PROBE_H * 4);
  private timerExt: any = null;
  private timers: { q: WebGLQuery; job: Job; first: boolean; K: number }[] = [];
  private restQ: WebGLQuery | null = null; // times everything in a frame but the iterations
  /** Length of a display frame in ms (0: unknown). */
  frameMs = 0;
  private counts: { q: WebGLQuery; job: Job; epoch: number }[] = [];
  private evq: { sync: WebGLSync; job: Job; epoch: number; cw: number; ch: number } | null = null;
  private evFbo!: WebGLFramebuffer;
  private evTex!: WebGLTexture;
  private evPbo!: WebGLBuffer;
  private evBuf = new Float32Array(EV_W * EV_H * 4);
  maxN = 6_000_000;
  /** Largest texture side the GPU takes: bounds how long an orbit can be stored. */
  maxTex = 4096;
  /** Iterations the current view would like the reference orbit to cover. */
  needIter = 0;
  private need = 0; // limit the last finished picture actually needed
  private needLs = 0;
  private resumeN = 0; // limit a job had reached when its reference was replaced
  private resumeBase = 0; // and what that job had expected to need
  private job: Job | null = null;
  private oldRefs: RefGpu[] = []; // replaced orbits a job is still running on
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
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
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
      iterate: null as unknown as Program, // chosen by setFormula
      resolve: this.compile(RESOLVE, ['uS1', 'uFinal']),
      count: this.compile(COUNT, ['uS1', 'uN']),
      events: this.compile(EVENTS, ['uS0', 'uS1', 'uT', 'uSize']),
      probe: this.compile(PROBE, ['uS1', 'uSize', 'uGrid']),
      display: this.compile(DISPLAY, [
        'uStack', 'uCount', 'uLX', 'uLS', 'uAA', 'uAACount', 'uDirect', 'uPal', 'uRes', 'uAspect', 'uPhase', 'uDensity',
      ]),
      down: this.compile(DOWN, ['uSrc', 'uSize']),
      up: this.compile(UP, ['uOwn', 'uUp', 'uSize', 'uTop']),
      fill: this.compile(FILL, ['uC', 'uUp', 'uRes']),
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
    this.evTex = this.floatTex(EV_W, EV_H, null);
    this.evFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.evFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.evTex, 0);
    this.evPbo = gl.createBuffer()!;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.evPbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, this.evBuf.byteLength, gl.STREAM_READ);
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
    const half = (cw: number, ch: number): Half => {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, cw, ch);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fbo, w: cw, h: ch };
    };
    for (const x of [...this.pull, ...this.push]) {
      gl.deleteTexture(x.tex);
      gl.deleteFramebuffer(x.fbo);
    }
    if (this.compTex) {
      gl.deleteTexture(this.compTex);
      gl.deleteFramebuffer(this.compFbo);
    }
    const comp = half(w, h);
    this.compTex = comp.tex;
    this.compFbo = comp.fbo;
    this.pull = [];
    this.push = [];
    for (let cw = w, ch = h; Math.max(cw, ch) > 6; ) {
      cw = Math.max(1, Math.ceil(cw / 2));
      ch = Math.max(1, Math.ceil(ch / 2));
      this.pull.push(half(cw, ch));
      this.push.push(half(cw, ch));
    }
    const arr = (old: WebGLTexture | undefined, n: number): WebGLTexture => {
      if (old) gl.deleteTexture(old);
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R32F, w, h, n);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return t;
    };
    this.stackTex = arr(this.stackTex, MAXL);
    this.aaTexs = [arr(this.aaTexs[0], this.maxSamples - 1), arr(this.aaTexs[1], this.maxSamples - 1)];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.layers = [];
    this.aa = { layer: null, count: 0, from: null };
    this.exact = null;
    this.job = null;
  }

  /** Video memory the frame-sized buffers take, in bytes: what is left of a budget is for orbits. */
  baseBytes(): number {
    // iteration state (two pairs of RGBA32F), layers and extra samples (R32F), the
    // composed picture with its halvings (RGBA16F), the canvas itself
    return this.W * this.H * (64 + 4 * MAXL + 8 * (this.maxSamples - 1) + 8 * (1 + 2 / 3) + 8);
  }

  /** Name of the graphics card as the driver reports it ('' if hidden). */
  gpuName(): string {
    const e = this.gl.getExtension('WEBGL_debug_renderer_info');
    return e ? String(this.gl.getParameter(e.UNMASKED_RENDERER_WEBGL)) : '';
  }

  /** Switch to another fractal; its shader is compiled on first use. */
  setFormula(spec: Spec): void {
    const key = specKey(spec);
    let prog = this.iterPrograms.get(key);
    if (!prog) {
      prog =
        spec.kind === 'custom'
          ? this.compile(customSrc(spec.custom, customInfo(spec.custom)), [
              'uS0', 'uS1', 'uOrbit', 'uInit', 'uRefLen', 'uEndA', 'uBase', 'uMaxIter', 'uSteps', 'uEc', 'uPow', 'uUOff',
              'uRes', 'uJitter', 'uDm', 'uAspect', 'uSm', 'uCref', 'uLref', 'uPerA', 'uPerB', 'uAccTex', 'uSkipIn', 'uAccMax', 'uAccLayer', 'uSrcX', 'uSrcSize', 'uAaFrom', 'uAaLayer', 'uAaShift',
            ])
          : this.compile(iterateSrc(spec), [
              'uS0', 'uS1', 'uOrbit', 'uBlaA', 'uBlaE', 'uInit', 'uRefLen', 'uEndA', 'uBase', 'uMaxIter', 'uSteps', 'uEc',
              'uBlaLevels', 'uBlaOff', 'uBlaCnt', 'uRes', 'uJitter', 'uDm', 'uAspect', 'uSm', 'uBlaBias', 'uEps2', 'uPow',
              'uPerA', 'uPerB', 'uAccTex', 'uSkipIn', 'uAccMax', 'uLP', 'uPH', 'uKp', 'uKe', 'uBlaB', 'uUOff', 'uAccLayer', 'uSrcX', 'uSrcSize', 'uAaFrom', 'uAaLayer', 'uAaShift',
            ]);
      this.iterPrograms.set(key, prog);
    }
    this.prog.iterate = prog;
    this.spec = spec;
    this.reset();
  }

  /** Forget everything drawn and the reference orbit (the formula changed). */
  reset(): void {
    if (this.ref) this.oldRefs.push(this.ref);
    this.job = null;
    this.sweep();
    this.ref = null;
    this.job = null;
    this.layers = [];
    this.aa = { layer: null, count: 0, from: null };
    this.exact = null;
    this.showTag = -1;
    this.holdUntil = performance.now() + 600;
    this.need = 0;
    this.resumeN = 0;
    this.needIter = 0;
  }

  // --- reference data --------------------------------------------------------

  /**
   * Install a reference orbit. `same` says it is a longer orbit of the point the
   * current one belongs to: a picture in progress then simply carries on with it.
   * Returns false, changing nothing, if the card has no memory for it.
   */
  setRef(r: RefResult, same = false): boolean {
    const gl = this.gl;
    const old = this.ref;
    const job = this.job;
    while (gl.getError() !== gl.NO_ERROR); // only errors from here on are of interest
    const orbitTex = this.floatTex(TEXW, r.orbitRows, r.orbit);
    const split = (v: number): [number, number] => {
      const hi = Math.fround(v);
      return [hi, Math.fround(v - hi)];
    };
    const cre = split(fixedToDbl(r.vx, r.prec, 0));
    const cim = split(fixedToDbl(r.vy, r.prec, 0));
    // (1 - C)/2 with its own exponent: it may be tiny when the view sits at c = 1
    const kx = (1n << BigInt(r.prec)) - r.vx;
    const ky = -r.vy;
    const kbits = Math.max((kx < 0n ? -kx : kx).toString(2).length, (ky < 0n ? -ky : ky).toString(2).length);
    const ksh = Math.max(0, kbits - 53);
    const kden = Math.pow(2, kbits - ksh);
    const kp: [number, number] = [Number(kx >> BigInt(ksh)) / kden, Number(ky >> BigInt(ksh)) / kden];
    const ke = kbits - r.prec - 1;
    const mod = Math.hypot(cre[0] + cre[1], cim[0] + cim[1]);
    const lre = split(mod > 0 ? Math.log(mod) : 0);
    const lim = split(mod > 0 ? Math.atan2(cim[0] + cim[1], cre[0] + cre[1]) : 0);
    this.ref = {
      id: r.id, cx: r.vx, cy: r.vy, prec: r.prec, kp, ke, cre: cre[0] + cre[1], cim: cim[0] + cim[1], M: r.M, lenA: r.lenA, base: r.base, cap: r.cap, spec: r.spec, tag: r.tag,
      perA: r.perA, perB: r.perB, interior: r.interior, uOff: r.uOff, cref: [cre[0], cim[0], cre[1], cim[1]], lref: [lre[0], lim[0], lre[1], lim[1]],
      escaped: r.escaped, maxIter: r.maxIter,
      orbitTex, blaA: null, blaB: null, blaE: null, blaOff: r.blaOff, blaCnt: r.blaCnt, blaLevels: 0, lcmax: r.lcmax, lrMax: r.lrMax,
    };
    this.setBla(r);
    const nr = this.ref;
    for (let err = gl.getError(); err !== gl.NO_ERROR; err = gl.getError()) {
      if (err !== gl.OUT_OF_MEMORY) continue;
      gl.deleteTexture(nr.orbitTex);
      if (nr.blaA) gl.deleteTexture(nr.blaA);
      if (nr.blaB) gl.deleteTexture(nr.blaB);
      if (nr.blaE) gl.deleteTexture(nr.blaE);
      this.ref = old;
      return false;
    }
    if (job && old && job.ref === old) {
      if (same && !job.aa && old.base === 0 && nr.base === 0) {
        // a longer orbit of the same point: the picture simply carries on with it
        // (not so with two orbits in one table: the second one has moved along)
        job.ref = nr;
        job.refId = nr.id;
        job.useBla = !this.noBla && r.spec.kind !== 'custom' && nr.blaLevels > 0 && job.lc <= nr.lcmax;
        if (job.stage === 2 && nr.cap > job.N) this.extendTo(job, Math.min(job.wantN, nr.cap));
      } else if (job.aa || nr.tag !== job.tag || job.stage === 2 || (!job.motion && job.lastActive > 0.15)) {
        this.drop(job);
        this.job = null;
      }
      // otherwise it finishes on the orbit it started with: that one is as true as
      // the new, and starting over would throw the work away
    }
    if (old) this.oldRefs.push(old);
    this.sweep();
    return true;
  }

  /** Free the orbits no job runs on any more. */
  private sweep(): void {
    const gl = this.gl;
    const used = this.job ? this.job.ref : null;
    this.oldRefs = this.oldRefs.filter((o) => {
      if (o === used) return true;
      gl.deleteTexture(o.orbitTex);
      if (o.blaA) gl.deleteTexture(o.blaA);
      if (o.blaB) gl.deleteTexture(o.blaB);
      if (o.blaE) gl.deleteTexture(o.blaE);
      return false;
    });
  }

  /** Install a BLA table. A rebuilt table for the same orbit (`same`) is as good as the old one mid-picture. */
  setBla(b: BlaData, same = false): void {
    const gl = this.gl;
    const ref = this.ref;
    if (!ref) return;
    if (ref.blaA) gl.deleteTexture(ref.blaA);
    if (ref.blaB) gl.deleteTexture(ref.blaB);
    if (ref.blaE) gl.deleteTexture(ref.blaE);
    ref.blaA = b.blaA ? this.floatTex(TEXW, b.blaRows, b.blaA) : null;
    ref.blaB = b.blaB ? this.floatTex(TEXW, b.blaRows, b.blaB) : null;
    ref.blaE = b.blaE ? this.floatTex(TEXW, b.blaRows, b.blaE) : null;
    ref.blaOff = b.blaOff;
    ref.blaCnt = b.blaCnt;
    ref.blaLevels = b.blaA ? b.blaLevels : 0;
    ref.lcmax = b.lcmax;
    ref.lrMax = b.lrMax;
    const job = this.job;
    if (job && same && job.ref === ref) job.useBla = !this.noBla && ref.spec.kind !== 'custom' && ref.blaLevels > 0 && job.lc <= ref.lcmax;
  }

  private extendTo(job: Job, to: number): void {
    job.Nprev = job.N;
    job.N = Math.floor(to);
    job.epoch++;
    job.hsteps = 0;
    job.noBudget = false;
    job.roundAt = performance.now();
    // every pixel that was waiting at the old limit runs again: start the steps per pass low
    job.K = Math.max(8, Math.min(job.K, this.targetMs / (this.c0 * job.w * job.h * Math.max(0.02, job.lastActive))));
    job.fracs = [];
    job.stage = 0;
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
    return { dm, lc: paramLc(ref.spec, lc, ref.cre, ref.cim) };
  }

  // --- frame -----------------------------------------------------------------

  /** True once something drawn with the current reference is on screen. */
  settled(): boolean {
    return !!this.ref && this.showTag === this.ref.tag && this.layers.length > 0;
  }

  /** The finished full-size picture of exactly this view, if the stack holds one. */
  private exactLayer(now: View): Layer | null {
    let best: Layer | null = null;
    for (const L of this.layers) {
      if (L.complete && L.w === this.W && L.h === this.H && L.tag === this.showTag && (!best || L.seq > best.seq) && sameView(L.view, now)) best = L;
    }
    return best;
  }

  /** True when the picture for `now` is finished and nothing is left to refine. */
  idle(now: View): boolean {
    const e = this.exactLayer(now);
    return !this.job && !!e && !!this.ref && e.tag === this.ref.tag && this.aa.layer === e && 1 + this.aa.count >= this.aaTarget;
  }

  /** Iteration limit of the finished picture on screen (0 if there is none). */
  finalN(): number {
    return this.exact ? this.exact.N : 0;
  }

  frame(now: View, isStatic: boolean, N: number): void {
    const t = performance.now();
    this.pollQueries();
    this.pollProbe();
    this.track(now, isStatic, t);
    const rest = isStatic;
    const ref = this.ref;
    let job = this.job;

    if (job) {
      {
        // the current limit is used up: every loop turn advances a pixel by at least one iteration
        if (job.stage === 0 && job.passes > 0 && (job.noBudget || job.hsteps >= job.N - job.Nprev + 2)) this.limitReached(job);
        const same = sameView(job.view, now);
        if (job.final) {
          this.finish(job, same, t);
          job = null;
        } else if (job.aa) {
          if (!same) job = null;
        } else if (job.stage === 2 && job.ref !== ref) {
          // it waits for a longer orbit, but only the newest one is ever made longer
          this.drop(job);
          job = null;
        } else if (!same && this.stale(job, now, rest)) {
          this.tuneScale(job, true);
          this.drop(job);
          job = null;
        } else if (this.enough(job)) {
          this.early(job, t);
          job = null;
        }
      }
    }
    if (!job && ref) job = this.startJob(now, rest, N, ref);
    // A pass that runs over its frame is still on the GPU when the next frame comes round,
    // and the next pass queues up behind it; a queue of them holds everything up for a
    // good many frames. So none is added while two are known to be in hand. (Word that a
    // pass is through arrives a frame late: one is "in hand" even when all is on time.)
    {
      const gl = this.gl;
      this.passSyncs = this.passSyncs.filter((s) => {
        if (gl.getSyncParameter(s.sync, gl.SYNC_STATUS) !== gl.SIGNALED && t - s.at < 250) return true;
        gl.deleteSync(s.sync);
        return false;
      });
      if (job && job.stage === 0) {
        if (this.passSyncs.length < 2) {
          this.pass(job, job.ref);
          this.passSyncs.push({ sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!, at: t });
        } else this.stats.held++;
      }
    }
    if (!job && this.oldRefs.length) this.sweep();
    const ext = this.timerExt;
    const gl = this.gl;
    let rq: WebGLQuery | null = null;
    if (ext && !this.restQ) {
      rq = gl.createQuery()!;
      gl.beginQuery(ext.TIME_ELAPSED_EXT, rq);
    }
    // the picture in progress is on screen as it stands: what it knows is drawn, the rest shows through
    if (job && !job.aa && job.dirty) this.snapshot(job, job.quiet && !job.motion, t);
    this.job = job;
    this.exact = this.exactLayer(now);

    this.stats.busy = !!job;
    this.stats.samples = this.exact ? (this.aa.layer === this.exact ? 1 + this.aa.count : 1) : 0;
    this.stats.scale = job && !job.aa ? job.w / this.W : this.exact ? 1 : this.q;
    this.stats.bla = !!job && job.useBla;
    this.stats.N = job ? job.N : this.exact ? this.exact.N : 0;
    this.stats.deepening = !!job && !job.aa && job.N > job.N0;
    this.display(now, t);
    if (rq) {
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      this.restQ = rq;
    }
  }

  /** Follow the view's motion, so that pictures can be aimed at where it is heading. */
  private track(now: View, isStatic: boolean, t: number): void {
    const lv = this.lastView;
    const dt = t - this.lastT;
    const v = this.vel;
    let ok = false;
    // (a view that did not move since the last frame has no velocity: the next picture is aimed right at it)
    if (lv && !isStatic && !sameView(lv, now) && dt > 0 && dt < 250) {
      const [dx, dy] = offsetIn(lv, now);
      const dl = now.ls - lv.ls;
      if (Math.abs(dl) < 1.5 && Math.hypot(dx, dy) < 1.5) {
        const k = 1 - Math.exp(-dt / 90);
        v.ls += (dl / dt - v.ls) * k;
        v.x += (dx / dt - v.x) * k;
        v.y += (dy / dt - v.y) * k;
        ok = true;
      }
    }
    if (!ok) v.ls = v.x = v.y = 0;
    this.lastView = cloneView(now);
    this.lastT = t;
  }

  /** Octaves or screen heights per second the view is moving at. */
  private speed(): number {
    return (Math.abs(this.vel.ls) + 0.7 * Math.hypot(this.vel.x, this.vel.y)) * 1000;
  }

  /** The view a moving camera will show by the time a picture started now is ready. */
  private ahead(now: View): View {
    const v = this.vel;
    const out = cloneView(now);
    const lead = this.leadMs * (v.ls < 0 ? 1.2 : 1.6);
    const dl = Math.max(-0.5, Math.min(0.5, v.ls * lead)); // half an octave ahead still covers half the screen
    // the centre drifts in screen units, and those shrink or grow with the view
    const f = Math.abs(dl) < 1e-3 ? 1 : (Math.pow(2, dl) - 1) / (dl * Math.LN2);
    let sx = v.x * lead * f;
    let sy = v.y * lead * f;
    const s = Math.hypot(sx, sy);
    if (s > 0.35) {
      sx *= 0.35 / s;
      sy *= 0.35 / s;
    }
    if (s > 1e-4) pan(out, sx, sy);
    if (Math.abs(dl) > 1e-4) zoomAt(out, 0, 0, dl);
    return out;
  }

  /** A picture in progress is for a view the screen has left: is it still worth finishing? */
  private stale(job: Job, now: View, isStatic: boolean): boolean {
    if (isStatic) return true; // the view has come to rest somewhere else
    const d = job.view.ls - now.ls; // octaves the job is wider than the screen
    const r = Math.pow(2, -d);
    const [dx, dy] = offsetIn(job.view, now);
    const a = this.W / this.H;
    // how much of the smaller of the two, job and screen, lies within the other
    const span = (c: number, half: number, lim: number) => Math.max(0, Math.min(c + half, lim) - Math.max(c - half, -lim));
    const cover = (span(dx, r * a, a) * span(dy, r, 1)) / (4 * a * Math.min(1, r * r));
    // (a resting picture caught by motion gives way at once, unless it is nearly done)
    return job.motion ? d > 0.9 || d < -1.3 || cover < 0.5 : job.lastActive > 0.15 || Math.abs(d) > 0.3 || cover < 0.8;
  }

  /**
   * While the view moves a picture need not wait for its stragglers. The pixels that
   * take longest are few - a rim, a speck - yet a picture waiting for them is late for
   * all the others, and what lies under them shows through anyway.
   */
  private enough(job: Job): boolean {
    if (!job.motion || job.aa || job.stage !== 0 || job.passes < 3) return false;
    const f = job.lastActive;
    if (f <= 0.015) return true;
    const h = job.fracs;
    // Or the rest has stopped coming in: a quarter more work finished next to nothing.
    // That only means something once the bulk has come in - at the start no pixel
    // has had the iterations to finish, and nothing coming in is just that.
    if (h.length >= 6 && f <= 0.5 * h[0]) {
      for (let i = h.length - 4; i >= 0; i -= 2) {
        if (h[i + 1] <= job.steps * 0.8) return h[i] - f < 0.004;
      }
    }
    return false;
  }

  /** End a moving picture that has enough (see above): its layer stays as it is. */
  private early(job: Job, t: number): void {
    this.tuneScale(job, false);
    const ms = t - job.t0;
    this.leadMs = Math.max(40, Math.min(900, this.leadMs + (ms - this.leadMs) * 0.4));
    this.resumeN = Math.max(this.resumeN, job.N);
    this.resumeBase = Math.max(job.lastBusy, job.base);
    if (job.layer) job.layer.done = true;
    this.lastJob = { passes: job.passes, steps: job.steps, ms, w: job.w, complete: false, fracs: job.fracs };
  }

  /** A job has ended with nothing more to learn. */
  private finish(job: Job, same: boolean, t: number): void {
    if (job.aa) {
      const a = this.aa;
      if (same && a.layer && a.layer === job.layer && a.count < this.maxSamples - 1) {
        // at the picture's own limit an unfinished sample is inside like an unfinished pixel
        // of the picture; at a lower one it says nothing (the display leaves it out)
        this.resolve(job, this.aaTexs[this.aaCur], a.count, job.N >= a.layer.N);
        a.count++;
      }
      return;
    }
    this.tuneScale(job, false);
    // only a picture made at rest may call what it has not finished inside
    this.snapshot(job, !job.motion, t);
    this.reveal(job);
    const L = job.layer!;
    L.complete = !job.motion;
    const A = this.aa.layer;
    if (L.complete && job.exact && A && A !== L && this.aa.count > 0 && A.tag === L.tag && A.w === L.w && A.h === L.h && A.view.ls === L.view.ls) {
      // Dragged from the picture the extra samples were made for, and on its pixel grid:
      // the same samples serve this one, shifted by whole pixels - only the strip that
      // came into view has to be made.
      const [dx, dy] = offsetIn(A.view, L.view);
      const kx = Math.round((dx * L.h) / 2);
      const ky = Math.round((dy * L.h) / 2);
      if (Math.abs((dx * L.h) / 2 - kx) < 1e-6 && Math.abs((dy * L.h) / 2 - ky) < 1e-6 && Math.abs(kx) < L.w && Math.abs(ky) < L.h) {
        this.aa = { layer: L, count: 0, from: { count: this.aa.count, kx, ky } };
        this.aaCur = 1 - this.aaCur;
      }
    }
    L.known = L.complete ? 1 : Math.max(0.5, 1 - job.lastActive);
    L.done = true;
    // where the next picture starts: as far as pixels were seen to finish
    this.need = job.lateN > 0 ? job.lateN : job.N;
    this.needLs = job.view.ls;
    // extra samples only smooth edges: that limit serves them too, and a
    // picture that took long to finish gets fewer of them
    L.Naa = Math.max(1, Math.min(job.N, Math.ceil(this.need)));
    const ms = t - job.t0;
    // (a picture mostly taken over was quick to put together, yet a sample of it is made in full)
    L.cost = Math.max(L.cost, ms);
    this.aaTarget = L.cost > 5000 ? 2 : L.cost > 1500 ? Math.min(4, this.maxSamples) : this.maxSamples;
    if (job.motion) this.leadMs = Math.max(40, Math.min(900, this.leadMs + (ms - this.leadMs) * 0.4));
    this.lastJob = { passes: job.passes, steps: job.steps, ms, w: job.w, complete: true, fracs: job.fracs };
    if (L.complete) this.prune(L);
  }

  /**
   * The newest careful layer of the same scale and size as a picture of `view` would
   * be, and overlapping it: that picture can lie on its pixel grid. `aligned` says
   * `view` already does; `snapped` is the nearest view that does.
   */
  private grid(view: View, w: number, h: number, tag: number): { layer: Layer; aligned: boolean; snapped: View; dx: number; dy: number } | null {
    if (this.noGrid) return null;
    const a = this.W / this.H;
    let best: Layer | null = null;
    for (const L of this.layers) {
      if (!L.strict || L.tag !== tag || L.tag !== this.showTag || L.w !== w || L.h !== h || L.view.ls !== view.ls) continue;
      const [dx, dy] = offsetIn(L.view, view);
      if (Math.abs(dx) > 1.6 * a || Math.abs(dy) > 1.6) continue; // a fifth of it still in view, at least
      if (!best || L.seq > best.seq) best = L;
    }
    if (!best) return null;
    const [dx, dy] = offsetIn(best.view, view);
    const kx = Math.round((dx * h) / 2);
    const ky = Math.round((dy * h) / 2);
    const snapped = cloneView(best.view);
    if (kx || ky) pan(snapped, (kx * 2) / h, (ky * 2) / h);
    const ex = (kx * 2) / h - dx;
    const ey = (ky * 2) / h - dy;
    return { layer: best, aligned: Math.abs(ex) * h < 1e-6 && Math.abs(ey) * h < 1e-6, snapped, dx: ex, dy: ey };
  }

  /**
   * A view that has come to rest a fraction of a pixel off the grid of a careful
   * picture of the same scale: how far to move it (in half-heights) to sit on that
   * grid, so that the resting picture need only work out what is new. Null if there
   * is no such picture or the view sits on its grid already.
   */
  align(now: View): [number, number] | null {
    const g = this.grid(now, this.W, this.H, this.showTag);
    return g && !g.aligned ? [g.dx, g.dy] : null;
  }

  /** The layer a new picture of `view` (h pixels high) overlaps best, to take its smooth parts from. */
  private source(view: View, h: number, tag: number): Layer | null {
    const g = view.ls - Math.log2(h);
    const a = this.W / this.H;
    let best: Layer | null = null;
    for (const L of this.layers) {
      if (L.tag !== tag || L.tag !== this.showTag) continue;
      const d = L.view.ls - Math.log2(L.h) - g; // octaves coarser than the new picture
      if (d > 1.2 || d < -2) continue;
      const r = Math.pow(2, view.ls - L.view.ls);
      const [dx, dy] = offsetIn(L.view, view);
      if (Math.abs(dx) > a * (0.5 + r) || Math.abs(dy) > 0.5 + r) continue; // hardly overlaps
      if (!best || L.seq > best.seq) best = L;
    }
    return best;
  }

  /** Give a job up. What it has found stays on screen as a layer. */
  private drop(job: Job): void {
    if (job.aa) return;
    this.resumeN = Math.max(this.resumeN, job.N, job.wantN);
    this.resumeBase = Math.max(job.lastBusy, job.base);
    const L = job.layer;
    const hidden = !!L && L.tag !== this.showTag;
    if (L && (hidden ? L.known < 0.5 && this.layers.some((x) => x.tag === this.showTag) : L.known < 0.02 && this.layers.some((x) => x !== L && x.tag === L.tag))) {
      // it has learnt nothing (or, not yet on screen, too little to replace what is):
      // its layer would only take a place in the stack, or blank the picture
      this.layers = this.layers.filter((x) => x !== L);
      job.layer = null;
    }
    this.reveal(job);
    this.lastJob = { passes: job.passes, steps: job.steps, ms: performance.now() - job.t0, w: job.w, complete: false, fracs: job.fracs };
  }

  /** Put a job's layer on screen although it shows another formula or parameter than what is there. */
  private reveal(job: Job): void {
    const L = job.layer;
    if (!L || L.tag === this.showTag) return;
    this.showTag = L.tag;
    this.layers = this.layers.filter((x) => x.tag === L.tag);
    if (this.aa.layer && this.aa.layer.tag !== L.tag) this.aa = { layer: null, count: 0, from: null };
    L.born = 0; // nothing of the old picture is left to fade from
  }

  /** Every unfinished pixel has had `job.N` iterations. Finish, or find out whether more would change anything. */
  private limitReached(job: Job): void {
    if (job.aa) {
      job.final = true;
      return;
    }
    const gl = this.gl;
    // every pixel has had its iterations: this is a picture, whatever the last probe (a few frames old) says
    if (job.layer) job.layer.known = Math.max(job.layer.known, 0.5);
    this.reveal(job); // show what there is while the question is open
    if (this.evq) gl.deleteSync(this.evq.sync);
    const t = this.targets[job.front];
    const { p, u } = this.prog.events;
    const cw = Math.min(EV_W, Math.ceil(job.w / 16));
    const ch = Math.min(EV_H, Math.ceil(job.h / 16));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.evFbo);
    gl.viewport(0, 0, cw, ch);
    gl.useProgram(p);
    this.bindTex(0, gl.TEXTURE_2D, t.s0);
    this.bindTex(1, gl.TEXTURE_2D, t.s1);
    gl.uniform1i(u.uS0, 0);
    gl.uniform1i(u.uS1, 1);
    gl.uniform1f(u.uT, job.N / 2);
    gl.uniform2i(u.uSize, job.w, job.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.evPbo);
    gl.readPixels(0, 0, cw, ch, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
    gl.flush();
    this.evq = { sync, job, epoch: job.epoch, cw, ch };
    job.stage = 1;
  }

  /**
   * The answer to limitReached: how many pixels finished late and how bunched they
   * are (in blocks finishing together, and in any block); how many are still
   * unknown, and the most of them in one block.
   */
  private decide(job: Job, total: number, densest: number, dense: number, unknown: number, patch: number): void {
    const ref: RefGpu | null = job.ref;
    const now = performance.now();
    // Patches still coming in, or simply a lot of pixels: the tail is not over.
    // But a round has to be worth its time: the longer the last one took, the more
    // it must have brought. A boundary, and dust above all, sheds stray pixels for
    // ever, each batch dearer than the one before; chasing them shows nothing new.
    const sec = (now - job.roundAt) / 1000;
    let more = densest >= EV_PATCH * Math.max(1, sec / 1.5) || total >= job.w * job.h * 0.004 * Math.max(1, sec / 0.15);
    if (more) {
      job.lastBusy = job.N;
      job.busyAt = now;
    }
    // How far finishing was seen to go, whether or not it is worth going on for:
    // the extra samples and the next picture must reach that far. The check looks
    // at the upper half of the limit; a stretch below it that no check has looked
    // at (the first limit, or a limit raised by a leap) counts as busy.
    if (dense >= EV_PATCH || total >= job.w * job.h * 0.004) job.lateN = job.N;
    else if (job.Nprev < job.N / 2) job.lateN = Math.max(job.lateN, job.N / 2);
    // A whole patch neither escaped nor proven inside. For an analytic formula that
    // is suspect - there the inside proves itself (the contraction test) - so more
    // likely the limit falls short of a later wave of escapes: look a good deal
    // further before calling it inside. Folded and other formulas do have insides
    // that prove nothing (bounded chaos): there it is drawn as inside from the first
    // limit on, and checked a little further.
    // (All this is for a picture at rest; one made on the move has no time to doubt.)
    const analytic = !!ref && ref.spec.kind === 'power';
    if (!more && !job.motion && patch >= 192 && job.N < Math.max(job.lastBusy, job.base) * (analytic ? 6 : 4) && now - job.busyAt < (analytic ? 2500 : 1200)) {
      more = true;
      job.doubts++;
    }
    // ...and after looking twice in vain it is drawn as inside there too, while the look goes on
    if (patch >= 192 && (!analytic || job.doubts >= 2)) job.quiet = true;
    // Nothing at all has finished and nothing proves the view lies inside the set:
    // the limit may simply be below the first escape. Keep looking for a while.
    if (!more && job.lastActive >= 1 && ref && !ref.interior && job.N < job.base * 8) more = true;
    let next = Math.min(this.maxN, Math.ceil(job.N * GROW));
    // the reference point is one of the pixels: if it escapes at n, the limit must clear n
    // (an orbit may have grown under the job since it chose its limit)
    const clear = ref && ref.escaped && ref.base === 0 ? Math.min(this.maxN, Math.ceil(ref.lenA * 1.2)) : 0;
    if (job.N < clear) {
      more = true;
      next = Math.max(next, clear);
    }
    if (!more || next <= job.N || !ref) {
      job.final = true;
      return;
    }
    const to = Math.min(next, ref.cap);
    if (to > job.N) this.extendTo(job, to);
    else {
      // the reference orbit itself is not long enough: ask for a longer one and wait
      job.wantN = next;
      job.stage = 2;
      this.needIter = Math.max(this.needIter, next * GROW);
    }
  }

  private tuneScale(job: Job, forced: boolean): void {
    if (!job.motion) return;
    // A picture should be ready before the view has moved half an octave on: the
    // faster the motion, the coarser the pictures. (Shorter ones would each be
    // fresher but so much coarser that the screen ends up softer.)
    const ms = performance.now() - job.t0;
    const hi = Math.max(160, Math.min(900, 550 / Math.max(this.speed(), 0.01)));
    if (forced || ms > hi) this.q = Math.max(0.33, this.q * 0.85);
    else if (ms < hi * 0.5) this.q = Math.min(1, this.q * 1.12);
    if (forced && ms > this.leadMs) this.leadMs = Math.min(900, this.leadMs + (ms - this.leadMs) * 0.4);
  }

  private startJob(now: View, isStatic: boolean, N: number, ref: RefGpu): Job | null {
    const exact = this.exactLayer(now);
    let aa = false;
    // a dive keeps full resolution: it is the camera that gives way (see `behind`)
    let q = isStatic || this.auto ? 1 : this.q;
    let jitter: [number, number] = [0, 0];
    if (exact && exact.tag === ref.tag) {
      if (!isStatic) return null;
      if (this.aa.layer !== exact) this.aa = { layer: exact, count: 0, from: null };
      if (this.aaKFor !== exact) {
        this.aaKFor = exact;
        this.aaK = [0, 0];
      }
      const n = 1 + this.aa.count;
      if (n >= this.aaTarget) return null;
      aa = true;
      q = 1;
      jitter = [halton(n, 2) - 0.5, halton(n, 3) - 0.5];
    }
    // while moving, aim at where the view will be when the picture is ready
    let view = aa || isStatic ? cloneView(now) : this.ahead(now);
    // A view that was only dragged since a careful picture was made can sit on that
    // picture's pixel grid: everything it knows is then taken over as it is, and
    // only the strip that came into view is worked out.
    let src: Layer | null = null;
    let onGrid = false;
    let share = 1;
    if (!aa) {
      const g = this.grid(view, Math.max(32, Math.round(this.W * q)), Math.max(32, Math.round(this.H * q)), ref.tag);
      if (g && (g.aligned || !isStatic)) {
        if (!isStatic) view = g.snapped;
        src = g.layer;
        onGrid = true;
        // what is left to work out: the strip outside that picture, and what it does not know
        const [ox, oy] = offsetIn(src.view, view);
        const a = this.W / this.H;
        const over = Math.max(0, 1 - Math.abs(ox) / (2 * a)) * Math.max(0, 1 - Math.abs(oy) / 2);
        share = Math.max(0.05, 1 - over * (src.complete ? 1 : Math.min(1, src.known)));
      } else if (!isStatic) {
        src = this.source(view, Math.max(32, Math.round(this.H * q)), ref.tag);
        if (src) share = 0;
      }
    }
    const carry = aa && this.aa.from && this.aa.count < this.aa.from.count ? 1 : 0;
    const off = this.refOffset(view)!;
    const dmLen = Math.hypot(off.dm[0], off.dm[1]);
    if (!Number.isFinite(dmLen) || dmLen > 3000 || ref.prec < -view.ls + 40) return null;
    const w = Math.max(32, Math.round(this.W * q));
    const h = Math.max(32, Math.round(this.H * q));
    const ec = Math.floor(view.ls);
    // the reference point is one of the pixels: if it escapes at n, the limit must clear n
    const floor = Math.max(N, ref.escaped && ref.base === 0 ? Math.ceil(ref.lenA * 1.2) : 0);
    if (Math.abs(now.ls - this.needLs) > 4) this.need = 0;
    const want = aa ? exact!.Naa : Math.max(floor, this.need, this.resumeN);
    const base = !aa && this.resumeBase > 0 ? Math.min(this.resumeBase, want) : want;
    if (!aa) {
      this.resumeN = 0;
      this.resumeBase = 0;
      this.needIter = Math.min(this.maxN, want * GROW);
    }
    const maxIter = Math.max(1, Math.floor(Math.min(want, ref.cap, this.maxN)));
    return {
      id: ++this.jobSeq,
      view,
      w,
      h,
      aa,
      motion: !isStatic,
      jitter,
      refId: ref.id,
      ref,
      N: maxIter,
      N0: maxIter,
      base: Math.max(1, Math.min(base, maxIter)),
      Nprev: 0,
      epoch: 0,
      stage: 0,
      final: false,
      wantN: 0,
      lastActive: 1,
      lastBusy: 0,
      // what is copied from a picture on the same grid needed that picture's limit
      lateN: onGrid && src ? src.Naa : 0,
      ec,
      sm: Math.pow(2, view.ls - ec),
      dm: off.dm,
      lc: off.lc,
      useBla: !this.noBla && ref.spec.kind !== 'custom' && ref.blaLevels > 0 && off.lc <= ref.lcmax,
      front: 0,
      passes: 0,
      steps: 0,
      hsteps: 0,
      K: aa && this.aaK[carry] > 0 ? this.aaK[carry] : Math.max(8, Math.min(4096, Math.round(this.targetMs / (this.c0 * w * h * (share || 1))))),
      noBudget: false,
      tag: ref.tag,
      layer: aa ? exact : null,
      dirty: false,
      src,
      exact: onGrid,
      share: aa ? 0 : share,
      carry,
      strict: aa || isStatic || onGrid,
      busyAt: performance.now(),
      roundAt: performance.now(),
      quiet: false,
      doubts: 0,
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
    this.bindTex(6, gl.TEXTURE_2D, ref.blaB || this.dummyTex);
    gl.uniform1i(u.uBlaB, 6);
    gl.uniform1i(u.uS0, 0);
    gl.uniform1i(u.uS1, 1);
    gl.uniform1i(u.uOrbit, 2);
    gl.uniform1i(u.uBlaA, 3);
    gl.uniform1i(u.uBlaE, 4);
    gl.uniform1i(u.uInit, job.passes === 0 ? 1 : 0);
    gl.uniform1i(u.uRefLen, ref.M);
    gl.uniform1i(u.uEndA, ref.lenA - 1);
    gl.uniform1i(u.uBase, ref.base);
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
    gl.uniform1i(u.uPow, ref.spec.power);
    gl.uniform1f(u.uLP, Math.log2(ref.spec.power));
    gl.uniform2f(u.uKp, ref.kp[0], ref.kp[1]);
    gl.uniform1i(u.uKe, ref.ke);
    gl.uniform1f(u.uPH, (ref.spec.power - 1) / 2);
    gl.uniform1i(u.uUOff, ref.uOff);
    gl.uniform1i(u.uPerA, ref.perA);
    gl.uniform1i(u.uPerB, ref.perB);
    // extra samples skip what the finished picture shows to be well inside
    this.bindTex(5, gl.TEXTURE_2D_ARRAY, this.stackTex);
    gl.uniform1i(u.uAccTex, 5);
    const from = job.aa ? job.layer : job.passes === 0 ? job.src : null;
    gl.uniform1i(u.uAccLayer, from ? from.slot : 0);
    // an extra sample starts from the same sample of the picture before, where that lies on this grid
    const af = job.aa && this.aa.layer === job.layer ? this.aa.from : null;
    const ai = af && this.aa.count < af.count ? this.aa.count : -1;
    this.bindTex(7, gl.TEXTURE_2D_ARRAY, this.aaTexs[1 - this.aaCur]);
    gl.uniform1i(u.uAaFrom, 7);
    gl.uniform1i(u.uAaLayer, job.passes === 0 ? ai : -1);
    gl.uniform2i(u.uAaShift, af ? af.kx : 0, af ? af.ky : 0);
    gl.uniform1i(u.uSkipIn, !from ? 0 : job.aa ? 1 : 2);
    if (from && !job.aa) {
      const [dx, dy] = offsetIn(from.view, job.view);
      gl.uniform4f(u.uSrcX, Math.pow(2, job.view.ls - from.view.ls), dx, dy, job.exact ? 1 : 0);
      gl.uniform2f(u.uSrcSize, from.w, from.h);
    }
    gl.uniform2i(u.uAccMax, job.w - 1, job.h - 1);
    gl.uniform4f(u.uCref, ref.cref[0], ref.cref[1], ref.cref[2], ref.cref[3]);
    gl.uniform4f(u.uLref, ref.lref[0], ref.lref[1], ref.lref[2], ref.lref[3]);
    // looser linearisation while moving (faster), tighter for the resting picture
    // a fold's distance does not stretch with the tolerance: never widen for folded formulas
    const bias0 = job.strict ? this.biasStatic : this.biasMotion;
    const bias = ref.spec.kind === 'abs' ? Math.min(0, bias0) : bias0;
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
    job.dirty = true;
    job.passes++;
    job.steps += K;
    job.hsteps += K;
    this.stats.steps = K;
    this.stats.passes = job.passes;
    if (!ext) job.K = Math.min(262144, job.K * 1.25);

    this.afterPass(job, dst);
  }

  private afterPass(job: Job, dst: Target): void {
    const gl = this.gl;
    // is any pixel still running with iterations left under the limit?
    if (this.counts.length < 4) {
      const cq = gl.createQuery()!;
      // nothing is written, so any target as large as the job will do - but not the
      // canvas: touching it would put an empty frame on screen when nothing else is drawn
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.compFbo);
      gl.colorMask(false, false, false, false);
      gl.useProgram(this.prog.count.p);
      this.bindTex(0, gl.TEXTURE_2D, dst.s1);
      gl.uniform1i(this.prog.count.u.uS1, 0);
      gl.uniform1f(this.prog.count.u.uN, job.N);
      gl.beginQuery(gl.ANY_SAMPLES_PASSED, cq);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.endQuery(gl.ANY_SAMPLES_PASSED);
      gl.colorMask(true, true, true, true);
      this.counts.push({ q: cq, job, epoch: job.epoch });
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
      if (t.first && !t.job.aa && t.job.share >= 0.1) {
        // (a pass that takes over an unknown part of an earlier picture says nothing about the cost of a step)
        const c = ms / (t.K * t.job.w * t.job.h * t.job.share);
        this.c0 = this.c0 * 0.5 + c * 0.5;
      }
      if (t.first && t.job.aa && t.job.layer === this.aaKFor) this.aaK[t.job.carry] = Math.max(8, Math.min(262144, (t.K * this.targetMs) / Math.max(ms, 0.05)));
      if (t.job === this.job) {
        const want = (t.K * this.targetMs) / Math.max(ms, 0.05);
        t.job.K = Math.max(8, Math.min(262144, Math.min(want, t.job.K * 2)));
      }
    }
    if (this.restQ && gl.getQueryParameter(this.restQ, gl.QUERY_RESULT_AVAILABLE)) {
      const ms = gl.getQueryParameter(this.restQ, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(this.restQ);
      this.restQ = null;
      if (!gl.getParameter(ext.GPU_DISJOINT_EXT) && ms > 0) this.stats.restMs += (ms - this.stats.restMs) * 0.2;
    }
    while (this.counts.length) {
      const c = this.counts[0];
      if (!gl.getQueryParameter(c.q, gl.QUERY_RESULT_AVAILABLE)) break;
      this.counts.shift();
      if (!gl.getQueryParameter(c.q, gl.QUERY_RESULT) && c.epoch === c.job.epoch) c.job.noBudget = true;
      gl.deleteQuery(c.q);
    }
    const ev = this.evq;
    if (ev && gl.getSyncParameter(ev.sync, gl.SYNC_STATUS) === gl.SIGNALED) {
      this.evq = null;
      const n = ev.cw * ev.ch * 4;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.evPbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.evBuf, 0, n);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.deleteSync(ev.sync);
      let total = 0;
      let densest = 0;
      let dense = 0;
      let unknown = 0;
      let patch = 0;
      for (let i = 0; i < n; i += 4) {
        const v = this.evBuf[i];
        const q = this.evBuf[i + 1];
        total += v;
        unknown += q;
        if (q > patch) patch = q;
        if (v > dense) dense = v;
        // a patch coming in is a block whose late pixels finish close together;
        // dust sheds its pixels at any count whatever, round after round
        if (v > densest && this.evBuf[i + 3] - this.evBuf[i + 2] <= 0.15 * ev.job.N) densest = v;
      }
      if (ev.job === this.job && ev.epoch === ev.job.epoch && ev.job.stage === 1) this.decide(ev.job, total, densest, dense, unknown, patch);
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
    job.lastActive = f;
    if (job.layer && !job.aa) job.layer.known = 1 - f;
    job.fracs.push(f);
    job.fracs.push(job.steps);
    if (this.onProbe) this.onProbe({ view: job.view, aspect: this.W / this.H, w: job.w, h: job.h, data: d, active: f });
  }

  /** Writes a job as it stands into one slice of `tex`. */
  private resolve(job: Job, tex: WebGLTexture, slot: number, final: boolean): void {
    const gl = this.gl;
    const { p, u } = this.prog.resolve;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tex, 0, slot);
    gl.viewport(0, 0, job.w, job.h);
    gl.useProgram(p);
    this.bindTex(0, gl.TEXTURE_2D, this.targets[job.front].s1);
    gl.uniform1i(u.uS1, 0);
    gl.uniform1i(u.uFinal, final ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Copy a job into its layer of the stack, taking a slot for it the first time. */
  private snapshot(job: Job, final: boolean, t: number): void {
    let L = job.layer;
    if (!L) {
      // one that takes over a picture on its grid is as dear as that was, and needs the limit that needed
      const from = job.exact ? job.src : null;
      L = { slot: this.freeSlot(job), view: job.view, w: job.w, h: job.h, N: job.N, Naa: from ? Math.min(job.N, from.Naa) : job.N, cost: from ? from.cost : 0, tag: job.tag, complete: false, known: 0, done: false, strict: job.strict, born: t, seq: ++this.layerSeq };
      this.layers.push(L);
      job.layer = L;
      // a picture of another formula or parameter waits out of sight, unless there is nothing else to show
      if (!this.layers.some((x) => x.tag === this.showTag)) this.reveal(job);
    }
    this.resolve(job, this.stackTex, L.slot, final);
    L.N = job.N;
    job.dirty = false;
    // one of another parameter than what is shown takes its place once it has a good part to show
    if (L.tag !== this.showTag && ((t - job.t0 > 250 && L.known >= 0.3) || t - job.t0 > 1500)) this.reveal(job);
  }

  private freeSlot(job: Job): number {
    if (this.layers.length >= MAXL) {
      const out = this.evict(job);
      this.layers = this.layers.filter((x) => x !== out);
      if (this.aa.layer === out) this.aa = { layer: null, count: 0, from: null };
    }
    for (let s = 0; s < MAXL; s++) if (!this.layers.some((x) => x.slot === s)) return s;
    return 0;
  }

  /** Octaves by which a layer's grain is coarser than the screen's at view `now`. */
  private grain(L: Layer, now: View): number {
    return L.view.ls - Math.log2(L.h) - (now.ls - Math.log2(this.H));
  }

  /** The stack is full: pick the layer that will be missed least. */
  private evict(job: Job): Layer {
    const now = this.lastView || job.view;
    const info = this.layers.map((L) => ({ L, d: this.grain(L, now), seen: !!this.place(now, L) }));
    // leftovers of another picture, layers off the screen, layers far too coarse or too fine to matter
    let out: Layer | null = null;
    let worst = 0;
    for (const i of info) {
      const s = i.L.tag !== this.showTag && i.L.tag !== job.tag ? 1e9 : !i.seen ? 1e6 : i.d > 6 ? i.d : i.d < -5 ? -i.d : 0;
      if (s > worst) {
        worst = s;
        out = i.L;
      }
    }
    if (out) return out;
    // then the emptiest: a layer that knows next to nothing
    for (const i of info) {
      if (!i.L.complete && i.L !== job.layer && i.L.known < 0.05 && (!out || i.L.seq < out.seq)) out = i.L;
    }
    if (out) return out;
    info.sort((a, b) => a.d - b.d || a.L.seq - b.L.seq);
    // two of the same grain: the unfinished one, or else the older, is the spare
    for (let i = 0; i + 1 < info.length; i++) {
      if (info[i + 1].d - info[i].d < 0.08) return info[i].L.complete && !info[i + 1].L.complete ? info[i + 1].L : info[i].L;
    }
    // otherwise thin the ladder out where it is densest; its rungs lie closer together near the screen's own grain
    const u = (d: number) => Math.sign(d) * Math.log(1 + Math.abs(d) / 0.5);
    let k = 1;
    let gap = Infinity;
    for (let i = 1; i < info.length - 1; i++) {
      // a finished picture is worth more than a rung's place: it goes only when nothing else can
      const g = u(info[i + 1].d) - u(info[i - 1].d) + (info[i].L.complete ? 100 : 0);
      if (g < gap) {
        gap = g;
        k = i;
      }
    }
    return info[k].L;
  }

  /** A finished layer has no gaps: whatever it covers at no finer a grain is of no more use. */
  private prune(L: Layer): void {
    const a = this.W / this.H;
    const g = L.view.ls - Math.log2(L.h);
    this.layers = this.layers.filter((M) => {
      if (M === L || M.tag !== L.tag) return true;
      if (M.view.ls - Math.log2(M.h) < g - 0.01) return true; // finer: still the better picture of its part
      const r = Math.pow(2, M.view.ls - L.view.ls);
      if (!(r < 256)) return true;
      const [dx, dy] = offsetIn(L.view, M.view);
      return !(Math.abs(dx) + r * a <= a * 1.0005 && Math.abs(dy) + r <= 1.0005);
    });
    if (this.aa.layer && !this.layers.includes(this.aa.layer)) this.aa = { layer: null, count: 0, from: null };
  }

  /** Where a layer lies on a screen showing `now` (null: nowhere), and whether it covers all of it. */
  private place(now: View, L: Layer): { ratio: number; dx: number; dy: number; full: boolean } | null {
    const ratio = Math.pow(2, now.ls - L.view.ls);
    if (!(ratio < 256 && ratio > 1 / 256)) return null;
    const [dx, dy] = offsetIn(L.view, now);
    const a = this.W / this.H;
    if (!(Math.abs(dx) < a * (1 + ratio) && Math.abs(dy) < 1 + ratio)) return null;
    return { ratio, dx, dy, full: Math.abs(dx) + ratio * a <= a * 1.0005 && Math.abs(dy) + ratio <= 1.0005 };
  }

  private display(now: View, t: number): void {
    const gl = this.gl;
    const { p, u } = this.prog.display;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(p);

    // Best layer first: the finished picture of this very view, then whatever comes
    // nearest the screen's grain (a little too fine is better than a little too coarse).
    const list: { L: Layer; at: { ratio: number; dx: number; dy: number; full: boolean }; cost: number }[] = [];
    for (const L of this.layers) {
      if (L.tag !== this.showTag || (!L.complete && L.known < 0.02)) continue; // nothing to show yet
      const at = this.place(now, L);
      if (!at) continue;
      const d = this.grain(L, now);
      list.push({ L, at, cost: L === this.exact ? -1 : d >= 0 ? d : -0.4 * d });
    }
    list.sort((x, y) => x.cost - y.cost || y.L.seq - x.L.seq);
    let behind = list.length ? 1 : 0;
    for (const x of list) {
      if (x.at.full && x.L.done) behind = Math.min(behind, Math.max(0, this.grain(x.L, now)));
    }
    this.behind = behind;
    // Just after a change of fractal there is nothing to draw yet. Drawing nothing at
    // all leaves the last frame on screen, which beats a black flash between two pictures.
    if (t < this.holdUntil && !list.some((x) => x.L.complete || x.L.known >= 0.3)) return;
    // with every spot known the picture goes straight to the screen; otherwise by way of FILL
    let direct = false;
    for (let i = 0; i < list.length; i++) {
      const { L, at } = list[i];
      const f = Math.min(1, (t - L.born) / Math.max(1, this.fadeMs));
      this.lx.set([at.ratio, at.dx, at.dy, L.slot], i * 4);
      // a layer that ends within the screen ends softly, not as a rectangle
      this.lsz.set([L.w, L.h, f * f * (3 - 2 * f), at.full ? 0 : Math.max(8, 0.03 * L.h)], i * 4);
      // nothing shows through a finished layer that covers the screen
      if (L.complete && at.full && f >= 1) {
        list.length = i + 1;
        direct = true;
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, direct ? null : this.compFbo);
    this.bindTex(0, gl.TEXTURE_2D_ARRAY, this.stackTex);
    gl.uniform1i(u.uStack, 0);
    this.bindTex(1, gl.TEXTURE_2D_ARRAY, this.aaTexs[this.aaCur]);
    gl.uniform1i(u.uAA, 1);
    gl.uniform1i(u.uCount, list.length);
    this.stats.layers = list.length;
    gl.uniform4fv(u.uLX, this.lx);
    gl.uniform4fv(u.uLS, this.lsz);
    gl.uniform1i(u.uAACount, list.length && list[0].L === this.exact && this.aa.layer === this.exact ? this.aa.count : 0);
    gl.uniform1i(u.uDirect, direct ? 1 : 0);
    this.bindTex(2, gl.TEXTURE_2D, this.palTex);
    gl.uniform1i(u.uPal, 2);
    gl.uniform2f(u.uRes, this.W, this.H);
    gl.uniform1f(u.uAspect, this.W / this.H);
    gl.uniform1f(u.uPhase, this.phase);
    gl.uniform1f(u.uDensity, this.density);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (direct) return;
    // fill what no layer knows from the colours around it (see FILL)
    const n = this.pull.length;
    const down = this.prog.down;
    gl.useProgram(down.p);
    gl.uniform1i(down.u.uSrc, 3);
    for (let i = 0; i < n; i++) {
      const d = this.pull[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
      gl.viewport(0, 0, d.w, d.h);
      this.bindTex(3, gl.TEXTURE_2D, i ? this.pull[i - 1].tex : this.compTex);
      gl.uniform2f(down.u.uSize, d.w, d.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    const up = this.prog.up;
    gl.useProgram(up.p);
    gl.uniform1i(up.u.uOwn, 3);
    gl.uniform1i(up.u.uUp, 4);
    for (let i = n - 1; i >= 0; i--) {
      const d = this.push[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
      gl.viewport(0, 0, d.w, d.h);
      this.bindTex(3, gl.TEXTURE_2D, this.pull[i].tex);
      this.bindTex(4, gl.TEXTURE_2D, i < n - 1 ? this.push[i + 1].tex : this.dummyTex);
      gl.uniform2f(up.u.uSize, d.w, d.h);
      gl.uniform1i(up.u.uTop, i === n - 1 ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    const fill = this.prog.fill;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(fill.p);
    this.bindTex(3, gl.TEXTURE_2D, this.compTex);
    this.bindTex(4, gl.TEXTURE_2D, n ? this.push[0].tex : this.dummyTex);
    gl.uniform1i(fill.u.uC, 3);
    gl.uniform1i(fill.u.uUp, 4);
    gl.uniform2f(fill.u.uRes, this.W, this.H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Synchronous read of the finished image at job pixel coordinates (tests only). */
  readNu(points: [number, number][]): number[] {
    const gl = this.gl;
    const out: number[] = [];
    const buf = new Float32Array(4);
    const L = this.exact || this.layers[this.layers.length - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.stackTex, 0, L ? L.slot : 0);
    for (const [x, y] of points) {
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, buf);
      out.push(buf[0]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}

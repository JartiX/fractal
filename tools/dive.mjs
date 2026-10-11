// Picture quality while moving: zooms without a stop and compares what is on
// screen mid-flight with the settled picture of the very same view.
//   node tools/dive.mjs <url> <formula> <bits> <secs> <speed> [outPrefix] [seed]
// Starts at a boundary point pinned down to 2^-(bits+secs*speed+12), seen from 2^(8-bits).
// speed < 0 zooms out instead (after first settling `secs*|speed|` octaves deeper).
// With HERE=1 it starts from the view in the url's hash instead (formula and bits are then ignored).
// With MANUAL=1 the view is zoomed directly, as by hand, instead of by the dive (which paces itself).
//   mad    mean absolute colour difference to the settled picture (0-255); floorMad is
//          what a perfect picture of the last view sampled a fraction of a pixel aside scores
//   sharp  gradient energy relative to the settled picture (1 = as crisp)
//   black  share of pixels shown black that are not black once settled
//   lost   share of pixels shown coloured that are black once settled
import puppeteer from 'puppeteer-core';
const [url, formula = 'mandelbrot', bits = '40', secs = '6', speed = '0.7', out, seed = '1'] = process.argv.slice(2);
const W = 1280;
const H = 720;
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: [`--window-size=${W},${H}`, '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate(`document.body.classList.add('clean')`);
const here = !!process.env.HERE;
if (!here) await page.evaluate((f) => window.__fx.setFormula(f), formula);
// one sample per pixel and still colours: the comparison is then about the motion alone
await page.evaluate(() => { window.__fx.setCycling(false); window.__fx.renderer.maxSamples = 1; });
if (process.env.EVALJS) await page.evaluate(process.env.EVALJS);
await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
if (!here) {
  const S = Number(seed);
  const rnd = (k) => (S * 0.6180339887 + k * 0.7548776662) % 1;
  const span = Number(secs) * Number(speed);
  const deep = Number(bits) + Math.abs(span) + 12;
  let ok = false;
  for (let i = 0; i < 20 && !ok; i++) {
    ok = await page.evaluate((i, r, deep) => {
      const fx = window.__fx;
      const b = fx.randomBracket((r[0] + i * 0.37) % 1, (r[1] + i * 0.11) % 1, r[2], r[3]) || fx.findBracket();
      return !!b && fx.gotoBoundary(b[0], b[1], b[2], b[3], deep);
    }, i, [rnd(1), rnd(2), rnd(3), rnd(4)], deep);
  }
  if (!ok) { console.log('no boundary found'); await browser.close(); process.exit(1); }
  await page.evaluate((ls) => { window.__fx.view.ls = ls; }, 8 - Number(bits) - (span < 0 ? -span : 0));
}
await page.waitForFunction('window.__fx.idle()', { timeout: 180000, polling: 100 }).catch(() => console.log('not idle at start'));

// helpers living in the page
await page.evaluate(() => {
  const fx = window.__fx;
  window.__grab = () => {
    const gl = fx.renderer.gl;
    const { W, H } = fx.renderer;
    const px = new Uint8Array(W * H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  };
  window.__cmp = (a, b) => {
    const { W, H } = fx.renderer;
    let mad = 0, ga = 0, gb = 0, black = 0, lost = 0, n = 0, low = 0, nb = 0;
    // the same in the large: 8x8 block means, blind to dust and to a pixel's shift
    for (let y = 0; y + 8 <= H; y += 8) {
      for (let x = 0; x + 8 <= W; x += 8) {
        for (let c = 0; c < 3; c++) {
          let sa = 0, sb = 0;
          for (let j = 0; j < 8; j++) for (let i = 0; i < 8; i++) { const k = ((y + j) * W + x + i) * 4 + c; sa += a[k]; sb += b[k]; }
          low += Math.abs(sa - sb) / 64;
        }
        nb += 3;
      }
    }
    const lum = (p, i) => p[i] + p[i + 1] + p[i + 2];
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W - 1; x++) {
        const i = (y * W + x) * 4;
        mad += (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
        const la = lum(a, i), lb = lum(b, i);
        ga += Math.abs(lum(a, i + 4) - la) + Math.abs(lum(a, i + W * 4) - la);
        gb += Math.abs(lum(b, i + 4) - lb) + Math.abs(lum(b, i + W * 4) - lb);
        if (la < 12 && lb > 60) black++;
        if (lb < 12 && la > 60) lost++;
        n++;
      }
    }
    return { mad: +(mad / n).toFixed(2), low: +(low / nb).toFixed(2), sharp: +(ga / Math.max(1, gb)).toFixed(3), black: +((100 * black) / n).toFixed(2), lost: +((100 * lost) / n).toFixed(2) };
  };
});

const snaps = await page.evaluate(async (secs, speed, manual) => {
  const fx = window.__fx;
  const d = fx.dive;
  if (!manual) { d.on = true; d.auto = false; d.target = null; d.lost = 0; d.speed = speed; d.rate = speed; }
  window.__snaps = [];
  const log = [];
  const t0 = performance.now();
  let next = 1200;
  let last = t0;
  let worst = 0;
  const dts = [], rest = [], gpu = [], lay = [];
  const slow = [];
  await new Promise((done) => {
    const tick = (t) => {
      { const j = fx.renderer.job; const cur = j ? { id: j.id, p: j.passes, K: Math.round(j.K), N: j.N, st: j.stage, w: j.w, src: !!j.src, age: Math.round(t - j.t0) } : null; if (last > 0 && t - last > 15 && slow.length < 14) slow.push({ dt: Math.round(t - last), prev: window.__pj, cur, g: +fx.renderer.stats.gpuMs.toFixed(1) }); window.__pj = cur; }
      if (manual && last > 0) fx.view.ls -= speed * Math.min(0.1, (t - last) / 1000);
      if (last > 0) { dts.push(t - last); rest.push(fx.renderer.stats.restMs); gpu.push(fx.renderer.stats.gpuMs); lay.push(fx.renderer.stats.layers); }
      if (last > 0) worst = Math.max(worst, t - last); last = t;
      const r = fx.renderer;
      if (t - t0 >= next) {
        next += 1200;
        const v = fx.view;
        window.__snaps.push({ px: window.__grab(), v: [v.cx.toString(), v.cy.toString(), v.prec, v.ls] });
        log.push({ t: Math.round(t - t0), ls: +v.ls.toFixed(2), scale: +r.stats.scale.toFixed(2), N: r.stats.N, q: +r.q.toFixed(2), jobMs: Math.round(r.lastJob.ms), frame: Math.round(worst) });
        worst = 0;
        last = 0;
      }
      if (t - t0 < secs * 1000) requestAnimationFrame(tick); else done();
    };
    requestAnimationFrame(tick);
  });
  d.on = false; d.rate = 0;
  window.__slow = slow;
  dts.sort((a, b) => a - b);
  window.__pace = { frames: dts.length, p50: +dts[dts.length >> 1].toFixed(1), p95: +dts[Math.floor(dts.length * 0.95)].toFixed(1), max: +dts[dts.length - 1].toFixed(1), over25: dts.filter((x) => x > 25).length, over10: dts.filter((x) => x > 10).length, rest: +(rest.reduce((a, b) => a + b, 0) / rest.length).toFixed(2), restMax: +Math.max(...rest).toFixed(2), gpu: +(gpu.reduce((a, b) => a + b, 0) / gpu.length).toFixed(2), layers: +(lay.reduce((a, b) => a + b, 0) / lay.length).toFixed(2) };
  return log;
}, Number(secs), Number(speed), !!process.env.MANUAL);

const res = [];
for (let i = 0; i < snaps.length; i++) {
  await page.evaluate((i) => {
    const fx = window.__fx; const s = window.__snaps[i].v;
    fx.view.cx = BigInt(s[0]); fx.view.cy = BigInt(s[1]); fx.view.prec = s[2]; fx.view.ls = s[3];
  }, i);
  await new Promise((r) => setTimeout(r, 300));
  await page.waitForFunction('window.__fx.idle()', { timeout: 180000, polling: 100 }).catch(() => console.log('not idle'));
  const c = await page.evaluate((i) => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => {
    const truth = window.__grab();
    const snap = window.__snaps[i].px;
    window.__truth = truth;
    done(window.__cmp(snap, truth));
  }))), i);
  res.push({ ...snaps[i], ...c });
  if (out) {
    await page.screenshot({ path: `${out}-${i}-truth.png` });
    const b64 = await page.evaluate((i) => {
      const { W, H } = window.__fx.renderer; const src = window.__snaps[i].px;
      const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
      const ctx = cv.getContext('2d'); const im = ctx.createImageData(W, H);
      for (let y = 0; y < H; y++) im.data.set(src.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
      for (let k = 3; k < im.data.length; k += 4) im.data[k] = 255;
      ctx.putImageData(im, 0, 0);
      return cv.toDataURL('image/png').slice(22);
    }, i);
    (await import('node:fs')).writeFileSync(`${out}-${i}-live.png`, Buffer.from(b64, 'base64'));
  }
}
await page.evaluate((H) => window.__fx.nudge(0.74 / H, 0.5 / H), H);
await new Promise((r) => setTimeout(r, 300));
await page.waitForFunction('window.__fx.idle()', { timeout: 180000, polling: 100 }).catch(() => console.log('not idle'));
const floor = await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => done(window.__cmp(window.__grab(), window.__truth))))));
for (const r of res) console.log(JSON.stringify(r));
const pace = await page.evaluate('window.__pace');
if (process.env.SLOW) for (const x of await page.evaluate('window.__slow')) console.log(JSON.stringify(x));
const avg = (k) => +(res.reduce((s, r) => s + r[k], 0) / Math.max(1, res.length)).toFixed(3);
console.log(JSON.stringify({ formula, bits: Number(bits), speed: Number(speed), mad: avg('mad'), floorMad: floor.mad, low: avg('low'), floorLow: floor.low, sharp: avg('sharp'), black: avg('black'), lost: avg('lost'), scale: avg('scale'), pace }));
await browser.close();

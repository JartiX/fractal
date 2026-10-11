// Dragging the view: how soon the picture is finished once the view has come to a stop,
// with and without taking pixels over from the picture before (same pixel grid).
//   node tools/pan.mjs <url#view> [pixels=240]
import puppeteer from 'puppeteer-core';
const [url, px = '240'] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'], defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 } });
for (const noGrid of [true, false]) {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[error]', e.message));
  await page.goto(url);
  await page.waitForFunction('window.__fx');
  await page.evaluate((g) => { document.body.classList.add('clean'); window.__fx.setCycling(false); window.__fx.renderer.noGrid = g; }, noGrid);
  await page.waitForFunction('window.__fx.idle()', { timeout: 120000, polling: 50 });
  await new Promise((r) => setTimeout(r, 1500));
  await page.waitForFunction('window.__fx.idle()', { timeout: 120000, polling: 50 });
  const times = [];
  for (let rep = 0; rep < 3; rep++) {
    const n = 30, d = Number(px) * (rep % 2 ? -1 : 1);
    await page.mouse.move(640, 360);
    await page.mouse.down();
    for (let i = 1; i <= n; i++) { await page.mouse.move(640 + (d * i) / n, 360 + (d * 0.4 * i) / n); await new Promise((r) => setTimeout(r, 12)); }
    const during = await page.evaluate(() => ({ known: window.__fx.renderer.layers.map((L) => +L.known.toFixed(2)).slice(-3) }));
    await page.mouse.up();
    // the view glides on for a while after the mouse is let go: time from where it stops
    const stopped = await page.evaluate(() => new Promise((done) => {
      let last = '', same = 0;
      const tick = () => { const v = window.__fx.view; const k = v.cx + ',' + v.cy; same = k === last ? same + 1 : 0; last = k; if (same >= 2) done(performance.now()); else requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }));
    const base = await page.evaluate((stopped) => new Promise((done) => { const tick = () => { if (window.__fx.renderer.finalN() > 0) done(Math.round(performance.now() - stopped)); else requestAnimationFrame(tick); }; tick(); }), stopped);
    const t0 = Date.now() - base;
    await page.waitForFunction('window.__fx.idle()', { timeout: 120000, polling: 20 }).catch(() => {});
    times.push({ base, idle: Date.now() - t0 });
    await new Promise((r) => setTimeout(r, 300));
  }
  console.log(noGrid ? 'worked out anew:  ' : 'taken over:       ', times.map((t) => `picture after ${t.base} ms, smoothed after ${t.idle} ms`).join(' | '));
  if (!noGrid) {
    // The picture put together from pieces must be the one a fresh render gives. Pixel by
    // pixel that cannot be asked: where detail is finer than a pixel (dust), any two renders
    // differ - another reference point alone moves the picture by a thousandth of a pixel,
    // and there that is another value. So: where the fresh picture is smooth, how far is
    // the other displaced against it (difference over slope, in pixels)?
    const diff = await page.evaluate(async () => {
      const fx = window.__fx; const R = fx.renderer; const gl = R.gl; const { W, H } = R;
      const shot = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => { const b = new Uint8Array(W * H * 4); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, b); done(b); })));
      const nu = () => { const b = new Float32Array(W * H * 4); gl.bindFramebuffer(gl.FRAMEBUFFER, R.accFbo); gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, R.stackTex, 0, R.exact.slot); gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, b); gl.bindFramebuffer(gl.FRAMEBUFFER, null); const o = new Float32Array(W * H); for (let i = 0; i < W * H; i++) o[i] = b[i * 4]; return o; };
      const lim = () => `N ${R.exact.N}, samples to ${R.exact.Naa}`;
      const a = await shot(); const na = nu(); const la = lim();
      R.noGrid = true; R.layers = []; R.aa = { layer: null, count: 0, from: null };
      await new Promise((r) => setTimeout(r, 300));
      while (!fx.idle()) await new Promise((r) => setTimeout(r, 50));
      const b = await shot(); const nb = nu();
      let sum = 0;
      for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
      let smooth = 0, off = 0, worst = 0, state = 0;
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const i = y * W + x; const v = nb[i], u = na[i];
        // inside in one, escaped in the other (a limit apart does that at the rim; count the rest)
        if ((v >= 0) !== (u >= 0) && (v === -1 || u === -1)) state++;
        if (!(v >= 0 && u >= 0)) continue;
        const l = nb[i - 1], r = nb[i + 1], d = nb[i - W], t = nb[i + W];
        if (!(l >= 0 && r >= 0 && d >= 0 && t >= 0)) continue;
        const g = Math.hypot((r - l) / 2, (t - d) / 2);
        if (!(g > 0.02 && g < 3 && Math.max(Math.abs(l + r - 2 * v), Math.abs(d + t - 2 * v)) < 0.5 * g)) continue;
        smooth++;
        const s = Math.abs(u - v) / g;
        if (s > 0.02) off++;
        if (s > worst) worst = s;
      }
      return { smoothPixels: smooth, displacedOver2Percent: off, worstPx: +worst.toFixed(4), insideMismatch: state, meanLevel: +(sum / a.length).toFixed(3), samples: R.stats.samples, takenOver: la, fresh: lim() };
    });
    console.log('                   against a fresh render of the same view:', JSON.stringify(diff));
  }
  await page.close();
}
await browser.close();

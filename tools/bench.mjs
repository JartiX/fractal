// Runs a dive for a few seconds and reports frame pacing.
//   node tools/bench.mjs <url> <seconds> [out.png] [speed]
import puppeteer from 'puppeteer-core';
const [url, secs = '8', out, speed = '1'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: ['--window-size=1920,1080', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'],
  defaultViewport: { width: 1920, height: 1080, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 60000, polling: 100 }).catch(() => console.log('not idle at start'));
await page.keyboard.press('Space'); // start the autopilot the way a user does
const res = await page.evaluate(async (secs, speed) => {
  const fx = window.__fx;
  fx.dive.speed = speed;
  const dts = [], qs = [], gpu = [], scale = [];
  const ls0 = fx.view.ls;
  let last = performance.now(); const t0 = last;
  await new Promise((done) => {
    const tick = (t) => {
      dts.push(t - last); last = t;
      qs.push(fx.renderer.q); gpu.push(fx.renderer.stats.gpuMs); scale.push(fx.renderer.stats.scale);
      if (t - t0 < secs * 1000) requestAnimationFrame(tick); else done();
    };
    requestAnimationFrame(tick);
  });
  fx.dive.on = false; fx.dive.speed = 0.7;
  await new Promise((r) => setTimeout(r, 1500));
  dts.shift();
  const s = [...dts].sort((a, b) => a - b);
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  return { frames: dts.length, avg: avg(dts), p50: s[s.length >> 1], p99: s[Math.floor(s.length * 0.99)], max: s[s.length - 1], over25: dts.filter((d) => d > 25).length, over50: dts.filter((d) => d > 50).length, qEnd: qs[qs.length - 1], qMin: Math.min(...qs), scaleAvg: avg(scale), gpuAvg: avg(gpu), gpuMax: Math.max(...gpu), ls0, ls1: fx.view.ls, info: fx.info(), hash: location.hash };
}, Number(secs), Number(speed));
console.log(JSON.stringify(res));
if (out) await page.screenshot({ path: out });
await browser.close();

// Runs the autopilot and measures how smoothly the camera moves.
//   node tools/autopilot.mjs <url> <seconds>
import puppeteer from 'puppeteer-core';
const [url, secs = '20'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 60000 }).catch(() => {});
await page.keyboard.press('Space');
const res = await page.evaluate(async (secs) => {
  const fx = window.__fx;
  // per frame: screen-space pan velocity (half-heights/s) and zoom rate (octaves/s)
  const vel = [], rate = [];
  let last = performance.now(), prev = fx.at(640, 360), t0 = last;
  const num = (a, b) => { const p = Math.max(a[2], b[2]); return [0, 1].map((i) => Number((BigInt(a[i]) << BigInt(p - a[2])) - (BigInt(b[i]) << BigInt(p - b[2]))) / 2 ** p / 2 ** b[3]); };
  await new Promise((done) => {
    const tick = (t) => {
      const dt = (t - last) / 1000; last = t;
      const cur = fx.at(640, 360);
      const d = num(cur, prev);
      vel.push([d[0] / dt, d[1] / dt]); rate.push((prev[3] - cur[3]) / dt);
      prev = cur;
      if (t - t0 < secs * 1000) requestAnimationFrame(tick); else done();
    };
    requestAnimationFrame(tick);
  });
  // jolts: frame-to-frame change in velocity, as a fraction of the screen per second
  const jolt = []; const rj = [];
  for (let i = 2; i < vel.length; i++) { jolt.push(Math.hypot(vel[i][0] - vel[i - 1][0], vel[i][1] - vel[i - 1][1])); rj.push(Math.abs(rate[i] - rate[i - 1])); }
  const q = (a, p) => [...a].sort((x, y) => x - y)[Math.floor(a.length * p)];
  const speed = vel.map((v) => Math.hypot(v[0], v[1]));
  return { frames: vel.length, panSpeedMax: Math.max(...speed), joltP99: q(jolt, 0.99), joltMax: Math.max(...jolt), rateJoltMax: Math.max(...rj), rateMin: Math.min(...rate.slice(5)), rateMax: Math.max(...rate.slice(5)), reversals: rate.filter((r, i) => i > 5 && r < -0.05).length, ls: fx.view.ls };
}, Number(secs));
console.log(JSON.stringify(res));
await browser.close();

// Loads the app in headless Chrome, waits for the picture to finish, saves a
// screenshot and compares sampled pixels against brute-force BigInt iteration.
//   node tools/shot.mjs <url> <out.png> [--check N] [--wait ms] [--eval "js"]
import puppeteer from 'puppeteer-core';

const [url, out, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(name);
  return i >= 0 ? rest[i + 1] : def;
};
const check = Number(opt('--check', 0));
const wait = Number(opt('--wait', 60000));
const script = opt('--eval', '');

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
if (script) await page.evaluate(script);
const t0 = Date.now();
let idle = false;
try {
  await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: wait, polling: 100 });
  idle = true;
} catch {}
console.log('idle:', idle, 'after', Date.now() - t0, 'ms');
console.log(JSON.stringify(await page.evaluate('window.__fx && window.__fx.info()')));
console.log(await page.evaluate(`(() => { const gl = window.__fx.renderer.gl; const e = gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'n/a'; })()`));
await page.screenshot({ path: out });

if (check > 0) {
  const res = await page.evaluate((n) => {
    const fx = window.__fx;
    const { W, H } = fx.renderer;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i * 0.61803398875) % 1;
      const b = (i * 0.75487766625) % 1;
      pts.push([Math.floor(a * W), Math.floor(b * H)]);
    }
    const gpu = fx.renderer.readNu(pts);
    const t = performance.now();
    const cpu = pts.map((p) => fx.bruteNu(p[0], p[1]));
    // local slope of nu, to express errors as an equivalent shift in pixels
    const grad = pts.map((p, i) => {
      const a = fx.bruteNu(p[0] + 0.01, p[1]);
      const b = fx.bruteNu(p[0], p[1] + 0.01);
      return a < 0 || b < 0 || cpu[i] < 0 ? NaN : Math.hypot(a - cpu[i], b - cpu[i]) / 0.01;
    });
    return { pts, gpu, cpu, grad, ms: performance.now() - t };
  }, check);
  let bad = 0;
  const errs = [];
  res.pts.forEach((p, i) => {
    const g = res.gpu[i];
    const c = res.cpu[i];
    const err = g < 0 || c < 0 ? (g < 0 && c < 0 ? 0 : Infinity) : Math.abs(g - c);
    errs.push(err);
    if (err > 0.05) {
      bad++;
      if (bad <= 4) console.log(`  px ${p} gpu ${g.toFixed(4)} cpu ${c.toFixed(4)}`);
    }
  });
  const shift = res.pts.map((p, i) => Math.abs(res.gpu[i] - res.cpu[i]) / res.grad[i]).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const mism = res.pts.filter((p, i) => (res.gpu[i] < 0) !== (res.cpu[i] < 0)).length;
  console.log(`shift px: n ${shift.length}, median ${shift[shift.length >> 1].toExponential(2)}, p90 ${shift[Math.floor(shift.length * 0.9)].toExponential(2)}, max ${shift[shift.length - 1].toExponential(2)}; inside/outside mismatches ${mism}`);
  errs.sort((a, b) => a - b);
  console.log(`check: ${check} px, bad ${bad}, median err ${errs[errs.length >> 1].toExponential(2)}, p90 ${errs[Math.floor(errs.length * 0.9)].toExponential(2)}, brute ${res.ms.toFixed(0)} ms`);
}
await browser.close();

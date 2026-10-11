// How long a picture takes to settle, for every fractal: the home view and a few
// boundary spots at the given depths. Flags anything that drags on.
//   node tools/settle.mjs <url> [formulas|all] [bits,bits,...] [outPrefix]
// With outPrefix, also saves what is on screen 350 ms into each render and when settled.
import puppeteer from 'puppeteer-core';
const [url, which = 'all', depths = '0,14,30', out] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate(`document.body.classList.add('clean')`);
await page.evaluate(() => { window.__fx.setCycling(false); });
if (process.env.EVALJS) await page.evaluate(process.env.EVALJS);
const all = await page.evaluate('__fx.formulas');
const ids = which === 'all' ? all : which.split(',');
let worst = 0;
for (const id of ids) {
  for (const bits of depths.split(',').map(Number)) {
    await page.evaluate((id) => window.__fx.setFormula(id), id);
    if (bits > 0) {
      await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
      const ok = await page.evaluate((bits) => {
        const fx = window.__fx;
        for (let i = 0; i < 12; i++) {
          const b = fx.randomBracket((0.31 + i * 0.37) % 1, (0.57 + i * 0.11) % 1, 0.73, 0.19) || fx.findBracket();
          if (b && fx.gotoBoundary(b[0], b[1], b[2], b[3], bits)) return true;
        }
        return false;
      }, bits);
      if (!ok) { console.log(id, bits, 'no boundary'); continue; }
    }
    const t0 = Date.now();
    await new Promise((r) => setTimeout(r, 350));
    if (out) await page.screenshot({ path: `${out}-${id}-${bits}-early.png` });
    const idle = await page.waitForFunction('window.__fx.idle()', { timeout: 60000, polling: 50 }).then(() => true, () => false);
    const ms = Date.now() - t0;
    const info = await page.evaluate(() => {
      const fx = window.__fx; const { W, H } = fx.renderer; const pts = [];
      for (let i = 0; i < 400; i++) pts.push([Math.floor(((i * 0.61803398875) % 1) * W), Math.floor(((i * 0.75487766625) % 1) * H)]);
      const v = fx.renderer.readNu(pts);
      const i = fx.info();
      return { N: i.N, N0: i.N0, proven: v.filter((x) => x === -1).length / 4, assumed: v.filter((x) => x === -1.25).length / 4 };
    });
    worst = Math.max(worst, ms);
    console.log(`${id.padEnd(11)} bits ${String(bits).padStart(3)}  ${idle ? '' : 'NOT IDLE '}${String(ms).padStart(6)} ms  N ${info.N} (from ${info.N0})  inside: proven ${info.proven}% assumed ${info.assumed}%${ms > 8000 ? '   <-- slow' : ''}`);
    if (out) await page.screenshot({ path: `${out}-${id}-${bits}.png` });
  }
}
console.log('worst', worst, 'ms');
await browser.close();

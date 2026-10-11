// Julia parameter changes: right-drag and the K morph. Reports how often the
// picture is redrawn and how even the frames are.
//   node tools/morph.mjs <url> [out.png]
import puppeteer from 'puppeteer-core';
const [url, out] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 60000 });
const sleep = (t) => new Promise((r) => setTimeout(r, t));
const c = () => page.evaluate('location.hash.match(/jr=[^&]*&ji=[^&]*/)?.[0]');
await sleep(800);
console.log('start      ', await c());
await page.mouse.move(640, 360);
await page.mouse.down({ button: 'right' });
for (let i = 1; i <= 20; i++) { await page.mouse.move(640 + i * 4, 360 - i * 2); await sleep(16); }
await page.mouse.up({ button: 'right' });
await sleep(1200);
console.log('after drag ', await c(), JSON.stringify(await page.evaluate('({dive: __fx.dive.on})')));
await page.keyboard.press('KeyK');
const res = await page.evaluate(async () => {
  const fx = window.__fx; const dts = []; const refs = new Set();
  let last = performance.now(); const t0 = last;
  await new Promise((done) => { const tick = (t) => { dts.push(t - last); last = t; if (fx.renderer.ref) refs.add(fx.renderer.ref.id); if (t - t0 < 6000) requestAnimationFrame(tick); else done(); }; requestAnimationFrame(tick); });
  dts.shift();
  return { frames: dts.length, over25: dts.filter((d) => d > 25).length, max: Math.max(...dts), redrawsPerSec: refs.size / 6 };
});
if (out) await page.screenshot({ path: out });
await page.keyboard.press('KeyK');
await sleep(1200);
console.log('after morph', await c(), JSON.stringify(res));
await browser.close();

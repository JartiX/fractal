// Screenshots taken in the middle of a continuous wheel zoom, to look for flicker.
//   node tools/motion.mjs <url> <outPrefix> [budgetMs]
import puppeteer from 'puppeteer-core';
const [url, out, ms = '0'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.evaluate(`document.body.classList.add('clean'); window.__ms = ${ms} || undefined`);
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 60000 }).catch(() => {});
const sleep = (t) => new Promise((r) => setTimeout(r, t));
await page.mouse.move(640, 360);
// fraction of near-black pixels in each frame of the zoom
const black = [];
for (let i = 0; i < 60; i++) {
  await page.mouse.wheel({ deltaY: -100 });
  await sleep(50);
  if (process.env.DBG) console.log(i, JSON.stringify(await page.evaluate('({ls: +__fx.view.ls.toFixed(2), N: __fx.info().N, ref: __fx.renderer.ref && {id: __fx.renderer.ref.id, M: __fx.renderer.ref.M, esc: __fx.renderer.ref.escaped, max: __fx.renderer.ref.maxIter, prec: __fx.renderer.ref.prec, lv: __fx.renderer.ref.blaLevels}, off: __fx.renderer.refOffset(__fx.view), st: __fx.renderer.stats})')));
  if (i % 3 === 2) {
    const shot = await page.screenshot({ encoding: 'binary', path: i === 29 || i === 59 ? `${out}-${i}.png` : undefined });
    black.push(await page.evaluate(async (b64) => {
      const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
      const c = new OffscreenCanvas(320, 180); const g = c.getContext('2d'); g.drawImage(img, 0, 0, 320, 180);
      const d = g.getImageData(0, 0, 320, 180).data; let n = 0;
      for (let k = 0; k < d.length; k += 4) if (d[k] + d[k + 1] + d[k + 2] < 12) n++;
      return +(n / (320 * 180)).toFixed(3);
    }, Buffer.from(shot).toString('base64')));
  }
}
console.log('black fraction during zoom:', black.join(' '));
await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => console.log('not idle'));
await page.screenshot({ path: `${out}-rest.png` });
console.log(JSON.stringify(await page.evaluate('({ls: __fx.view.ls, q: __fx.renderer.q})')));
await browser.close();

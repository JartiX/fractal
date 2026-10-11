// How long reference orbits take: the exact orbit itself and its BLA table, per iteration.
//   node tools/reftime.mjs <url#view> [<url#view> ...]
// Views worth timing are ones whose orbit neither escapes nor closes a cycle, e.g.
//   "#re=-1.9&im=0&z=-1000&i=0.25"  (Mandelbrot on the real axis; i scales the iteration limit)
import puppeteer from 'puppeteer-core';
const urls = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--window-size=640,360', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'], defaultViewport: { width: 640, height: 360, deviceScaleFactor: 1 } });
for (const url of urls) {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[error]', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text().slice(0, 200)); });
  await page.goto(url);
  await page.waitForFunction('window.__fx');
  let best = null; const t0 = Date.now();
  while (Date.now() - t0 < 200000) {
    const s = await page.evaluate(() => { const i = window.__fx.info(); const r = window.__fx.renderer.ref; return r ? { M: r.M, prec: r.prec, esc: r.escaped, per: r.perA, ms: Math.round(i.refMs), orbit: Math.round(i.refOrbitMs), levels: r.blaLevels, idle: window.__fx.idle() } : null; }).catch(() => null);
    if (s && (!best || s.M >= best.M)) best = s;
    if (s && s.idle) break;
    await new Promise((r) => setTimeout(r, 300));
  }
  console.log(decodeURIComponent(url.slice(url.indexOf('#'), url.indexOf('#') + 34)).padEnd(34), JSON.stringify(best), best ? `orbit ${(best.orbit / best.M * 1000).toFixed(2)} us/iter, table ${((best.ms - best.orbit) / best.M * 1000).toFixed(3)} us/iter` : '');
  await page.close();
}
await browser.close();

// Renders the home view of every fractal, for a contact sheet.
//   node tools/gallery.mjs <baseUrl> <outPrefix>
import puppeteer from 'puppeteer-core';
const [url, out] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=640,360', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 640, height: 360, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text().slice(0, 300)); });
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate(`document.body.classList.add('clean')`);
await page.keyboard.press('KeyP');
const ids = await page.evaluate('__fx.formulas');
for (const id of ids) {
  await page.evaluate((id) => { __fx.setFormula(id); __fx.renderer.phase = 0; }, id);
  const ok = await page.waitForFunction('window.__fx.idle()', { timeout: 30000 }).then(() => true, () => false);
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: `${out}-${id}.png` });
  console.log(id, ok ? 'ok' : 'NOT IDLE');
}
console.log(ids.join(','));
await browser.close();

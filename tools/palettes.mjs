// Renders one view in every palette, for a contact sheet.
//   node tools/palettes.mjs <baseUrlWithHash> <outPrefix> <count>
import puppeteer from 'puppeteer-core';
const [url, out, count = '14'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=640,360', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 640, height: 360, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
for (let i = 0; i < Number(count); i++) {
  await page.goto('about:blank');
  await page.goto(`${url}&p=${i}`);
  await page.waitForFunction('window.__fx');
  await page.evaluate(`document.body.classList.add('clean')`);
  await page.keyboard.press('KeyP');
  await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
  await page.evaluate('__fx.renderer.phase = 0');
  await new Promise((r) => setTimeout(r, 150));
  await page.screenshot({ path: `${out}-${i}.png` });
}
await browser.close();

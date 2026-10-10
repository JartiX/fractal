// Drives real mouse/keyboard input and checks that the point under the cursor stays put.
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
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 30000 });
const at = (x, y) => page.evaluate((x, y) => window.__fx.at(x, y), x, y);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// error in pixels between two fixed-point positions, at the current scale
const errPx = (a, b) => {
  const p = Math.max(a[2], b[2]);
  const dx = Number((BigInt(a[0]) << BigInt(p - a[2])) - (BigInt(b[0]) << BigInt(p - b[2]))) / 2 ** p;
  const dy = Number((BigInt(a[1]) << BigInt(p - a[2])) - (BigInt(b[1]) << BigInt(p - b[2]))) / 2 ** p;
  return (Math.hypot(dx, dy) / 2 ** b[3]) * 360;
};
await page.mouse.move(900, 200);
let a = await at(900, 200);
for (let i = 0; i < 12; i++) { await page.mouse.wheel({ deltaY: -100 }); await sleep(40); }
await sleep(900);
let b = await at(900, 200);
console.log('wheel: zoomed', (a[3] - b[3]).toFixed(3), 'octaves, cursor point drift', errPx(a, b).toExponential(2), 'px');
a = await at(600, 400);
await page.mouse.move(600, 400); await page.mouse.down();
for (let i = 1; i <= 10; i++) { await page.mouse.move(600 + i * 10, 400 + i * 5); await sleep(16); }
b = await at(700, 450);
console.log('drag: grabbed point drift', errPx(a, b).toExponential(2), 'px');
await sleep(300); await page.mouse.up(); await sleep(600);
await page.keyboard.press('Digit3'); await sleep(200);
await page.mouse.click(640, 360, { count: 2 }); await sleep(2500);
console.log('dblclick dive:', JSON.stringify(await page.evaluate('({on: __fx.dive.on, auto: __fx.dive.auto, ls: __fx.view.ls})')));
await page.keyboard.press('Space'); await sleep(300);
console.log('space:', JSON.stringify(await page.evaluate('({on: __fx.dive.on})')));
await page.waitForFunction('window.__fx.idle()', { timeout: 30000 }).catch(() => console.log('not idle'));
if (out) await page.screenshot({ path: out });
await browser.close();

// Watches a picture being computed: iteration limit, stage, share of pixels still running.
//   node tools/prof.mjs <url> [bits] [seconds] [evalJs]
// With bits, first bisects to a boundary point at that depth (as tools/deep.mjs does).
import puppeteer from 'puppeteer-core';
const [url, bits = '', secs = '20', js = ''] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'], defaultViewport: { width: 1280, height: 720 } });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
if (js) await page.evaluate(js);
if (bits) {
  await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
  const ok = await page.evaluate((b) => { const fx = window.__fx; const br = fx.findBracket(); return !!br && fx.gotoBoundary(br[0], br[1], br[2], br[3], b); }, Number(bits));
  if (!ok) console.log('no boundary found');
}
const t0 = Date.now();
let last = '';
while (Date.now() - t0 < Number(secs) * 1000) {
  await new Promise((r) => setTimeout(r, 500));
  const s = await page.evaluate('(() => { const r = __fx.renderer; const j = r.job; const ref = r.ref; return j ? {N: j.N, stage: j.stage, aa: j.aa, K: Math.round(j.K), passes: j.passes, active: +j.lastActive.toFixed(4), gpuMs: +r.stats.gpuMs.toFixed(1), bla: j.useBla, ref: ref && [ref.lenA, ref.cap > 1e8 ? "open" : ref.cap, ref.perA, ref.interior]} : {idle: r.idle(__fx.view), samples: r.stats.samples, N: r.stats.N, ref: ref && [ref.lenA, ref.perA]}; })()');
  const line = JSON.stringify(s);
  if (line !== last) console.log(((Date.now() - t0) / 1000).toFixed(1), line);
  last = line;
  if (s.idle) break;
}
await browser.close();

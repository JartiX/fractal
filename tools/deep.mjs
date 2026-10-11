// Accuracy at depth for any formula: bisects to a boundary point with BigInt,
// views it at the requested depth and compares pixels with brute-force iteration.
//   node tools/deep.mjs <url> <bits> <ax> <ay> <bx> <by> [out.png] [nPixels]
// The two points must have different outcomes; pass "auto 0 0 0" to pick them from the home view.
import puppeteer from 'puppeteer-core';
const [url, bits, ax, ay, bx, by, out, n = '40'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate(`document.body.classList.add('clean')`);
await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
const br = ax === 'auto' ? await page.evaluate('window.__fx.findBracket()') : [ax, ay, bx, by].map(Number);
if (!br) { console.log('no two outcomes found in the home view'); await browser.close(); process.exit(1); }
const ok = await page.evaluate((b, r) => window.__fx.gotoBoundary(r[0], r[1], r[2], r[3], b), Number(bits), br);
if (!ok) { console.log('bad bracket: the two points have the same outcome'); await browser.close(); process.exit(1); }
const t0 = Date.now();
const idle = await page.waitForFunction('window.__fx.idle()', { timeout: 180000, polling: 100 }).then(() => true, () => false);
const info = await page.evaluate('window.__fx.info()');
const res = await page.evaluate((n) => {
  const fx = window.__fx; const { W, H } = fx.renderer; const pts = [];
  for (let i = 0; i < n; i++) pts.push([Math.floor(((i * 0.61803398875) % 1) * W), Math.floor(((i * 0.75487766625) % 1) * H)]);
  const gpu = fx.renderer.readNu(pts);
  const cpu = pts.map((p) => fx.bruteNu(p[0], p[1]));
  const grad = pts.map((p, i) => { const a = fx.bruteNu(p[0] + 0.01, p[1]); const b = fx.bruteNu(p[0], p[1] + 0.01); return a < 0 || b < 0 || cpu[i] < 0 ? NaN : Math.hypot(a - cpu[i], b - cpu[i]) / 0.01; });
  return { gpu, cpu, grad };
}, Number(n));
const shift = res.gpu.map((g, i) => Math.abs(g - res.cpu[i]) / res.grad[i]).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
const mism = res.gpu.filter((g, i) => (g < 0) !== (res.cpu[i] < 0)).length;
const esc = res.cpu.filter((c) => c >= 0).length;
const errs = res.gpu.map((g, i) => Math.abs(g - res.cpu[i])).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
const span = Math.max(...res.cpu) - Math.min(...res.cpu.filter((c) => c >= 0));
const q = (p) => (shift.length ? shift[Math.min(shift.length - 1, Math.floor(shift.length * p))].toExponential(1) : '-');
console.log(`2^-${bits}: idle ${idle} in ${Date.now() - t0} ms, N ${info.N}, ref ${info.ref ? info.ref.M : '-'}; escaped ${esc}/${n}, shift px median ${q(0.5)} p90 ${q(0.9)} max ${q(1)}, inside/outside mismatches ${mism}; |err| median ${errs[errs.length >> 1].toExponential(1)} max ${errs[errs.length - 1].toExponential(1)}, nu spread over the view ${span.toExponential(1)}`);
if (out) await page.screenshot({ path: out });
await browser.close();

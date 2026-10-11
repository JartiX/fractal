// Stress test: many random views of one fractal, at random depths and offsets
// from a boundary, each compared pixel by pixel with brute-force iteration.
//   node tools/stress.mjs <url> [views=12] [maxBits=60] [seed=1] [outPrefix]
// A pixel is "bad" when exact iteration gives one steady answer at that pixel and a
// twentieth of a pixel around it, and the GPU says something else. ("Touchy" pixels
// sit in chaotic dust, where the exact answer itself changes within that distance.)
import puppeteer from 'puppeteer-core';
const [url, views = '12', maxBits = '60', seed = '1', out, only = ''] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=960,540', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 960, height: 540, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate(`document.body.classList.add('clean')`);
if (process.env.EVALJS) await page.evaluate(process.env.EVALJS);
await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});

let rng = Number(seed) * 2654435761 % 4294967296;
const rand = () => ((rng = (rng * 1664525 + 1013904223) % 4294967296) / 4294967296);
let worst = 0, total = 0, totalBad = 0, totalLate = 0, totalBlack = 0;
const lines = [];
for (let v = 0; v < Number(views); v++) {
  const bits = 4 + Math.floor(rand() * (Number(maxBits) - 4));
  const pick = [rand(), rand(), rand(), rand(), rand() * 2 - 1, rand() * 2 - 1];
  if (only !== '' && v !== Number(only)) continue; // replay one view, with the same random draws
  // VIEWS="re,im,ls;re,im,ls" replaces the random boundary views with fixed ones
  const fixed = process.env.VIEWS ? process.env.VIEWS.split(';').map((t) => t.split(',').map(Number)) : null;
  if (fixed && v >= fixed.length) break;
  const ok = await page.evaluate((bits, r, f) => {
    const fx = window.__fx;
    if (f) return fx.goto(f[0], f[1], f[2]);
    const br = fx.randomBracket(r[0], r[1], r[2], r[3]);
    if (!br || !fx.gotoBoundary(br[0], br[1], br[2], br[3], bits)) return false;
    fx.nudge(r[4] * 0.8, r[5] * 0.8); // do not sit exactly on the bisected point
    return true;
  }, bits, pick, fixed ? fixed[v] : null);
  if (!ok) { lines.push(`view ${v}: no boundary found`); continue; }
  // Move about as a user would, so the reference orbit gets re-chosen from what is
  // on screen: a picture must not depend on which reference it was drawn with.
  await page.waitForFunction('window.__fx.idle()', { timeout: 120000, polling: 100 }).catch(() => {});
  for (const [dx, dy] of [[1.7, 0.3], [-3.2, -0.9], [1.5, 0.6]]) {
    await page.evaluate((dx, dy) => window.__fx.nudge(dx, dy), dx, dy);
    await new Promise((r) => setTimeout(r, 450));
  }
  const idle = await page.waitForFunction('window.__fx.idle()', { timeout: 120000, polling: 100 }).then(() => true, () => false);
  const res = await page.evaluate((n) => {
    const fx = window.__fx; const { W, H } = fx.renderer; const pts = [];
    for (let i = 0; i < n; i++) pts.push([Math.floor(((i * 0.61803398875) % 1) * W), Math.floor(((i * 0.75487766625) % 1) * H)]);
    const gpu = fx.renderer.readNu(pts);
    const cpu = pts.map((p) => fx.bruteNu(p[0], p[1]));
    // Where GPU and exact disagree, look at the exact values a twentieth of a pixel
    // around: in chaotic dust they scatter, and any of them is as right as another.
    const near = pts.map((p, i) => {
      const same = (gpu[i] < 0) === (cpu[i] < 0) && (cpu[i] < 0 || Math.abs(gpu[i] - cpu[i]) <= 0.5);
      if (same) return null;
      const out = [];
      for (let k = 0; k < 8; k++) out.push(fx.bruteNu(p[0] + 0.05 * Math.cos(k * 0.7854), p[1] + 0.05 * Math.sin(k * 0.7854)));
      return out;
    });
    // "Black that is not really black": pixels drawn as inside which would have
    // finished had the picture been given four times the iterations.
    const N = fx.info().N;
    let late = 0, black = 0;
    pts.forEach((p, i) => { if (gpu[i] < 0 && black < 8) { black++; if (fx.bruteNu(p[0], p[1], 4 * N) >= 0) late++; } });
    return { gpu, cpu, near, late, black, hash: location.hash, ls: fx.view.ls, info: fx.info() };
  }, 40);
  let bad = 0;
  let touchy = 0;
  res.gpu.forEach((g, i) => {
    const c = res.cpu[i];
    const around = res.near[i];
    if (!around) return; // agrees outright
    const all = [c, ...around];
    const lo = Math.min(...all), hi = Math.max(...all);
    // Where the exact picture itself changes within a twentieth of a pixel (mixed
    // outcomes, or iteration counts more than 1 apart) the pixel is chaotic dust:
    // no single value is the right one. Only a calm neighbourhood can show an error.
    const calm = lo < 0 ? hi < 0 : hi - lo <= 1;
    if (!calm) { touchy++; return; }
    bad++;
    if (only !== '') lines.push(`   px ${i}: gpu ${g.toFixed(3)} exact ${c.toFixed(3)} around [${around.map((v) => v.toFixed(1)).join(' ')}]`);
  });
  if (only !== '') lines.push('   ' + JSON.stringify(res.info).slice(0, 300));
  if (only !== '' && process.env.ORBIT) lines.push('   orbits: ' + JSON.stringify(await page.evaluate((k) => { const fx = window.__fx; const { W, H } = fx.renderer; return [0, 1, 2, 3].map((i) => { const p = [Math.floor(((i * 0.61803398875) % 1) * W), Math.floor(((i * 0.75487766625) % 1) * H)]; return fx.orbitAt(p[0], p[1], k).map((z) => z.map((v) => +v.toPrecision(9))); }); }, Number(process.env.ORBIT))));
  total += res.gpu.length; totalBad += bad; worst = Math.max(worst, bad); totalLate += res.late; totalBlack += res.black;
  lines.push(`view ${v}: 2^${res.ls.toFixed(0)} bad ${bad}/40, touchy ${touchy}, N ${res.info.N} (start ${res.info.N0}), black ${res.black} of which late ${res.late}${idle ? '' : ' (not idle)'}${bad > 0 || res.late > 0 ? '  <-- ' + res.hash.slice(0, 160) : ''}`);
  if (out && bad > 2) await page.screenshot({ path: `${out}-${v}.png` });
}
console.log(lines.join('\n'));
console.log(`TOTAL ${url.split('#')[1] || 'mandelbrot'}: ${totalBad}/${total} bad pixels (${((100 * totalBad) / Math.max(1, total)).toFixed(1)}%), worst view ${worst}/40; black pixels ${totalBlack}, of which would finish with 4x the iterations: ${totalLate}`);
await browser.close();

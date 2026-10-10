// Zooms continuously and measures frame-to-frame change; blinking shows up as spikes.
//   node tools/flicker.mjs <url> [budgetMs]
import puppeteer from 'puppeteer-core';
const [url, ms = '0'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.evaluate(`document.body.classList.add('clean'); window.__ms = ${ms} || undefined; __fx.renderer.phase = 0;`);
await page.keyboard.press('KeyP'); // no colour cycling, so change comes only from the picture
await page.waitForFunction('window.__fx && window.__fx.idle()', { timeout: 60000 }).catch(() => {});
await page.mouse.move(700, 300);
const sleep = (t) => new Promise((r) => setTimeout(r, t));
let stop = false;
const wheel = (async () => { while (!stop) { await page.mouse.wheel({ deltaY: -60 }); await sleep(30); } })();
// sample the canvas inside the page every animation frame
const res = await page.evaluate(async () => {
  const src = document.getElementById('view');
  const W = src.width, H = src.height, B = 16, bw = Math.floor(W / B), bh = Math.floor(H / B);
  const c = new OffscreenCanvas(W, H); const g = c.getContext('2d', { willReadFrequently: true });
  let prev = null, prev2 = null; const diffs = []; const info = [];
  let f1 = null, f2 = null, best = 0, snap = null;
  const t0 = performance.now();
  await new Promise((done) => {
    const tick = () => {
      g.drawImage(src, 0, 0);
      const d = g.getImageData(0, 0, W, H).data;
      // mean brightness of 16x16 blocks, so single-pixel speckle does not count
      const m = new Float32Array(bw * bh);
      for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) {
        let s = 0;
        for (let y = 0; y < B; y += 2) { let k = ((by * B + y) * W + bx * B) * 4; for (let x = 0; x < B; x += 2, k += 8) s += d[k] + d[k + 1] + d[k + 2]; }
        m[by * bw + bx] = s / (3 * B * B / 4);
      }
      // a blink: the block jumps and is back where it was one frame later
      if (prev2) { let n = 0; for (let k = 0; k < m.length; k++) if (Math.abs(prev[k] - prev2[k]) > 10 && Math.abs(m[k] - prev[k]) > 10 && Math.abs(m[k] - prev2[k]) < 4) n++; diffs.push(n / m.length); if (n > best && diffs.length > 60) { best = n; snap = [f2, f1, d.slice()]; } const st = __fx.renderer.stats; info.push(st.passes + '/' + st.scale.toFixed(2) + '/' + (__fx.renderer.ref ? __fx.renderer.ref.id : 0)); }
      prev2 = prev; prev = m; f2 = f1; f1 = d.slice();
      if (performance.now() - t0 < 6000) requestAnimationFrame(tick); else done();
    };
    requestAnimationFrame(tick);
  });
  const worst = diffs.indexOf(Math.max(...diffs));
  const pngs = [];
  for (const f of snap || []) { g.putImageData(new ImageData(f, W, H), 0, 0); const b = await c.convertToBlob({ type: 'image/png' }); const u8 = new Uint8Array(await b.arrayBuffer()); let bin = ''; for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode(...u8.subarray(i, i + 8192)); pngs.push(btoa(bin)); }
  const s = [...diffs].sort((a, b) => a - b);
  return { frames: diffs.length, median: s[s.length >> 1], p99: s[Math.floor(s.length * 0.99)], max: s[s.length - 1], blinks: diffs.filter((v) => v > 0.01).length, ls: __fx.view.ls, around: info.slice(Math.max(0, worst - 3), worst + 3).join(' '), pngs };
});
stop = true; await wheel;
const fs = await import('node:fs');
(res.pngs || []).forEach((b, i) => fs.writeFileSync(`${process.env.OUT || 'blink'}-${i}.png`, Buffer.from(b, 'base64')));
delete res.pngs;
console.log('share of blocks blinking (A-B-A within 3 frames):', JSON.stringify(res));
await browser.close();

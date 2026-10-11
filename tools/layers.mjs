// What the layer stack holds a few seconds into a dive (debugging aid).
//   node tools/layers.mjs <url> <formula> <bits> <secs> <speed>
import puppeteer from 'puppeteer-core';
const [url, id, bits, secs = '4', speed = '0.7'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  args: ['--window-size=1280,720', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=d3d11', '--hide-scrollbars'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[error]', e.message));
await page.goto(url);
await page.waitForFunction('window.__fx');
await page.evaluate((id) => { window.__fx.setCycling(false); window.__fx.setFormula(id); }, id);
await page.waitForFunction('window.__fx.idle()', { timeout: 60000 }).catch(() => {});
await page.evaluate((deep) => {
  const fx = window.__fx;
  for (let i = 0; i < 20; i++) {
    const b = fx.randomBracket((0.6180339887 + 0.7548776662 + i * 0.37) % 1, (0.6180339887 + 2 * 0.7548776662 + i * 0.11) % 1, (0.6180339887 + 3 * 0.7548776662) % 1, (0.6180339887 + 4 * 0.7548776662) % 1) || fx.findBracket();
    if (b && fx.gotoBoundary(b[0], b[1], b[2], b[3], deep)) return true;
  }
}, Number(bits) + Math.abs(Number(secs) * Number(speed)) + 12);
await page.evaluate((ls) => { window.__fx.view.ls = ls; }, 8 - Number(bits));
await page.waitForFunction('window.__fx.idle()', { timeout: 180000, polling: 100 }).catch(() => console.log('not idle at start'));
const dump = () => page.evaluate(() => {
  const fx = window.__fx; const r = fx.renderer; const gl = r.gl;
  const out = [];
  const buf = new Float32Array(4);
  for (const L of r.layers) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, r.accFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, r.stackTex, 0, L.slot);
    const c = { esc: 0, proven: 0, assumed: 0, unknown: 0 };
    for (let j = 0; j < 12; j++) for (let i = 0; i < 20; i++) {
      gl.readPixels(Math.floor((i + 0.5) / 20 * L.w), Math.floor((j + 0.5) / 12 * L.h), 1, 1, gl.RGBA, gl.FLOAT, buf);
      const v = buf[0];
      if (v >= 0) c.esc++; else if (v === -1) c.proven++; else if (v > -1.5) c.assumed++; else c.unknown++;
    }
    out.push({ seq: L.seq, slot: L.slot, ls: +L.view.ls.toFixed(2), w: L.w, N: L.N, complete: L.complete, ...c });
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const j = r.job;
  return { view: +fx.view.ls.toFixed(2), q: +r.q.toFixed(2), lead: Math.round(r.leadMs), need: Math.round(r.need), job: j && { N: j.N, w: j.w, ls: +j.view.ls.toFixed(2), st: j.stage, p: j.passes, age: Math.round(performance.now() - j.t0), act: j.lastActive, src: j.src && j.src.seq }, layers: out };
});
console.log(JSON.stringify(await dump()));
await page.evaluate((speed) => { const d = window.__fx.dive; d.on = true; d.auto = false; d.target = null; d.lost = 0; d.speed = speed; d.rate = speed; }, Number(speed));
for (let k = 0; k < Number(secs); k++) {
  await new Promise((r) => setTimeout(r, 1000));
  if (process.env.NODUMP) continue;
  const d = await dump();
  console.log(JSON.stringify({ view: d.view, q: d.q, lead: d.lead, need: d.need, job: d.job }));
  for (const L of d.layers) console.log('   ', JSON.stringify(L));
}
await browser.close();

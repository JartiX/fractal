// One of several threads building BLA tables: takes a stretch of the reference
// orbit, returns its levels (see bla.ts).

import { blaRange, BlaParams, BlaRange } from './bla';

export interface BlaJob {
  id: number;
  P: BlaParams;
  zr: Float64Array;
  zi: Float64Array;
  ze: Int32Array;
  sre: Uint8Array | null;
  n: number;
  upTo: number;
}

export interface BlaJobResult extends BlaRange {
  id: number;
}

// The page starts this thread and gives it one end of a channel; the reference worker,
// which holds the other end, sends the stretches and gets the levels back directly.
self.onmessage = (ev: MessageEvent<{ port: MessagePort }>) => {
  const port = ev.data.port;
  port.onmessage = (e: MessageEvent<BlaJob>) => {
    const m = e.data;
    const r = blaRange(m.P, m.zr, m.zi, m.ze, m.sre, m.n, m.upTo);
    const out: BlaJobResult = { id: m.id, ...r };
    const t = r.top;
    const transfer: Transferable[] = [t.a.buffer, t.b.buffer, t.ae.buffer, t.be.buffer, t.la.buffer, t.lb.buffer, t.lr.buffer];
    for (const l of r.levels) {
      transfer.push(l.A.buffer, l.E.buffer);
      if (l.B) transfer.push(l.B.buffer);
    }
    port.postMessage(out, transfer);
  };
};

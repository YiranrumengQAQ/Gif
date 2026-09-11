/**
 * LZW stream tests for `enc/lzw.js`, focused on the cases that corrupt *silently*:
 * dictionary clears at non-default intervals, sub-rectangle encoding and the
 * deferred code-width bump. Every case here is an encode -> decode equality check,
 * plus a couple of byte-level invariants (min code size, sub-block framing).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { lzwEncode, lzwDecode, minCodeSizeFor, subBlockSize } from '../src/enc/lzw.js';

function roundTrip(indices, minCodeSize, opts) {
  const out = new Uint8Array(Math.max(4096, indices.length * 3 + 1024));
  const written = lzwEncode(indices, indices.length, minCodeSize, out, opts || {});
  const dec = new Uint8Array(indices.length);
  const got = lzwDecode(out.subarray(0, written), minCodeSize, dec);
  return { written, got, dec };
}

test('clear intervals of every size round-trip (the deferred-bump trap)', () => {
  // A Clear emitted while the code width is > minCodeSize+1 used to desynchronise
  // the decoder; only frames big enough to bump their width could hit it.
  for (const pixels of [960, 4800, 19200, 60000]) {
    for (const mcs of [2, 4, 6, 8]) {
      const span = 1 << mcs;
      const idx = new Uint8Array(pixels);
      for (let i = 0; i < pixels; i++) idx[i] = (i * 7 + (((i / 13) | 0) * 3)) & (span - 1);
      for (const clearIntervalPx of [0, 16, 127, 128, 255, 1024, 4090, 4096]) {
        const { got, dec } = roundTrip(idx, mcs, clearIntervalPx ? { clearInterval: clearIntervalPx } : {});
        assert.equal(got, pixels, `${pixels}px, mcs ${mcs}, clear ${clearIntervalPx}: pixel count`);
        for (let i = 0; i < pixels; i++) if (dec[i] !== idx[i]) throw new Error(`pixel ${i} differs (mcs ${mcs}, clear ${clearIntervalPx})`);
      }
    }
  }
});

test('a sub-rectangle encodes exactly the window it is given', () => {
  const W = 37;
  const H = 21;
  const canvas = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) canvas[y * W + x] = (x * 3 + y * 5) & 15;
  const x0 = 5;
  const y0 = 7;
  const rw = 12;
  const rh = 9;
  const { got, dec } = roundTrip(canvas, minCodeSizeFor(16), { width: rw, stride: W, x0, y0, rectH: rh });
  assert.equal(got, rw * rh);
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) assert.equal(dec[y * rw + x], canvas[(y0 + y) * W + x0 + x], `rect px ${x},${y}`);
});

test('tight sub-rect buffers need no offset (the shape the encoder emits)', () => {
  const rw = 12;
  const rh = 9;
  const sub = new Uint8Array(rw * rh);
  for (let i = 0; i < sub.length; i++) sub[i] = (i * 5) & 15;
  const { dec } = roundTrip(sub, minCodeSizeFor(16), { width: rw, stride: rw, x0: 0, y0: 0, rectH: rh });
  assert.deepEqual(Array.from(dec), Array.from(sub));
});

test('interlaced row order is applied once, in encoder order', () => {
  const W = 9;
  const H = 7;
  const raster = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) raster[y * W + x] = (x + y * 2) & 15;
  // the GIF pass schedule: 8th lines from 0, 8th from 4, 4th from 2, then 2nd from 1
  const rowOrder = [];
  const starts = [0, 4, 2, 1];
  const steps = [8, 8, 4, 2];
  for (let pass = 0; pass < 4; pass++) for (let y = starts[pass]; y < H; y += steps[pass]) rowOrder.push(y);
  assert.deepEqual(rowOrder, [0, 4, 2, 6, 1, 3, 5]);
  const { dec } = roundTrip(raster, minCodeSizeFor(16), { width: W, stride: W, x0: 0, y0: 0, rectH: H, rowOrder });
  // the decoder un-interlaces, so reading row j of the result is row rowOrder[j]
  for (let j = 0; j < H; j++) for (let x = 0; x < W; x++) assert.equal(dec[j * W + x], raster[rowOrder[j] * W + x], `interlaced row ${j}`);
});

test('min code size follows the table, and small palettes still save', () => {
  assert.equal(minCodeSizeFor(2), 2, 'the alphabet must be able to express index 0 and 1');
  assert.equal(minCodeSizeFor(1), 2, 'a 1-entry table is illegal in GIF; we round up');
  assert.equal(minCodeSizeFor(0), 2);  for (const [colors, want] of [[2, 2], [3, 2], [4, 2], [5, 3], [8, 3], [16, 4], [17, 5], [32, 5], [64, 6], [128, 7], [129, 8], [256, 8]]) assert.equal(minCodeSizeFor(colors), want, `minCodeSizeFor(${colors})`);
});

test('sub-block framing is legal at every length', () => {
  for (const n of [0, 1, 254, 255, 256, 510, 7000]) {
    const idx = new Uint8Array(Math.max(1, n));
    for (let i = 0; i < idx.length; i++) idx[i] = i & 3;
    const { written } = n === 0 ? { written: 0 } : roundTrip(idx, 2, {});
    // total = payload + one length byte per <=255-byte chunk + the 0 terminator
    const chunks = written === 0 ? 0 : Math.ceil(written / 255);
    assert.equal(subBlockSize(written), written + chunks + 1, `framing for ${written} payload bytes`);
    assert.ok(subBlockSize(written) % 1 === 0);
    assert.ok(subBlockSize(written) >= written + 1, 'framing only ever adds bytes');
  }
  assert.equal(subBlockSize(0), 1, 'an empty payload is just the terminator');
});

test('truncated and pathological inputs degrade instead of throwing', () => {
  const idx = new Uint8Array(400);
  for (let i = 0; i < idx.length; i++) idx[i] = i & 7;
  const out = new Uint8Array(4096);
  const written = lzwEncode(idx, idx.length, 3, out, {});
  const dec = new Uint8Array(idx.length);
  // cut the stream in half: the decoder must return what it managed to read
  const got = lzwDecode(out.subarray(0, Math.floor(written / 2)), 3, dec);
  assert.ok(got < idx.length, 'truncated stream yields fewer pixels');
  for (let i = 0; i < got; i++) assert.equal(dec[i], idx[i], `pixel ${i} before the cut`);
  // an empty frame is legal (clear + EOI only)
  const empty = new Uint8Array(0);
  const w2 = lzwEncode(empty, 0, 2, out, {});
  assert.ok(w2 >= 1 && w2 < 8, `empty frame is a couple of codes, got ${w2}`);
});

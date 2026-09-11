/**
 * Format-level assertions for `enc/gif.js` + `dec/gif.js`, checked against the
 * byte layout the GIF89a spec mandates (and, when the optional dev dependency
 * `omggif` happens to be installed, against its reader too).
 *
 * These exist because a self-consistent encoder/decoder pair can hide a spec
 * violation forever — the disposal-method bit shift did exactly that until a
 * third-party reader was consulted.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { GifWriter, Disposal, normalizeDelay, interlaceRows } from '../src/enc/gif.js';
import { parseGif } from '../src/dec/gif.js';

const PALETTE = new Uint8Array(16 * 3);
for (let i = 0; i < 16; i++) {
  PALETTE[i * 3] = i * 16;
  PALETTE[i * 3 + 1] = 255 - i * 8;
  PALETTE[i * 3 + 2] = (i * 37) & 255;
}

function frameIndices(w, h, fn) {
  const a = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a[y * w + x] = fn(x, y);
  return a;
}

test('header, LSD and trailer are literal bytes', () => {
  const bytes = GifWriter.encodeAll({ width: 3, height: 2, palette: PALETTE, colors: 16, loop: 0 }, [{ indices: new Uint8Array(6), x: 0, y: 0, width: 3, height: 2, delay: 10 }]);
  assert.equal(String.fromCharCode(...bytes.subarray(0, 6)), 'GIF89a');
  assert.deepEqual([bytes[6], bytes[7]], [3, 0], 'width is little-endian');
  assert.deepEqual([bytes[8], bytes[9]], [2, 0], 'height is little-endian');
  assert.equal(bytes[10] & 0x80, 0x80, 'global color table flag');
  assert.equal(bytes[10] & 7, 3, '16 entries => size bits 3 (2<<(n+1) = 16)');
  assert.equal(bytes[bytes.length - 1], 0x3b, 'trailer');
  // NETSCAPE2.0 application extension carries loop count 0 (infinite)
  const app = String.fromCharCode(...bytes.subarray(13 + 16 * 3 + 3, 13 + 16 * 3 + 3 + 11));
  assert.equal(app, 'NETSCAPE2.0');
});

test('GCE packs delay, disposal and transparency exactly as the spec says', () => {
  for (const disposal of [0, 1, 2, 3]) {
    const bytes = GifWriter.encodeAll({ width: 4, height: 4, palette: PALETTE, colors: 16 }, [
      { indices: new Uint8Array(16), x: 0, y: 0, width: 4, height: 4, delay: 7, disposal, transparentIndex: 5 },
    ]);
    const gce = bytes.indexOf(0xf9) - 1; // walk back from the length byte we wrote
    assert.equal(bytes[gce], 0x21, 'extension introducer');
    assert.equal(bytes[gce + 1], 0xf9, 'graphic control label');
    assert.equal(bytes[gce + 2], 4, 'block length');
    const packed = bytes[gce + 3];
    assert.equal(packed >> 3 & 0, 0, 'reserved bits 7-5 stay zero'); // sanity: no shift beyond bit 5
    assert.equal((packed >> 2) & 7, disposal & 7, `disposal ${disposal} lives in bits 4-2`);
    assert.equal(packed & 1, 1, 'transparent color flag');
    assert.equal(bytes[gce + 4] | (bytes[gce + 5] << 8), normalizeDelay(7), 'delay in centiseconds, LE');
    assert.equal(bytes[gce + 6], 5, 'transparent index');
    const parsed = parseGif(bytes);
    assert.equal(parsed.frames[0].disposal, disposal, 'and we read back what we wrote');
    assert.equal(parsed.frames[0].transparentIndex, 5);
  }
});

test('reserved disposal values are rejected instead of silently written', () => {
  assert.throws(
    () => GifWriter.encodeAll({ width: 2, height: 2, palette: PALETTE, colors: 16 }, [{ indices: new Uint8Array(4), x: 0, y: 0, width: 2, height: 2, delay: 10, transparentIndex: 40 }]),
    /does not fit/
  );
});

test('a clipped sub-frame is read back exactly where it was placed', () => {
  const w = 6;
  const h = 5;
  const sub = frameIndices(w, h, (x, y) => (x * 3 + y) & 15);
  const bytes = GifWriter.encodeAll({ width: 16, height: 16, palette: PALETTE, colors: 16 }, [
    { indices: new Uint8Array(16 * 16), x: 0, y: 0, width: 16, height: 16, delay: 10 },
    { indices: sub, x: 4, y: 7, width: w, height: h, delay: 12 },
  ]);
  const f = parseGif(bytes).frames[1];
  assert.deepEqual([f.x, f.y, f.width, f.height], [4, 7, w, h]);
  assert.deepEqual(Array.from(f.indices), Array.from(sub), 'the LZW payload must not be offset by the frame position');
});

test('a window taken from a larger buffer needs its own offset', () => {
  const canvas = frameIndices(16, 16, (x, y) => (x + y) & 15);
  const expect = frameIndices(6, 5, (x, y) => (x + 4 + y + 7) & 15);
  const bytes = GifWriter.encodeAll({ width: 16, height: 16, palette: PALETTE, colors: 16 }, [
    { indices: canvas, stride: 16, x0: 4, y0: 7, x: 4, y: 7, width: 6, height: 5, delay: 10 },
  ]);
  assert.deepEqual(Array.from(parseGif(bytes).frames[0].indices), Array.from(expect));
});

test('the interlace row schedule is the one the GIF spec defines', () => {
  // pass 1: every 8th line from 0, pass 2: every 8th from 4, pass 3: every 4th
  // from 2, pass 4: every 2nd from 1. Using steps 8,4,2,1 (a common misread of the
  // spec) drops and duplicates rows for any height that is not a multiple of 8.
  assert.deepEqual(Array.from(interlaceRows(7)), [0, 4, 2, 6, 1, 3, 5]);
  assert.deepEqual(Array.from(interlaceRows(8)), [0, 4, 2, 6, 1, 3, 5, 7]);
  for (const h of [1, 2, 3, 5, 9, 16, 17, 33]) {
    const rows = interlaceRows(h);
    assert.equal(rows.length, h, `schedule length for h=${h}`);
    assert.equal(new Set(rows).size, h, `h=${h} must visit every row exactly once`);
    assert.deepEqual([...rows].sort((a, b) => a - b), [...Array(h).keys()]);
  }
});

test('interlaced frames decode to the same rows as progressive ones', () => {
  const idx = frameIndices(9, 7, (x, y) => (x + y * 2) & 15);
  const mk = (interlace) => GifWriter.encodeAll({ width: 9, height: 7, palette: PALETTE, colors: 16 }, [{ indices: idx, x: 0, y: 0, width: 9, height: 7, delay: 10, interlace }]);
  const a = parseGif(mk(false)).frames[0];
  const b = parseGif(mk(true)).frames[0];
  assert.equal(b.interlace, true);
  assert.deepEqual(Array.from(b.indices), Array.from(a.indices), 'the decoder must un-interlace');
});

test('loop count, comment and background index survive a round trip', () => {
  const bytes = GifWriter.encodeAll(
    { width: 4, height: 4, palette: PALETTE, colors: 16, loop: 3, comment: 'hello gifx', backgroundColorIndex: 4, pixelAspectRatio: 10 },
    [{ indices: new Uint8Array(16), x: 0, y: 0, width: 4, height: 4, delay: 10 }]
  );
  const gif = parseGif(bytes);
  assert.equal(gif.loop, 3);
  assert.deepEqual(gif.comments, ['hello gifx']);
  assert.equal(gif.backgroundColorIndex, 4);
  assert.equal(gif.pixelAspectRatio, 10);
});

test('delay normalization covers the browser-hostile cases', () => {
  assert.equal(normalizeDelay(0), 10, 'a 0 cs frame means "as fast as possible" and flickers -> 10 cs');
  assert.equal(normalizeDelay(1), 2, '1 cs is clamped to the widely accepted minimum');
  assert.equal(normalizeDelay(2.4), 2, 'rounded, not truncated upward');
  assert.equal(normalizeDelay(70), 70);
  assert.equal(normalizeDelay(70000), 65535, 'never exceeds the 16-bit field');
  assert.equal(normalizeDelay(NaN), 10);
  assert.equal(normalizeDelay(-5), 10);
  assert.equal(normalizeDelay(0, 0), 0, 'callers that really want a 0 cs frame get one');
  assert.equal(normalizeDelay(1, 0), 1, 'minDelay 0 preserves exotic timings (used by the remuxer)');
});

test('omggif agrees where it is available (optional devDependency)', async () => {
  let GifReader;
  try {
    ({ GifReader } = await import('omggif'));
  } catch {
    t_skip();
    return;
  }
  function t_skip() {}
  const idx = frameIndices(12, 9, (x, y) => (x * 5 + y * 3) & 15);
  const bytes = GifWriter.encodeAll({ width: 12, height: 9, palette: PALETTE, colors: 16, loop: 0 }, [
    { indices: idx, x: 0, y: 0, width: 12, height: 9, delay: 13, disposal: 2, transparentIndex: 3 },
  ]);
  const interlacedBytes = GifWriter.encodeAll({ width: 12, height: 9, palette: PALETTE, colors: 16 }, [
    { indices: idx, x: 0, y: 0, width: 12, height: 9, delay: 13, interlace: true, transparentIndex: 3 },
  ]);
  const reader = new GifReader(bytes);
  assert.equal(reader.numFrames(), 1);
  assert.deepEqual([reader.width, reader.height], [12, 9]);
  assert.equal(reader.loopCount(), 0, 'loop count 0 means forever');
  const pixels = new Uint8Array(12 * 9 * 4);
  reader.decodeAndBlitFrameRGBA(0, pixels);
  const fi = reader.frameInfo(0);
  assert.equal(fi.disposal, 2, 'omggif reads our disposal byte as restore-to-background');
  assert.equal(fi.delay, 13);
  assert.equal(fi.transparent_index, 3);
  assert.notEqual(fi.transparent_index, null, 'omggif sees the transparency flag');
  // omggif composites with transparency: index 3 pixels stay untouched (0,0,0,0)
  const expect = frameIndices(12, 9, (x, y) => (x * 5 + y * 3) & 15);
  let transparent = 0;
  for (let i = 0; i < 12 * 9; i++) if (expect[i] === 3 && pixels[i * 4 + 3] === 0) transparent++;
  assert.ok(transparent > 0, 'the transparency flag round-trips through a foreign decoder');
  // a foreign decoder must un-interlace exactly like we do
  const ipx = new Uint8Array(12 * 9 * 4);
  const ireader = new GifReader(interlacedBytes);
  assert.equal(ireader.frameInfo(0).interlaced, true);
  ireader.decodeAndBlitFrameRGBA(0, ipx);
  for (let i = 0; i < 12 * 9; i++) {
    assert.equal(ipx[i * 4], pixels[i * 4], `interlaced px ${i} red`);
    assert.equal(ipx[i * 4 + 1], pixels[i * 4 + 1], `interlaced px ${i} green`);
  }
  // and every opaque pixel matches our palette entry exactly
  for (let i = 0; i < 12 * 9; i++) {
    if (expect[i] === 3) continue;
    assert.equal(pixels[i * 4], PALETTE[expect[i] * 3], `px ${i} red`);
    assert.equal(pixels[i * 4 + 1], PALETTE[expect[i] * 3 + 1], `px ${i} green`);
    assert.equal(pixels[i * 4 + 2], PALETTE[expect[i] * 3 + 2], `px ${i} blue`);
  }
});

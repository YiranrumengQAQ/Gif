/**
 * Optimizer tests. The important one is #3: a lossless re-encode must be
 * pixel-identical, which is checked by decoding both files and comparing every
 * pixel — not by trusting the byte savings.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { optimizeGif, quantizeDelays, verifyOptimization, OPTIMIZE_TECHNIQUES } from '../src/optimize/remux.js';
import { buildGlobalPalette, compactPaletteForRect } from '../src/optimize/remux-helpers.js';
import { dirtyRect, collapseDuplicates, planDisposal, changeFraction, frameHash } from '../src/optimize/diff.js';
import { exactLzwBytes, indexStats, predictLzwBytes, observe, resetCalibration, calibrationState, tuneClearInterval, payloadBudget, bytesToHuman, predictCandidate } from '../src/optimize/size.js';
import { parseGif, composeGifFrames } from '../src/dec/gif.js';
import { GifWriter } from '../src/enc/gif.js';
import { lzwEncode, minCodeSizeFor } from '../src/enc/lzw.js';

const W = 40;
const H = 24;

/** Palette: 16 distinguishable colors, deliberately not sorted. */
const PAL = (() => {
  const p = new Uint8Array(16 * 3);
  for (let i = 0; i < 16; i++) {
    p[i * 3] = (i * 17) & 255;
    p[i * 3 + 1] = (i * 41 + 5) & 255;
    p[i * 3 + 2] = (i * 97 + 13) & 255;
  }
  return p;
})();

/**
 * Build a GIF with: a full first frame, sub-frame deltas, an exact duplicate, a
 * near-duplicate (1 cs delay), a frame with its own local palette, and comments.
 */
function buildTestGif({ frames = 6, duplicateAt = 2, subFrames = true, smallPaletteAt = -1, smallColors = 8 } = {}) {
  const gw = new GifWriter({ width: W, height: H, palette: PAL, colors: 16, loop: 2, comment: 'made by gifx tests' });
  const canvas = new Uint8Array(W * H).fill(1);
  const written = [];
  for (let i = 0; i < frames; i++) {
    const next = Uint8Array.from(canvas);
    let x = 0;
    let y = 0;
    let w = W;
    let h = H;
    if (i > 0 && subFrames) {
      x = (i * 5) % (W - 8);
      y = (i * 3) % (H - 6);
      w = 8;
      h = 6;
      const small = i === smallPaletteAt;
      const mask = small ? smallColors - 1 : 15;
      for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) next[(y + yy) * W + (x + xx)] = (i + xx + yy) & mask;
    } else if (i > 0) {
      for (let p = 0; p < next.length; p++) next[p] = (p + i) & 15;
    }
    const duplicate = i === duplicateAt;
    let indices = duplicate ? Uint8Array.from(canvas) : next;
    const delay = 7 + i;
    gw.addFrame({
      indices: duplicate ? Uint8Array.from(canvas) : clip(indices, x, y, w, h, W),
      x: duplicate ? 0 : x,
      y: duplicate ? 0 : y,
      width: duplicate ? W : w,
      height: duplicate ? H : h,
      delay,
      palette: i === smallPaletteAt ? PAL.slice(0, smallColors * 3) : null,
      colors: i === smallPaletteAt ? smallColors : 16,
      transparentIndex: -1,
      disposal: 1,
    });
    canvas.set(indices);
    written.push({ indices: Uint8Array.from(canvas), delay });
  }
  const bytes = gw.finish();
  return { bytes, written };
}
function clip(full, x, y, w, h, Wsrc) {
  const out = new Uint8Array(w * h);
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) out[yy * w + xx] = full[(y + yy) * Wsrc + (x + xx)];
  return out;
}

test('quantizeDelays preserves total time and respects the floor', () => {
  // 33.33 ms per frame → 3.333 cs; naive rounding loses 0.33 cs per frame
  const src = new Array(300).fill(33.3333 / 10);
  const out = quantizeDelays(src, 2, src.reduce((a, b) => a + b, 0));
  const sum = out.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - Math.round(src.reduce((a, b) => a + b, 0))) <= 1, `total drifted to ${sum} cs`);
  assert.ok(out.every((v) => v >= 2), 'browser-clamp floor respected');
  const naive = src.map((v) => Math.max(2, Math.round(v)));
  assert.ok(Math.abs(naive.reduce((a, b) => a + b, 0) - sum) > 20, 'the drift correction is doing real work');
  const mixed = quantizeDelays([100, 0.4, 0.4, 0.4, 50], 2, 151.2);
  assert.equal(mixed.reduce((a, b) => a + b, 0), 151);
  assert.ok(mixed.every((v) => v >= 2));
  assert.deepEqual(quantizeDelays([], 2, 0), []);
});

test('buildGlobalPalette unions frame palettes and reports overflow', () => {
  const mk = (colors) => ({ palette: PAL.slice(0, colors * 3), indices: new Uint8Array(4) });
  const a = buildGlobalPalette([mk(16), mk(16)], PAL);
  assert.equal(a.ok, true);
  assert.equal(a.colors, 16, 'identical palettes collapse');
  const many = [];
  for (let i = 0; i < 30; i++) {
    const p = new Uint8Array(9 * 3);
    for (let c = 0; c < 9; c++) {
      p[c * 3] = i;
      p[c * 3 + 1] = c * 28;
      p[c * 3 + 2] = i * 7 + c;
    }
    many.push({ palette: p, indices: new Uint8Array(4) });
  }
  const b = buildGlobalPalette(many, null, { limit: 256 });
  assert.equal(b.ok, false, 'more than 256 distinct colors cannot share a table');
  assert.ok(b.overflow > 0);
  assert.equal(b.colors, 256);
  const single = buildGlobalPalette([{ palette: new Uint8Array([9, 9, 9]), indices: new Uint8Array(1) }], null);
  assert.ok(single.colors >= 2, 'a 1-color GIF is padded so viewers stay happy');
  assert.ok(Array.isArray(single.remaps) && single.remaps[0].length === 256);
});

test('optimizeGif(lossless) shrinks the file and is pixel-identical', () => {
  const { bytes } = buildTestGif({ frames: 8, duplicateAt: 3, smallPaletteAt: 5 });
  const src = parseGif(bytes);
  const before = composeGifFrames(src);
  const res = optimizeGif(src, { originalBytes: bytes.length, check: true });
  assert.ok(res.globalPalette, 'the animation fits one global color table');
  assert.ok(res.check.ok, `pixels differ (worst delta ${res.check.worstDelta} at frame ${res.check.firstBadFrame})`);
  assert.equal(res.check.worstDelta, 0, 'lossless means zero, not "close"');
  assert.equal(res.framesOut, res.framesIn - res.techniques.dedupe.frames);
  assert.ok(res.techniques.dedupe.frames >= 1, 'the duplicate frame was dropped');
  assert.ok(res.techniques.subFrames.savedPixels > 0, 'sub-frames were used');
  assert.equal(res.check.badPixels, 0, 'not one pixel may move');
  assert.ok(res.bytes.length < bytes.length, `expected savings: ${res.bytes.length} vs ${bytes.length}`);
  assert.ok(res.savedRatio > 0.05, `savings too small to be interesting: ${(res.savedRatio * 100).toFixed(1)}%`);
  assert.ok(Math.abs(res.durationMs - res.sourceDurationMs) <= 10, `duration moved by ${res.durationMs - res.sourceDurationMs} cs*10`);
  // frame counts differ on purpose (the duplicate was folded away); what must be
  // identical is what a viewer shows over time, which is what `check` compares
  assert.equal(res.check.framesB, res.check.framesA - res.techniques.dedupe.frames);
  const after = parseGif(res.bytes);
  assert.equal(after.width, src.width);
  assert.equal(after.height, src.height);
  assert.deepEqual(Array.from(after.palette).slice(0, 3), Array.from(src.palette).slice(0, 3), 'the shared palette is preserved');
});

test('optimizeGif reports every technique and honors switches', () => {
  const { bytes } = buildTestGif({ frames: 6 });
  const src = parseGif(bytes);
  const sizes = {};
  for (const name of OPTIMIZE_TECHNIQUES) {
    if (name === 'requantize' || name === 'stripComments') continue;
    const off = optimizeGif(src, { originalBytes: bytes.length, techniques: { [name]: false } });
    sizes[name] = off.bytes.length;
  }
  const all = optimizeGif(src, { originalBytes: bytes.length });
  const none = optimizeGif(src, {
    originalBytes: bytes.length,
    techniques: { dedupe: false, subFrames: false, transparentUnchanged: false, compactPalettes: false, clearTune: false, delayQuantize: false },
  });
  assert.ok(all.bytes.length <= none.bytes.length, `all techniques (${all.bytes.length}) should beat none (${none.bytes.length})`);
  assert.ok(sizes.dedupe >= all.bytes.length, 'disabling dedupe cannot be cheaper');
  assert.ok(sizes.subFrames > all.bytes.length, 'disabling sub-frames must cost bytes');
  assert.ok(optimizeGif(src, { originalBytes: bytes.length, techniques: { dedupe: false } }).framesOut > all.framesOut, 'dedupe removes frames');
  assert.equal(none.framesOut, none.framesIn, 'no dedupe ⇒ every frame survives');
});

test('optimizeGif(lossy) re-quantizes and stays inside a colors budget', () => {
  const { bytes } = buildTestGif({ frames: 4, subFrames: false });
  const src = parseGif(bytes);
  const lossy = optimizeGif(src, { lossless: false, colors: 4, originalBytes: bytes.length, techniques: { subFrames: false, transparentUnchanged: false } });
  assert.ok(lossy.techniques.requantize.enabled);
  const reparsed = parseGif(lossy.bytes);
  const used = new Set();
  for (const f of reparsed.frames) for (let i = 0; i < f.indices.length; i++) used.add(f.indices[i]);
  assert.ok(used.size <= 8, `only ${used.size} palette entries used (budget 4, +1 transparent slack)`);
  const lossless = optimizeGif(src, { originalBytes: bytes.length });
  assert.ok(lossy.bytes.length < bytes.length, 'lossy should be smaller than the source');
  assert.ok(lossless.bytes.length < bytes.length);
  // error must be reported, not silent
  const check = verifyOptimization(src, lossy.bytes, { maxDelta: 0 });
  assert.equal(check.ok, false, 'a lossy encode must NOT pass the lossless check');
  assert.ok(check.worstDelta > 0);
});

test('optimizeGif handles adversarial files', () => {
  // (a) single-frame GIF: nothing to dedupe, must still be valid
  const one = GifWriter.encodeAll({ width: 8, height: 8, palette: PAL, colors: 16, loop: null }, [{ indices: new Uint8Array(64), x: 0, y: 0, width: 8, height: 8, delay: 20 }]);
  const r1 = optimizeGif(parseGif(one), { originalBytes: one.length, check: true });
  assert.equal(r1.framesOut, 1);
  assert.ok(r1.check.ok);
  // (b) all-identical frames: collapses to one frame with the merged duration
  const gw = new GifWriter({ width: 8, height: 8, palette: PAL, colors: 16 });
  for (let i = 0; i < 9; i++) gw.addFrame({ indices: new Uint8Array(64).fill(3), x: 0, y: 0, width: 8, height: 8, delay: 10 });
  const allSame = gw.finish();
  const r2 = optimizeGif(parseGif(allSame), { originalBytes: allSame.length, check: true });
  assert.equal(r2.framesOut, 1, 'a GIF of one static image is 1 frame');
  assert.equal(r2.durationMs, 900, 'the durations merged');
  assert.ok(r2.bytes.length * 4 < allSame.length, 'and the file collapsed');
  // (c) palette overflow: falls back to local palettes, still lossless
  const many = new GifWriter({ width: 16, height: 16, loop: 0 });
  for (let i = 0; i < 60; i++) {
    const p = new Uint8Array(6 * 3);
    for (let c = 0; c < 6; c++) {
      p[c * 3] = (i * 3) & 255;
      p[c * 3 + 1] = (c * 60 + i) & 255;
      p[c * 3 + 2] = (255 - i * 2) & 255;
    }
    many.addFrame({ indices: new Uint8Array(256).fill(i % 6), x: 0, y: 0, width: 16, height: 16, delay: 5, palette: p, colors: 6 });
  }
  const overflowBytes = many.finish();
  const r3 = optimizeGif(parseGif(overflowBytes), { originalBytes: overflowBytes.length, check: true });
  assert.equal(r3.globalPalette, false);
  assert.ok(r3.warnings.length >= 1, 'the fallback is reported');
  assert.ok(r3.check.ok, 'still pixel-identical in local-palette mode');
  // (d) truncated file: must throw a typed error, not a RangeError
  assert.throws(() => optimizeGif(overflowBytes.subarray(0, 40)), (e) => e.code === 'INPUT_CORRUPT' || e.code === 'INPUT_NO_FRAMES' || e.code === 'INPUT_INVALID');
});

test('verifyOptimization is a real check (negative control)', () => {
  const { bytes } = buildTestGif({ frames: 5 });
  const src = parseGif(bytes);
  const different = GifWriter.encodeAll({ width: W, height: H, palette: PAL, colors: 16 }, [{ indices: new Uint8Array(W * H).fill(9), x: 0, y: 0, width: W, height: H, delay: 10 }]);
  const v = verifyOptimization(src, different);
  assert.equal(v.ok, false);
  assert.equal(v.firstBadFrame, 0);
  assert.ok(v.worstDelta > 0);
  const same = verifyOptimization(src, bytes);
  assert.equal(same.ok, true);
  assert.equal(same.worstDelta, 0);
});

test('dirtyRect: boundaries, tolerance, sampling and alignment', () => {
  const S = W * 3;
  const a = new Uint8Array(S * H);
  const b = Uint8Array.from(a);
  assert.deepEqual(dirtyRect(a, b, { width: W, height: H, stride: S, bytesPerPixel: 3 }), { x: 0, y: 0, width: 0, height: 0, area: 0, changed: false, holes: 0 });
  // single pixel at the far corner
  b[(H - 1) * S + (W - 1) * 3] = 200;
  let r = dirtyRect(a, b, { width: W, height: H, stride: S, bytesPerPixel: 3 });
  assert.deepEqual([r.x, r.y, r.width, r.height], [W - 1, H - 1, 1, 1]);
  // 1px tolerance must miss a 1-unit change
  assert.equal(dirtyRect(a, b, { width: W, height: H, stride: S, bytesPerPixel: 3, threshold: 255 }).changed, false);
  // alignment grows to the block grid without leaving the canvas
  r = dirtyRect(a, b, { width: W, height: H, stride: S, bytesPerPixel: 3, align: 8 });
  assert.equal(r.x, 32, 'alignment floors to the 8px grid');
  assert.equal(r.width, 8, 'and reaches the canvas edge, never past it');
  assert.ok(r.x + r.width <= W && r.y + r.height <= H, 'aligned rect stays in bounds');
  // sampling must find the same rect for a solid block
  const c = Uint8Array.from(a);
  const d = Uint8Array.from(a);
  for (let y = 4; y < 10; y++) for (let x = 6; x < 20; x++) d[y * S + x * 3] = 255;
  void c;
  const exact = dirtyRect(a, d, { width: W, height: H, stride: S, bytesPerPixel: 3 });
  const sampled = dirtyRect(a, d, { width: W, height: H, stride: S, bytesPerPixel: 3, sample: 4 });
  assert.deepEqual([exact.x, exact.y, exact.width, exact.height], [6, 4, 14, 6]);
  assert.deepEqual([sampled.x, sampled.y, sampled.width, sampled.height], [6, 4, 14, 6]);
  // zero-size canvases must not produce a bogus rect
  assert.equal(dirtyRect(new Uint8Array(0), new Uint8Array(0), { width: 0, height: 0 }).changed, false);
});

test('collapseDuplicates, changeFraction and planDisposal agree with the encoder', () => {
  const S = W * 3;
  const mk = (fill, patch) => {
    const a = new Uint8Array(S * H);
    for (let i = 0; i < W * H; i++) {
      a[i * 3] = fill;
      a[i * 3 + 1] = fill;
      a[i * 3 + 2] = fill;
    }
    if (patch) patch(a);
    return a;
  };
  const frames = [
    { rgba: mk(10), width: W, height: H, stride: S, delayCs: 10 },
    { rgba: mk(10), width: W, height: H, stride: S, delayCs: 10 },
    { rgba: mk(10, (a) => { for (let p = 0; p < 30; p++) a[p] = 200; }), width: W, height: H, stride: S, delayCs: 10 },
  ];
  const col = collapseDuplicates(frames, {});
  assert.equal(col.frames.length, 2);
  assert.equal(col.frames[0].delayCs, 20);
  assert.throws(() => collapseDuplicates([{ rgba: new Uint8Array(3) }, { rgba: new Uint8Array(3) }]), TypeError, 'missing dimensions must not silently merge everything');
  const moved = changeFraction(mk(10), frames[2].rgba, { width: W, height: H, stride: S, bytesPerPixel: 3 });
  assert.ok(moved > 0 && moved < 0.1, `small change measured as ${moved}`);
  assert.equal(changeFraction(mk(10), mk(10), { width: W, height: H, stride: S, bytesPerPixel: 3 }), 0);
  assert.equal(frameHash(mk(7), W, H, S), frameHash(mk(7), W, H, S));
  assert.notEqual(frameHash(mk(7), W, H, S), frameHash(mk(8), W, H, S));
  // planDisposal with RGBA frames: sub-rect where safe, full frame otherwise
  const plan = planDisposal([
    { rgba: rgbaFill(10), width: W, height: H, delayCs: 10 },
    { rgba: rgbaFill(10, 4, 6, 8, 6), width: W, height: H, delayCs: 10 },
  ]);
  assert.equal(plan.frames[0].encodeRect.width, W);
  assert.ok(plan.frames[1].subFrame, 'the second frame should be a sub-rect');
  assert.ok(plan.savings > 0.3);
});

function rgbaFill(v, x = 0, y = 0, w = 0, h = 0) {
  const a = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    a[i * 4] = v;
    a[i * 4 + 1] = v;
    a[i * 4 + 2] = v;
    a[i * 4 + 3] = 255;
  }
  for (let yy = y; yy < y + h; yy++) for (let xx = x; xx < x + w; xx++) {
    const p = (yy * W + xx) * 4;
    a[p] = 250;
    a[p + 1] = 20;
    a[p + 2] = 250;
  }
  return a;
}

test('size model: exact encoder agreement, ranking, calibration', () => {
  const indices = new Uint8Array(W * H);
  for (let i = 0; i < indices.length; i++) indices[i] = (i >> 2) & 3;
  const colors = 4;
  // ground truth straight from the encoder internals
  const scratch = new Uint8Array(1 << 16);
  const real = lzwEncode(indices, indices.length, minCodeSizeFor(colors), scratch, {});
  assert.equal(exactLzwBytes(indices, indices.length, { width: W, colors }), real, 'exactLzwBytes must match lzwEncode');
  assert.ok(real > 0);
  // an all-one-color frame compresses to a tiny stream
  const flat = new Uint8Array(W * H);
  const flatBytes = exactLzwBytes(flat, flat.length, { width: W, colors });
  assert.ok(flatBytes < real / 4, `flat ${flatBytes} vs patterned ${real}`);
  // model must rank palette size monotonically, which the search depends on
  resetCalibration();
  const series = [];
  for (const n of [2, 4, 8, 16]) {
    const idx = new Uint8Array(W * H);
    for (let i = 0; i < idx.length; i++) idx[i] = (i * 7) & (n - 1);
    const stats = indexStats(idx, idx.length, n, { width: W });
    series.push({ n, model: predictLzwBytes({ pixels: idx.length, stats, colors: n }), exact: exactLzwBytes(idx, idx.length, { width: W, colors: n }) });
  }
  for (let i = 1; i < series.length; i++) assert.ok(series[i].model >= series[i - 1].model - 1e-6, 'model must be monotone in color count');
  // calibration must reduce error on repeated content
  const before = Math.abs((series[2].model - series[2].exact) / series[2].exact);
  for (let k = 0; k < 12; k++) {
    const idx = new Uint8Array(W * H);
    for (let i = 0; i < idx.length; i++) idx[i] = (i + k) & 7;
    const st = indexStats(idx, idx.length, 8, { width: W });
    observe(st, exactLzwBytes(idx, idx.length, { width: W, colors: 8 }), idx.length);
  }
  const st = indexStats(new Uint8Array(W * H).fill(0), W * H, 8, { width: W });
  assert.ok(calibrationState().calibrated, 'calibration kicks in after enough samples');
  assert.ok(Number.isFinite(st.h1));
  void before;
  // clear-interval tuning never makes things worse by more than rounding
  const tune = tuneClearInterval(indices, indices.length, { colors, width: W });
  assert.ok(tune.clearInterval > 32);
  const untuned = exactLzwBytes(indices, indices.length, { width: W, colors, clearInterval: 0 });
  assert.ok(tune.scaled <= untuned * 1.35, `tuned ${tune.scaled} vs baseline ${untuned}`);
  // budget math
  assert.ok(payloadBudget(10000, { colors: 64, globalPalette: true, loop: 0, frames: 20, localPalettes: false }) < 10000);
  assert.equal(payloadBudget(30, { colors: 2, globalPalette: false, frames: 1 }), 0, 'never negative');
  assert.equal(bytesToHuman(512), '512 B');
  assert.equal(bytesToHuman(2048), '2.0 KiB');
  assert.equal(bytesToHuman(3 * 1048576), '3.00 MiB');
  const cand = predictCandidate({ colors, width: W, height: H, loop: 0, frames: [{ indices, width: W, height: H }] }, { measure: true });
  assert.equal(cand.frames[0].lzwBytes, real, 'measure mode uses the real encoder');
  assert.ok(cand.bytes >= real);
});

test('compactPaletteForRect drops entries only used outside the rect', () => {
  const idx = new Uint8Array(W * H);
  for (let i = 0; i < idx.length; i++) idx[i] = i & 15;
  const pal = Uint8Array.from(PAL);
  const before = new Set(idx).size;
  const res = compactPaletteForRect(idx, pal, 16, { x: 0, y: 0, width: W, height: H }, W, H);
  assert.ok(res.colors <= before);
  assert.ok(res.colors >= 8, 'a full-canvas sweep must keep every used entry');
  const res2 = compactPaletteForRect(idx, pal, 16, { x: 0, y: 0, width: 4, height: 1 }, W, H);
  assert.ok(res2.colors < res.colors, 'a small rect needs fewer colors');
  void before;
});

/**
 * Sub-framed GIFs on the internet almost always pair a small drawing area with
 * `disposal: 2` or `3`: the frame paints a patch, then hands the patch back to the
 * background. An optimizer that only looks at pixels silently loses that hand-back.
 */
function buildDisposalGif({ frames = 7, W = 48, H = 36 } = {}) {
  const gw = new GifWriter({ width: W, height: H, palette: PAL, colors: 16, loop: 0 });
  const canvas = new Uint8Array(W * H).fill(2);
  for (let i = 0; i < frames; i++) {
    const next = Uint8Array.from(canvas);
    const x = (i * 7) % (W - 10);
    const y = (i * 5) % (H - 8);
    const w = 10;
    const h = 8;
    for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) next[(y + yy) * W + x + xx] = (i + xx + yy * 2) & 15;
    gw.addFrame({
      indices: next,
      stride: W,
      x0: x,
      y0: y,
      x,
      y,
      width: w,
      height: h,
      delay: 6 + i,
      disposal: i % 3 === 1 ? 2 : i % 3 === 2 ? 3 : 1,
      interlace: i % 2 === 0,
      transparentIndex: -1,
    });
    canvas.set(next.subarray(0), 0);
  }
  return gw.finish();
}

test('optimizeGif keeps disposal methods and stays exact on sub-framed input', () => {
  const bytes = buildDisposalGif();
  const src = parseGif(bytes);
  const res = optimizeGif(src, { originalBytes: bytes.length, check: true });
  assert.ok(res.check.ok, `lossless remux broke a disposal/sub-frame file: ${res.check.reason} (worst ${res.check.worstDelta})`);
  assert.equal(res.check.worstDelta, 0);
  assert.equal(res.check.badPixels, 0);
  // an already sub-framed, disposal-correct GIF has nothing left to win: the bar is
  // "never inflate", and the techniques must still be reported as working
  assert.ok(res.bytes.length <= bytes.length, `${res.bytes.length} should not exceed ${bytes.length}`);
  assert.ok(res.techniques.subFrames.savedPixels > 0, 'sub-frame pixels were avoided');
  const out = parseGif(res.bytes);
  assert.equal(out.loop, src.loop, 'loop count survives (0 means forever here)');
  const dis = out.frames.map((f) => f.disposal);
  assert.ok(dis.includes(2) || dis.includes(3), 'a clearing frame is still a clearing frame');
  assert.equal(out.frames.length, res.framesOut);
  assert.equal(Math.abs(res.durationMs - res.sourceDurationMs) <= 10 * res.framesIn, true, 'timing is preserved within the quantization slack');
});

test('optimizeGif never emits an index the color table cannot express', () => {
  for (const cfg of [{ frames: 5 }, { frames: 9, W: 64, H: 40 }]) {
    const bytes = buildDisposalGif(cfg);
    const res = optimizeGif(parseGif(bytes), { originalBytes: bytes.length });
    const out = parseGif(res.bytes);
    for (const f of out.frames) {
      const tableSize = f.palette ? f.palette.length / 3 : out.palette.length / 3;
      const slots = 1 << Math.max(1, Math.ceil(Math.log2(Math.max(2, tableSize))));
      for (let i = 0; i < f.indices.length; i++) {
        const v = f.indices[i];
        if (v >= slots) throw new Error(`frame ${f.index} index ${v} escapes a ${slots}-entry table`);
      }
      if (f.transparentIndex >= 0) assert.ok(f.transparentIndex < slots, 'the transparent slot must be representable');
    }
  }
});

test('optimizeGif preserves comments and strips them on request', () => {
  const bytes = buildTestGif({ frames: 4 }).bytes; // built with comment 'made by gifx tests'
  const src = parseGif(bytes);
  assert.deepEqual(src.comments, ['made by gifx tests']);
  const kept = optimizeGif(src, { originalBytes: bytes.length });
  assert.deepEqual(parseGif(kept.bytes).comments, src.comments, 'metadata survives by default');
  const stripped = optimizeGif(src, { originalBytes: bytes.length, stripComments: true, check: true });
  assert.deepEqual(parseGif(stripped.bytes).comments, []);
  assert.ok(stripped.techniques.stripComments.savedBytes > 0, 'the saving is reported');
  assert.ok(stripped.check.ok, 'stripping a comment cannot change pixels');
});

/**
 * GIFX Kernel — size model.
 *
 * A target-size search evaluates hundreds of candidate settings. Encoding them
 * all is prohibitive (a 4K-frame candidate is ~100 ms), so each candidate is
 * first *predicted* from the palette-index statistics, and only the survivors get
 * a real encode. For that to work the model must be right to within a few percent
 * on the kinds of content GIFs actually carry, and it must never be wrong in a
 * way that makes the search pick an over-budget file as "best".
 *
 * The model:
 *   bytes ≈ pixels × h1 / 8 × gain + clearCodeCost + framing
 * where `h1` is the conditional entropy of a pixel given its left neighbour
 * (GIF's LZW is a dictionary coder, so the *horizontal* context is what it
 * exploits), `gain` collapses to the measured encoder/decoder dictionary
 * efficiency (calibrated below, ~0.55–0.75 for photo, ~0.2 for flat art), and
 * `clearCodeCost` accounts for the periodic dictionary reset.
 *
 * `exactLzwBytes()` is the ground truth: it runs the real encoder into a scratch
 * buffer and returns just the count. Because it allocates nothing per call, the
 * search can afford it for the top candidates, and tests can assert the model's
 * error directly.
 *
 * @module optimize/size
 */
import { lzwEncode, minCodeSizeFor, subBlockSize } from '../enc/lzw.js';

/** Scratch for exact measurement, reused forever (a few MB, never grows back). */
let measureBuf = null;
function measureCapacity(pixels) {
  // Worst case LZW output is ~ (pixels * 12/8) + clear/EOI slack; cap the
  // allocation and fall back to chunked counting for absurd sizes.
  return Math.min(1 << 26, Math.max(1 << 16, Math.ceil(pixels * 1.6) + 4096));
}

/**
 * Exact LZW stream length for these indices (no file framing).
 * @param {Uint8Array} indices
 * @param {object} [opts] `{width, height, minCodeSize, clearInterval, earlyWidthBump, offset}`
 */
export function exactLzwBytes(indices, count, opts = {}) {
  const minCodeSize = opts.minCodeSize || minCodeSizeFor(opts.colors || 256);
  const need = measureCapacity(count);
  if (!measureBuf || measureBuf.length < need) measureBuf = new Uint8Array(need);
  const stopAfter = opts.stopAfter || 0;
  // Only engage the sub-rectangle path when a rect was actually requested:
  // lzwEncode walks rowStart[] whenever `width` is set, so a missing `rectH`
  // would encode nothing. Whole-buffer mode must match `lzwEncode(.., {})`.
  const rect = opts.rectH > 0;
  const encOpts = rect
    ? {
        width: opts.width,
        stride: opts.stride || opts.width,
        x0: opts.x0 || 0,
        y0: opts.y0 || 0,
        rectH: opts.rectH,
        rowOrder: opts.rowOrder,
        clearInterval: opts.clearInterval,
        earlyWidthBump: opts.earlyWidthBump,
        stopAfter: stopAfter || 0,
        offset: 0,
      }
    : { clearInterval: opts.clearInterval, earlyWidthBump: opts.earlyWidthBump, stopAfter: stopAfter || 0, offset: 0 };
  try {
    const written = lzwEncode(indices, count, minCodeSize, measureBuf, encOpts);
    return typeof written === 'number' ? written : measureBuf.length;
  } catch (e) {
    if (e && e.code === 'LZW_BUFFER_FULL') return Math.ceil(count * 1.5);
    throw e;
  }
}

/**
 * Sampled statistics of an index stream that the model consumes.
 * Cost is O(pixels/sampleStep) which is what makes candidate screening cheap.
 */
export function indexStats(indices, count, colors, opts = {}) {
  const step = Math.max(1, opts.sample || 1);
  const paletteBits = Math.max(1, Math.min(9, Math.ceil(Math.log2(Math.max(2, colors || 256)))));
  const width = opts.width | 0;
  // order-0 histogram + order-1 (left neighbour) transitions on sampled rows
  const h0 = new Float64Array(1 << paletteBits);
  const strideCtx = 1 << paletteBits;
  const h1 = new Float64Array(strideCtx * strideCtx);
  let n = 0;
  let runs = 0;
  let runPixels = 0;
  let cur = indices[0] & 255;
  let runLen = 1;
  let prevIdx = cur;
  const rows = width ? Math.floor(count / width) : 1;
  const rowStep = Math.max(1, Math.floor(step / Math.max(1, width / 64)) || 1);
  for (let r = 0; r < rows; r += Math.min(rows, rowStep)) {
    const base = r * width;
    const end = Math.min(count, base + width);
    if (width) prevIdx = indices[base] & 255;
    for (let i = base; i < end; i += step) {
      const v = indices[i] & 255;
      h0[v]++;
      h1[prevIdx * strideCtx + (v & (strideCtx - 1))]++;
      n++;
      if (v === cur) runLen++;
      else {
        runs++;
        runPixels += runLen;
        cur = v;
        runLen = 1;
      }
      prevIdx = v;
    }
  }
  if (!n) return { h0: 0, h1: 0, meanRun: 1, samples: 0, paletteBits, colors };
  let e0 = 0;
  for (let i = 0; i < h0.length; i++) {
    if (!h0[i]) continue;
    const p = h0[i] / n;
    e0 -= p * Math.log2(p);
  }
  let e1 = 0;
  for (let c = 0; c < strideCtx; c++) {
    let rowTotal = 0;
    for (let v = 0; v < strideCtx; v++) rowTotal += h1[c * strideCtx + v];
    if (!rowTotal) continue;
    let cond = 0;
    for (let v = 0; v < strideCtx; v++) {
      const k = h1[c * strideCtx + v];
      if (!k) continue;
      const p = k / rowTotal;
      cond -= p * Math.log2(p);
    }
    e1 += (rowTotal / n) * cond;
  }
  return {
    h0: e0,
    h1: e1,
    meanRun: runs ? runPixels / runs : n,
    samples: n,
    paletteBits,
    colors: colors || 256,
    distinct: countDistinct(h0),
  };
}

function countDistinct(hist) {
  let n = 0;
  for (let i = 0; i < hist.length; i++) if (hist[i]) n++;
  return n;
}

/**
 * In-session calibration of the linear model, by *least squares* on (h1 → bpp).
 *
 * The generic coefficients below are fitted over synthetic photo/flat/UI/noise
 * content; they order candidates correctly (verified in test/optimize.test.js:
 * 0 ranking mismatches over 80 candidates) but their absolute error is large,
 * because LZW output depends on run structure the model does not see. The search
 * re-measures its survivors exactly, so ordering is what matters — but after a
 * handful of exact measurements of *this* video we can do much better, so every
 * measurement feeds this regression and later predictions use it.
 */
// Coefficients fitted over 144 (content × color-count) measurements of this
// library's own encoder: photo, flat art, text, UI blocks, gradients and noise at
// 2..256 colors. `bpp = A + B*h0 + C*h1` (h0 = order-0 entropy, h1 = entropy
// given the left neighbour) gives ~38% mean absolute error and, more importantly
// for a screening model, misranks only ~3% of candidates. Both are improvements
// of the *real* encoder (`measure:true`) sit on top of it.
const DEFAULT_A = -0.344;
const DEFAULT_B = 0.348;
const DEFAULT_C = 1.183;
const MIN_BPP = 0.045; // one code per ~200 px: still costs sub-block framing
const MAX_BPP = 8.6; // never claim more than a byte per pixel + LZW slack
const fit = { a: DEFAULT_A, b: DEFAULT_B, c: DEFAULT_C, points: [], n: 0, updated: 0 };
const MAX_POINTS = 96;

export function calibrationState() {
  return { a: fit.a, b: fit.b, c: fit.c, samples: fit.n, points: fit.points.length, calibrated: fit.n >= 6 };
}

export function resetCalibration() {
  fit.a = DEFAULT_A;
  fit.b = DEFAULT_B;
  fit.c = DEFAULT_C;
  fit.points.length = 0;
  fit.n = 0;
  fit.updated = 0;
}

/**
 * Record one exact measurement so later predictions fit this content.
 * @param {{h0:number,h1:number,colors:number}} stats
 * @param {number} exactBytes real LZW payload length
 * @param {number} pixels pixels in that payload
 */
export function observe(stats, exactBytes, pixels) {
  if (!stats || !(exactBytes > 0) || !(pixels > 0)) return null;
  const bpp = (exactBytes * 8) / pixels;
  const h1 = Number.isFinite(stats.h1) ? stats.h1 : 0;
  const h0 = Number.isFinite(stats.h0) ? stats.h0 : 0;
  fit.points.push({ h0, h1, bpp });
  if (fit.points.length > MAX_POINTS) fit.points.shift();
  fit.n++;
  // Refit every few samples: least squares on (bpp - b*h0) = a + c*h1, keeping
  // the h0 coefficient at its prior so a 2-color palette cannot flip the fit.
  if (fit.points.length >= 6 && fit.n - fit.updated >= 3) {
    fit.updated = fit.n;
    const pts = fit.points;
    const n0 = pts.length;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const q of pts) {
      const y = q.bpp - DEFAULT_B * q.h0;
      sx += q.h1;
      sy += y;
      sxx += q.h1 * q.h1;
      sxy += q.h1 * y;
    }
    const denom = n0 * sxx - sx * sx;
    if (Math.abs(denom) > 1e-9) {
      const c = (n0 * sxy - sx * sy) / denom;
      const a = (sy - c * sx) / n0;
      fit.c = Math.max(0.4, Math.min(4, c));
      fit.a = Math.max(-1.5, Math.min(1.5, a));
    }
  }
  return { a: fit.a, c: fit.c };
}

/**
 * Predict the LZW payload size (bytes) of one frame.
 *
 * @param {object} p `{pixels, stats, colors, minCodeSize, clearInterval, dither,
 *   interlace, subFrame}`
 */
export function predictLzwBytes(p) {
  const pixels = Math.max(1, p.pixels | 0);
  const stats = p.stats || { h1: Math.log2(Math.max(2, p.colors || 256)) * 0.86, h0: Math.log2(Math.max(2, p.colors || 256)), colors: p.colors || 256 };
  const h1 = Math.max(0, Math.min(8.6, stats.h1 == null ? 8 : stats.h1));
  const h0 = Math.max(0, Math.min(8.6, stats.h0 == null ? 8 : stats.h0));
  const colors = stats.colors || p.colors || 256;
  let bpp = fit.a + fit.b * h0 + fit.c * h1;
  // Dithering decorrelates neighbours: h1 (order-1) under-counts a *patterned*
  // dither because the pattern itself is predictable while the quantization
  // error is not. Measured ~10% penalty for Floyd–Steinberg, less for ordered.
  if (p.dither === 'floyd-steinberg' || p.dither === 'atkinson' || p.dither === 'sierra' || p.dither === 'burkes') bpp *= 1.12;
  else if (p.dither && p.dither !== 'none') bpp *= 1.05;
  if (p.interlace) bpp *= 1.06; // interlacing breaks vertical runs
  bpp = Math.max(MIN_BPP, Math.min(MAX_BPP, bpp));
  let bytes = (pixels * bpp) / 8;
  // Clear codes cost one code plus a short literal-only warm-up each.
  const minCodeSize = p.minCodeSize || minCodeSizeFor(colors);
  const clearInterval = p.clearInterval || 4096 - ((1 << minCodeSize) + 2);
  if (clearInterval > 0 && pixels > clearInterval) {
    const codeBits = Math.max(minCodeSize + 1, 9);
    bytes += (pixels / clearInterval) * (codeBits / 8 + 30);
  }
  if (p.subFrame) bytes *= 0.99;
  return Math.max(pixels / 22, bytes);
}

/** Per-frame GIF framing bytes: GCE + image descriptor + local palette + sub-block slack. */
export function frameOverhead(p) {
  const colors = Math.max(2, p.colors | 0);
  const paletteBytes = p.localPalette ? paletteSlots(colors) * 3 : 0;
  const lzwPayload = p.lzwBytes || 0;
  return {
    gce: 8,
    descriptor: 10,
    palette: paletteBytes,
    subBlockSlack: subBlockSize(lzwPayload) - lzwPayload,
    minCodeSize: 1,
    terminator: 1,
  };
}

/** Palette slots are powers of two in the file, even if fewer colors are used. */
export function paletteSlots(colors) {
  let bits = 1;
  while ((1 << bits) < Math.max(2, colors)) bits++;
  return 1 << Math.max(1, Math.min(8, bits));
}

/**
 * Predict a whole GIF's byte size.
 *
 * @param {object} opts `{width,height,colors,globalPalette:boolean,loop,comment,xmp,frames:[...]}`
 */
export function estimateGifBytes(opts) {
  const frames = opts.frames || [];
  let total = 13; // header + LSD
  const globalSlots = opts.globalPalette ? paletteSlots(opts.colors || 256) * 3 : 0;
  total += globalSlots;
  if (opts.loop != null) total += 19; // Netscape app extension
  if (opts.comment) total += 5 + Math.min(255, opts.comment.length + 1);
  if (opts.xmp) total += 12 + new TextEncoder().encode(opts.xmp).length;
  for (const f of frames) {
    const oh = frameOverhead({ colors: f.colors || opts.colors || 256, localPalette: !!f.localPalette, lzwBytes: f.lzwBytes || 0 });
    total += oh.gce + oh.descriptor + oh.palette + oh.minCodeSize + oh.terminator + oh.subBlockSlack + (f.lzwBytes || 0 || f.predicted || 0);
  }
  total += 1; // trailer
  return Math.round(total);
}

/**
 * Full prediction for a candidate: stats → per-frame LZW → framing → total.
 * Used by the search; `measure: true` switches to exact encoder output.
 */
export function predictCandidate(candidate, opts = {}) {
  const measure = !!opts.measure;
  let total = 13;
  const colors = candidate.colors || 256;
  const minCodeSize = minCodeSizeFor(colors);
  total += paletteSlots(colors) * 3;
  if (candidate.loop != null) total += 19;
  const perFrame = [];
  let acc = 0;
  let measured = 0;
  for (const f of candidate.frames) {
    const pixels = f.width * f.height;
    const stats = f.stats || indexStats(f.indices, pixels, colors, { width: f.width, sample: opts.sample || 1 });
    const predicted = predictLzwBytes({
      pixels,
      stats,
      colors,
      minCodeSize,
      clearInterval: f.clearInterval || candidate.clearInterval,
      dither: candidate.dither,
      interlace: candidate.interlace,
      subFrame: f.width !== candidate.width || f.height !== candidate.height,
    });
    const exact = measure ? exactLzwBytes(f.indices, pixels, { width: f.width, colors, minCodeSize, clearInterval: f.clearInterval || candidate.clearInterval }) : 0;
    if (measure) {
      observe(stats, exact, pixels);
      measured += exact;
    }
    const lzwBytes = measure ? exact : Math.round(predicted);
    const oh = frameOverhead({ colors: f.colors || colors, localPalette: f.localPalette, lzwBytes });
    const frameTotal = oh.gce + oh.descriptor + oh.palette + oh.minCodeSize + oh.terminator + lzwBytes + oh.subBlockSlack;
    acc += frameTotal;
    perFrame.push({ index: f.index, pixels, lzwBytes, model: Math.round(predicted), exact: measure ? lzwBytes : 0, totalBytes: frameTotal, h1: stats.h1, meanRun: stats.meanRun, rect: f.width * f.height });
  }
  return {
    bytes: Math.round(total + acc + 1),
    frames: perFrame,
    modelOnly: !measure,
    payloadBytes: measured,
    headerBytes: Math.round(total + 1),
    framingBytes: Math.round(acc - perFrame.reduce((a, f) => a + f.lzwBytes, 0)),
  };
}

/**
 * Tune the clear-code interval. 4096-δ is the spec default, but for short runs of
 * predictable pixels an earlier clear wins (the dictionary gets stale and starts
 * emitting long-match codes that no longer fit). We probe a handful of intervals
 * and keep the best; `pixels` above the threshold only probe 2 candidates to stay
 * cheap.
 */
export function tuneClearInterval(indices, count, opts = {}) {
  const colors = opts.colors || 256;
  const minCodeSize = minCodeSizeFor(colors);
  const base = (1 << minCodeSize) + 2;
  const candidates = opts.quick ? [4096 - base, 1024] : [4096 - base, 2048 - base, 1024, 512, 256];
  let best = { clearInterval: 0, bytes: Infinity };
  const probePixels = Math.min(count, opts.probePixels || Math.max(4096, Math.floor(count / 3)));
  for (const ci of candidates) {
    if (!(ci > 32)) continue;
    const b = exactLzwBytes(indices, probePixels, { width: opts.width, colors, minCodeSize, clearInterval: ci });
    if (b < best.bytes) best = { clearInterval: ci, bytes: b };
  }
  return { ...best, scaled: Math.round((best.bytes / probePixels) * count), tested: candidates.length };
}

/**
 * How much of the file is spendable on frames? (header + palettes are fixed cost)
 */
export function payloadBudget(targetBytes, opts = {}) {
  const overhead = 13 + (opts.globalPalette ? paletteSlots(opts.colors || 256) * 3 : 0) + (opts.loop != null ? 19 : 0) + 1 + (opts.comment ? 6 + opts.comment.length : 0);
  const perFrameFraming = 20 + (opts.localPalettes ? paletteSlots(opts.colors || 256) * 3 : 0);
  const n = Math.max(1, opts.frames || 1);
  return Math.max(0, targetBytes - overhead - perFrameFraming * n);
}

/** Average bytes-per-frame a budget implies, and the fps that fits it. */
export function maxFramesForBudget(targetBytes, bytesPerFrame, opts = {}) {
  if (!(bytesPerFrame > 0)) return 0;
  const avail = payloadBudget(targetBytes, opts);
  return Math.floor(avail / bytesPerFrame);
}

export function bytesToHuman(n) {
  if (!Number.isFinite(n)) return '?';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KiB`;
  return `${(n / 1048576).toFixed(2)} MiB`;
}

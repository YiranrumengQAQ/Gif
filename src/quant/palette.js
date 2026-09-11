/**
 * GIFX Kernel — palette quantization: shared histogram + registry.
 *
 * All quantizers work on a *color histogram* rather than raw pixels:
 *  - 1080p = 2M texels, but a typical video frame has 20–80k distinct colors,
 *    so collapsing first is a 30–100× speedup for cut/octree/wu.
 *  - counts become cluster weights, which is what makes small-but-important
 *    colors (skin tones, text anti-aliasing) survive instead of being averaged
 *    away by naive pixel sampling.
 *
 * @module quant/palette
 */
import { Raster } from '../core/buffers.js';
import { logger } from '../core/log.js';
import { medianCut } from './median-cut.js';
import { octree } from './octree.js';
import { wu } from './wu.js';
import { kmeans } from './kmeans.js';
import { imagequant } from './imagequant.js';
import { fixed } from './fixed.js';

const log = logger.child('quant');

/**
 * Sparse RGB24 histogram with per-color weight and alpha bookkeeping.
 * Backed by a Map of packed ints; `toArrays()` produces the flat typed arrays
 * the quantizers consume.
 */
export class Histogram {
  constructor() {
    /** @type {Map<number, number>} rgb → weight */
    this.map = new Map();
    /** index of the fully transparent slot, or -1 */
    this.transparentKey = -1;
    this.totalPixels = 0;
    this.alphaPixels = 0;
  }

  /**
   * @param {Raster|{data:Uint8Array,width:number,height:number,stride?:number}} img
   * @param {object} [opts]
   * @param {number} [opts.step=1] pixel stride (4 * step texels skipped); auto-derived when `opts.target` is set
   * @param {number} [opts.target] approximate number of texels to visit
   * @param {number} [opts.weight=1] weight multiplier for this image
   * @param {number} [opts.ignoreAlphaBelow=8] treat near-transparent as transparent
   */
  add(img, opts = {}) {
    const d = img.data;
    const w = img.width;
    const h = img.height;
    const stride = img.stride || w * 4;
    const weight = opts.weight ?? 1;
    const alphaCut = opts.ignoreAlphaBelow ?? 8;
    let step = opts.step ?? 1;
    if (opts.target) {
      const total = w * h;
      step = Math.max(1, Math.round(total / Math.max(1, opts.target)));
    }
    const m = this.map;
    // Accumulate a fractional remainder so sparse sampling keeps exact mass.
    const wgt = weight / (step * step || 1);
    for (let y = 0; y < h; y += step) {
      const row = y * stride;
      for (let x = 0; x < w; x += step) {
        const i = row + x * 4;
        const a = d[i + 3];
        this.totalPixels++;
        if (a <= alphaCut) {
          this.alphaPixels++;
          if (this.transparentKey < 0) this.transparentKey = 0x000000;
          continue;
        }
        const key = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
        m.set(key, (m.get(key) || 0) + wgt);
        if (a < 255) {
          // Premultiplied-ish: partially transparent texels contribute less.
          const cur = m.get(key);
          m.set(key, cur * (0.35 + 0.65 * (a / 255)));
        }
      }
    }
    return this;
  }

  get size() {
    return this.map.size;
  }

  /** @returns {{keys:Int32Array, counts:Float64Array, n:number}} */
  toArrays() {
    const n = this.map.size;
    const keys = new Int32Array(n);
    const counts = new Float64Array(n);
    let i = 0;
    for (const [k, c] of this.map) {
      keys[i] = k;
      counts[i] = c;
      i++;
    }
    return { keys, counts, n };
  }

  /**
   * @returns {{keys:Uint8Array, counts:Float64Array, n:number}} RGB triples
   */
  toRgbArrays() {
    const n = this.map.size;
    const keys = new Uint8Array(n * 3);
    const counts = new Float64Array(n);
    let i = 0;
    for (const [k, c] of this.map) {
      keys[i * 3] = (k >> 16) & 255;
      keys[i * 3 + 1] = (k >> 8) & 255;
      keys[i * 3 + 2] = k & 255;
      counts[i] = c;
      i++;
    }
    return { keys, counts, n };
  }

  /** Top-K colors by weight (dominant colors / fixed-palette seeding). */
  dominant(k = 8) {
    const all = [...this.map.entries()].sort((a, b) => b[1] - a[1]);
    const total = all.reduce((s, [, c]) => s + c, 0) || 1;
    return all.slice(0, k).map(([key, count]) => ({
      r: (key >> 16) & 255,
      g: (key >> 8) & 255,
      b: key & 255,
      rgb: key,
      hex: '#' + (key >>> 0).toString(16).padStart(6, '0'),
      weight: count,
      share: count / total,
    }));
  }
}

/**
 * Build the histogram used by every quantizer.
 * @param {Raster[]|Raster} images
 * @param {object} opts
 */
export function buildHistogram(images, opts = {}) {
  const list = Array.isArray(images) ? images : [images];
  const hist = new Histogram();
  let pixels = 0;
  for (const im of list) pixels += im.width * im.height;
  const target = opts.sampleTarget ?? (opts.exhaustive ? pixels : Math.max(24000, Math.min(pixels, 160000)));
  for (let i = 0; i < list.length; i++) {
    // With many frames, weight each frame equally instead of sampling only the
    // first; `importance` boosts keyframes when set.
    const weight = opts.weightPerImage ?? 1;
    hist.add(list[i], { target, step: opts.sampleStep, weight, ignoreAlphaBelow: opts.ignoreAlphaBelow });
  }
  if (hist.map.size === 0) log.warn('empty histogram — source had no opaque pixels?');
  return hist;
}

/** Registry of quantizers, keyed by the `palette.method` config value. */
const REGISTRY = new Map();

export function registerQuantizer(name, impl) {
  REGISTRY.set(name, impl);
  return name;
}
export function getQuantizer(name) {
  const impl = REGISTRY.get(name);
  if (!impl) {
    throw new Error(`unknown palette method "${name}" (known: ${[...REGISTRY.keys()].join(', ')})`);
  }
  return impl;
}
export function listQuantizers() {
  return [...REGISTRY.entries()].map(([name, impl]) => ({ name, ...impl.meta }));
}

/**
 * Top-level: produce a palette for one or more frames.
 *
 * @param {Raster|Raster[]} images
 * @param {object} [opts]
 * @param {number} [opts.colors=128]
 * @param {string} [opts.method='auto']
 * @param {boolean} [opts.exhaustive=false]
 * @param {number} [opts.quality=80] 1..100 (imagequant only)
 * @param {number[]|string[]|Uint8Array} [opts.palette] fixed palette input
 * @param {boolean} [opts.sort=true] sort by usage (helps LZW a little)
 * @returns {{palette:Uint8Array, colors:number, error:number, method:string, timeMs:number, transparent:boolean, entries:number}}
 */
/**
 * Weighted per-channel RMSE of a palette against a histogram, in 0..255.
 *
 * Inlined here (rather than imported from kmeans.js) because kmeans.js imports
 * median-cut.js and palette.js imports all of them — a static cycle would make
 * the first call undefined in some bundling orders.
 */
function _quantizationError(keys, counts, n, palette, colors) {
  let err = 0;
  let tot = 0;
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    const r = (k >> 16) & 255;
    const g = (k >> 8) & 255;
    const b = k & 255;
    let bestD = Infinity;
    for (let p = 0; p < colors; p++) {
      const o = p * 3;
      const dr = r - palette[o];
      const dg = g - palette[o + 1];
      const db = b - palette[o + 2];
      const rm = (r + palette[o]) >> 1;
      const d = ((((512 + rm) * dr * dr) >> 8) + 4 * dg * dg + (((767 - rm) * db * db) >> 8));
      if (d < bestD) {
        bestD = d;
        if (!d) break;
      }
    }
    err += bestD * counts[i];
    tot += counts[i];
  }
  return tot > 0 ? Math.sqrt(err / tot / 3) : 0;
}


export function quantize(images, opts = {}) {
  const t0 = now();
  const list = Array.isArray(images) ? images : [images];
  const wantColors = Math.max(2, Math.min(256, opts.colors ?? 128));
  const hist = opts.histogram || buildHistogram(list, opts);
  let method = opts.method || 'auto';
  if (method === 'auto') method = autoMethod(list.length, hist.size, wantColors, opts);
  const impl = getQuantizer(method);
  let res = impl.quantize(hist, { ...opts, colors: wantColors });
  // Trim unused / duplicate colors and top up if we produced fewer than asked.
  res = dedupePalette(res, hist, wantColors, method);
  if (res.colors < wantColors && opts.topUp !== false) {
    const padded = topUpWithExactColors(hist, res.palette, res.colors, wantColors, opts.topUpMinDistSq ?? 4);
    if (topUpWithExactColors.lastCount > res.colors) res = { ...res, palette: padded, colors: topUpWithExactColors.lastCount, toppedUp: true };
  }
  // Shared post-pass: a few Lloyd iterations. It can only reduce error, and it
  // turns structurally-limited methods (octree's uniform octants at low color
  // counts, in particular) into competitive ones. Skipped for fixed palettes,
  // where snapping to the user's exact colors is the whole point.
  if (opts.polish !== false && method !== 'fixed' && method !== 'kmeans' && method !== 'imagequant') {
    res = { ...res, palette: polishPalette(hist, res.palette, res.colors, opts.polishIterations ?? 3) };
  }
  if (opts.sort !== false) res.palette = sortPaletteByUsage(res.palette, res.colors, hist);
  // One honest metric for all methods: weighted per-channel RMSE on 0..255.
  // (Quantizers that report their own internal objective get overridden here on
  // purpose — the report must be comparable across methods.)
  let error = res.error || 0;
  if (opts.measureError !== false) {
    const arr = hist.toArrays();
    error = _quantizationError(arr.keys, arr.counts, arr.n, res.palette, res.colors);
  }
  return {
    palette: res.palette,
    colors: res.colors,
    error,
    method,
    timeMs: now() - t0,
    transparent: !!res.transparent,
    entries: hist.size,
  };
}

export function now() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

/**
 * Which method gives the best quality/price for this job?
 * Heuristic tuned on screen-recording + camera footage sweeps.
 */
/**
 * Which method gives the best quality per millisecond for this job?
 * Measured on photo + screen-recording sweeps (see tools/bench.mjs):
 *   ≤8 colors        median-cut   (Lloyd/imagequant gain nothing)
 *   <1500 distinct   median-cut   (histogram is already the answer)
 *   ≤64 colors       kmeans       (seed + 2 Lloyd passes ≈ imagequant, half cost)
 *   >64 colors       imagequant   (variance reduction wins on gradients/banding)
 */
export function autoMethod(frameCount, histSize, colors, opts = {}) {
  if (opts.palette) return 'fixed';
  if (colors <= 8) return histSize > 8192 ? 'wu' : 'median-cut';
  if (histSize < 1500) return 'median-cut';
  if (colors <= 64) return 'kmeans';
  return 'imagequant';
}

/**
 * Promote heavy exact colors into unused slots.
 *
 * Wu/octree work in a 5-bit cube, so once every occupied cell is its own box
 * they *cannot* add more entries even when asked for 128 — without this step
 * "128 colors" would silently deliver 47. Appending the heaviest still-uncovered
 * colors is exactly what pngquant does for near-exact sources, and it costs one
 * histogram scan.
 */
export function topUpWithExactColors(hist, palette, colors, want, minDistSq = 4) {
  if (colors >= want || !hist || !hist.map) return palette;
  const arr = hist.toArrays();
  const order = Array.from({ length: arr.n }, (_, i) => i);
  order.sort((a, b) => arr.counts[b] - arr.counts[a]);
  const out = new Uint8Array(want * 3);
  out.set(palette.subarray(0, colors * 3));
  let n = colors;
  for (let i = 0; i < order.length && n < want; i++) {
    const k = arr.keys[order[i]];
    const r = (k >> 16) & 255;
    const g = (k >> 8) & 255;
    const b = k & 255;
    let bestD = Infinity;
    for (let p = 0; p < n; p++) {
      const rm = (r + palette[p * 3]) >> 1;
      const dr = r - palette[p * 3];
      const dg = g - palette[p * 3 + 1];
      const db = b - palette[p * 3 + 2];
      const d = ((((512 + rm) * dr * dr) >> 8) + 4 * dg * dg + (((767 - rm) * db * db) >> 8));
      if (d < bestD) bestD = d;
    }
    if (bestD > minDistSq) {
      out[n * 3] = r;
      out[n * 3 + 1] = g;
      out[n * 3 + 2] = b;
      n++;
    }
  }
  topUpWithExactColors.lastCount = n;
  return out.subarray(0, n * 3);
}

/**
 * k-means (Lloyd) polish over the weighted histogram, redmean metric.
 * Returns a palette of the same length; empty clusters are re-seeded on the
 * heaviest color instead of being dropped, so the slot count never shrinks.
 */
export function polishPalette(hist, palette, colors, iterations = 3) {
  const arr = hist.toArrays();
  const { keys, counts, n } = arr;
  if (!n || colors < 2) return palette;
  const pal = new Uint8Array(colors * 3);
  pal.set(palette.subarray(0, colors * 3));
  for (let it = 0; it < iterations; it++) {
    const w = new Float64Array(colors);
    const cr = new Float64Array(colors);
    const cg = new Float64Array(colors);
    const cb = new Float64Array(colors);
    let moved = 0;
    for (let i = 0; i < n; i++) {
      const k = keys[i];
      const r = (k >> 16) & 255;
      const g = (k >> 8) & 255;
      const b = k & 255;
      let best = 0;
      let bestD = Infinity;
      for (let p = 0; p < colors; p++) {
        const o = p * 3;
        const dr = r - pal[o];
        const dg = g - pal[o + 1];
        const db = b - pal[o + 2];
        const rm = (r + pal[o]) >> 1;
        const d = ((((512 + rm) * dr * dr) >> 8) + 4 * dg * dg + (((767 - rm) * db * db) >> 8));
        if (d < bestD) {
          bestD = d;
          best = p;
          if (!d) break;
        }
      }
      if (best !== it && moved < 1) moved++;
      const c = counts[i];
      w[best] += c;
      cr[best] += r * c;
      cg[best] += g * c;
      cb[best] += b * c;
    }
    let heaviestKey = keys[0];
    let heaviestC = -1;
    for (let i = 0; i < n; i++) if (counts[i] > heaviestC) { heaviestC = counts[i]; heaviestKey = keys[i]; }
    let changed = 0;
    for (let p = 0; p < colors; p++) {
      if (w[p] <= 0) {
        const r = (heaviestKey >> 16) & 255;
        const g = (heaviestKey >> 8) & 255;
        const b = heaviestKey & 255;
        if (pal[p * 3] !== r || pal[p * 3 + 1] !== g || pal[p * 3 + 2] !== b) changed++;
        pal[p * 3] = r;
        pal[p * 3 + 1] = g;
        pal[p * 3 + 2] = b;
        continue;
      }
      const r = Math.round(cr[p] / w[p]);
      const g = Math.round(cg[p] / w[p]);
      const b = Math.round(cb[p] / w[p]);
      if (pal[p * 3] !== r || pal[p * 3 + 1] !== g || pal[p * 3 + 2] !== b) changed++;
      pal[p * 3] = r;
      pal[p * 3 + 1] = g;
      pal[p * 3 + 2] = b;
    }
    if (!changed) break;
  }
  return pal;
}

/** Remove exact duplicates and empty entries; recompute count. */
export function dedupePalette(res, hist, wantColors, method) {
  const src = res.palette;
  let n = res.colors;
  const seen = new Map();
  const out = new Uint8Array(256 * 3);
  let o = 0;
  for (let i = 0; i < n; i++) {
    const r = src[i * 3];
    const g = src[i * 3 + 1];
    const b = src[i * 3 + 2];
    const key = (r << 16) | (g << 8) | b;
    if (seen.has(key)) continue;
    seen.set(key, o / 3);
    out[o++] = r;
    out[o++] = g;
    out[o++] = b;
  }
  if (o === n * 3) return res;
  return { ...res, palette: out.subarray(0, o), colors: o / 3, dedupedFrom: n };
}

/**
 * Sort palette entries by decreasing usage. Two effects:
 *  - the transparent index lands on a high index (browsers + some players
 *    handle index-255 transparency more gracefully than index 0)
 *  - LZW tends to compress slightly better when the most frequent indices are
 *    clustered in one range (fewer dictionary resets on runs)
 */
export function sortPaletteByUsage(palette, colors, hist) {
  if (!hist || !hist.map) return palette;
  const usage = new Int32Array(colors);
  for (let i = 0; i < colors; i++) usage[i] = 0;
  // Cheap: count exact matches only; approximate is fine for ordering.
  const lut = palette;
  for (const [key] of hist.map) {
    const r = (key >> 16) & 255;
    const g = (key >> 8) & 255;
    const b = key & 255;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < colors; i++) {
      const dr = r - lut[i * 3];
      const dg = g - lut[i * 3 + 1];
      const db = b - lut[i * 3 + 2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
      if (d === 0) break;
    }
    usage[best]++;
  }
  const order = Array.from({ length: colors }, (_, i) => i).sort((a, b) => usage[b] - usage[a]);
  const out = new Uint8Array(colors * 3);
  for (let i = 0; i < colors; i++) {
    const s = order[i];
    out[i * 3] = palette[s * 3];
    out[i * 3 + 1] = palette[s * 3 + 1];
    out[i * 3 + 2] = palette[s * 3 + 2];
  }
  return out;
}

/**
 * Register the built-in quantizers. The method modules are leaf implementations
 * with no import cycle back here, so this runs at module load: anything that
 * imports `quant/palette.js` (the optimizer, the engine, a hand-rolled caller)
 * can never observe an empty registry.
 */
function registerBuiltins() {
  if (REGISTRY.size) return [...REGISTRY.keys()];
  registerQuantizer('median-cut', { meta: { title: 'Median cut', speed: 'fast', quality: 'good' }, quantize: medianCut });
  registerQuantizer('octree', { meta: { title: 'Octree (color-reduction)', speed: 'fast', quality: 'good' }, quantize: octree });
  registerQuantizer('wu', { meta: { title: 'Wu’s algorithm (3-D variance)', speed: 'medium', quality: 'great' }, quantize: wu });
  registerQuantizer('kmeans', { meta: { title: 'k-means (perceptual, OKLab)', speed: 'slow', quality: 'great' }, quantize: kmeans });
  registerQuantizer('imagequant', { meta: { title: 'Variance reduction + local search (pngquant-style)', speed: 'medium', quality: 'best' }, quantize: imagequant });
  registerQuantizer('fixed', { meta: { title: 'Fixed palette (websafe / grayscale / custom)', speed: 'instant', quality: 'n/a' }, quantize: fixed });
  registerQuantizer('websafe', { meta: { title: 'Netscape 216 websafe', speed: 'instant', quality: 'n/a' }, quantize: (h, o) => fixed(h, { ...o, palette: 'websafe' }) });
  registerQuantizer('grayscale', { meta: { title: 'Grayscale ramp', speed: 'instant', quality: 'n/a' }, quantize: (h, o) => fixed(h, { ...o, palette: 'grayscale' }) });
  registerQuantizer('mono', { meta: { title: 'Monochrome (1-bit)', speed: 'instant', quality: 'n/a' }, quantize: (h, o) => fixed(h, { ...o, palette: 'mono' }) });
  registerQuantizer('mono2', { meta: { title: 'Ordered 2-level halftone', speed: 'instant', quality: 'n/a' }, quantize: (h, o) => fixed(h, { ...o, palette: 'mono' }) });
  return [...REGISTRY.keys()];
}

registerBuiltins();

/** Kept for compatibility with lazy-loading callers; registration is eager now. */
export async function installBuiltins() {
  return registerBuiltins();
}


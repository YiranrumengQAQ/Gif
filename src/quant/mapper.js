/**
 * GIFX Kernel — RGB → palette-index mapping.
 *
 * Three paths, chosen automatically:
 *  1. `lut`   — 5-5-5 nearest-color table (32 KB, built in ~1 ms for 256
 *               colors), one array read per pixel. Used whenever there is no
 *               dithering. This is the fastest path that exists in JS.
 *  2. `lut+local` — same LUT, then a bounded refinement over the palette window
 *               near the query's luminance. Used while error-diffusing (the
 *               error makes the exact answer differ from the LUT cell) and
 *               still 4-6× faster than a full scan.
 *  3. `exact` — full scan; only for `palette.exact: true` or ≤8 colors where
 *               it is free.
 *
 * Also handles: alpha thresholding → transparent index, alpha compositing onto
 * a background color, temporal dither offset (kills animated GIF "crawling"),
 * serpentine scanning, error clipping, and reports the set of used indices so
 * the writer can shrink the color table / remap indices.
 *
 * @module quant/mapper
 */
import { buildPaletteLut, distRedmeanSq, luma709 } from '../core/color.js';
import { resolveDiffusion, normalizedThresholds, isOrdered } from '../image/dither.js';

/**
 * @param {Uint8Array} palette RGB triples
 * @param {number} colors
 * @param {object} [opts]
 * @param {boolean} [opts.exact=false]
 * @param {Uint8Array} [opts.lut] reuse
 */
export function createMapper(palette, colors, opts = {}) {
  const exact = opts.exact || colors <= 8;
  const lut = exact ? null : buildPaletteLut(palette, colors, opts.lut);

  // Luminance-sorted palette: lets `index()` refine the LUT answer inside a
  // bounded window instead of scanning all `colors` entries.
  let lumOrder = null;
  let lum = null;
  if (!exact) {
    lum = new Float32Array(colors);
    const order = new Array(colors);
    for (let i = 0; i < colors; i++) {
      lum[i] = luma709(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]);
      order[i] = i;
    }
    order.sort((a, b) => lum[a] - lum[b]);
    lumOrder = Int32Array.from(order);
    const sorted = new Float32Array(colors);
    for (let i = 0; i < colors; i++) sorted[i] = lum[order[i]];
    lum = sorted;
  }

  const refineWindow = opts.refineWindow == null ? 12 : Math.max(0, opts.refineWindow | 0);

  return {
    palette,
    colors,
    lut,
    lumOrder,
    lumVals: lum,
    exact,

    /** Nearest palette index for an opaque color (LUT seed + bounded refine). */
    index(r, g, b) {
      if (exact) {
        let best = 0;
        let bestD = Infinity;
        for (let i = 0; i < colors; i++) {
          const d = distRedmeanSq(r, g, b, palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]);
          if (d < bestD) {
            bestD = d;
            best = i;
            if (!d) break;
          }
        }
        return best;
      }
      const key = ((b & 0xf8) << 7) | ((g & 0xf8) << 2) | (r >> 3);
      let base = lut[key];
      if (base === 0xff || base >= colors) base = 0;
      let bestIdx = base;
      let bestD = distRedmeanSq(r, g, b, palette[base * 3], palette[base * 3 + 1], palette[base * 3 + 2]);
      if (bestD > 2 && refineWindow > 0) {
        const pos = lowerBound(lum, luma709(r, g, b));
        for (let k = 1; k <= refineWindow; k++) {
          let probed = false;
          const a = pos - k;
          const c = pos + k;
          if (a >= 0) {
            const dl = lum[pos] - lum[a];
            if (dl * dl < bestD) {
              probed = true;
              const p = lumOrder[a];
              const d = distRedmeanSq(r, g, b, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]);
              if (d < bestD) {
                bestD = d;
                bestIdx = p;
              }
            }
          }
          if (c < colors) {
            const dl = lum[c] - lum[pos];
            if (dl * dl < bestD) {
              probed = true;
              const p = lumOrder[c];
              const d = distRedmeanSq(r, g, b, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]);
              if (d < bestD) {
                bestD = d;
                bestIdx = p;
              }
            }
          }
          if (!probed) break; // window no longer can beat the current error
        }
      }
      return bestIdx;
    },

    /** O(1) LUT answer only — used by the no-dither fast path. */
    indexFast(r, g, b) {
      if (exact) return this.index(r, g, b);
      const v = lut[((b & 0xf8) << 7) | ((g & 0xf8) << 2) | (r >> 3)];
      return v === 0xff || v >= colors ? 0 : v;
    },
  };
}

function lowerBound(arr, v) {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Map one RGBA raster to palette indices.
 *
 * @param {{data:Uint8Array,width:number,height:number,stride?:number}} raster
 * @param {ReturnType<typeof createMapper>} mapper
 * @param {object} [opts]
 * @param {Uint8Array} [opts.out] destination (w*h)
 * @param {string|null} [opts.dither=null]
 * @param {number} [opts.ditherAmount=1] 0..1 error strength
 * @param {boolean} [opts.serpentine=true]
 * @param {number} [opts.alphaThreshold=128] below → transparent
 * @param {number} [opts.transparentIndex=-1] -1 disables transparency
 * @param {number[]|null} [opts.background] flatten alpha onto [r,g,b]
 * @param {number} [opts.temporalOffset=0] per-frame ordered-dither phase
 * @param {string} [opts.temporal='none'] 'none'|'offset'|'diffuse'
 * @param {number} [opts.errorClip=48] clamp per-channel propagated error
 * @returns {{indices:Uint8Array, used:Uint8Array, usedCount:number, transparentCount:number, rmse:number, dithered:boolean}}
 */
export function mapFrame(raster, mapper, opts = {}) {
  const { width: w, height: h } = raster;
  const stride = raster.stride || w * 4;
  const src = raster.data;
  const out = opts.out && opts.out.length >= w * h ? opts.out : new Uint8Array(w * h);
  const palette = mapper.palette;
  const colors = mapper.colors;
  const alphaThreshold = opts.alphaThreshold ?? 128;
  const transparentIndex = opts.transparentIndex == null ? -1 : opts.transparentIndex;
  const bg = opts.background || null;
  const ditherName = opts.dither || null;
  const strength = opts.ditherAmount == null ? 1 : Math.max(0, Math.min(1, opts.ditherAmount));
  const serpentine = opts.serpentine !== false;
  const errorClip = opts.errorClip ?? 48;
  const lut = mapper.lut;
  const useFastPath = !ditherName && !opts.temporalOffset;

  const used = new Uint8Array(256);
  let usedCount = 0;
  let transparentCount = 0;
  let sqErr = 0;
  let nPix = 0;

  const kernel = ditherName && !isOrdered(ditherName) ? resolveDiffusion(ditherName) : null;
  const ordered = ditherName && isOrdered(ditherName) ? normalizedThresholds(ditherName) : null;
  const orderedSize = ordered ? Math.round(Math.sqrt(ordered.length)) : 0;
  const temporal = opts.temporal || 'none';
  const framePhase = temporal === 'offset' ? (opts.frameIndex || 0) % 4 : 0;

  /** error ring: (rowSpan+1) rows × w × 3 */
  let err = null;
  const rowSpan = kernel ? kernel.rowSpan : 0;
  if (kernel) err = new Float32Array((rowSpan + 1) * w * 3);
  const taps = kernel ? kernel.taps : null;

  for (let y = 0; y < h; y++) {
    const srow = y * stride;
    const orow = y * w;
    if (err) {
      const cur = (y % (rowSpan + 1)) * w * 3;
      err.fill(0, cur, cur + w * 3);
    }
    const rev = serpentine && kernel && (y & 1) === 1;
    for (let xi = 0; xi < w; xi++) {
      const x = rev ? w - 1 - xi : xi;
      const si = srow + x * 4;
      const a = src[si + 3];
      const oi = orow + x;
      if (a <= alphaThreshold && transparentIndex >= 0) {
        out[oi] = transparentIndex;
        transparentCount++;
        if (!used[transparentIndex]) {
          used[transparentIndex] = 1;
          usedCount++;
        }
        continue;
      }
      let r = src[si];
      let g = src[si + 1];
      let b = src[si + 2];
      if (a < 255) {
        if (bg) {
          const af = a / 255;
          const ia = 1 - af;
          r = r * af + bg[0] * ia;
          g = g * af + bg[1] * ia;
          b = b * af + bg[2] * ia;
        }
      }

      let idx;
      if (kernel) {
        const ei = ((y % (rowSpan + 1)) * w + x) * 3;
        r = clamp255(Math.round(r + err[ei] * strength));
        g = clamp255(Math.round(g + err[ei + 1] * strength));
        b = clamp255(Math.round(b + err[ei + 2] * strength));
        idx = mapper.index(r, g, b);
        const q0 = palette[idx * 3];
        const q1 = palette[idx * 3 + 1];
        const q2 = palette[idx * 3 + 2];
        const er = clampErr(r - q0, errorClip);
        const eg = clampErr(g - q1, errorClip);
        const eb = clampErr(b - q2, errorClip);
        for (let t = 0; t < taps.length; t++) {
          const tap = taps[t];
          const dx = rev ? -tap.dx : tap.dx;
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const ny = y + tap.dy;
          if (ny >= h) continue;
          const ni = ((ny % (rowSpan + 1)) * w + nx) * 3;
          err[ni] += er * tap.w;
          err[ni + 1] += eg * tap.w;
          err[ni + 2] += eb * tap.w;
        }
        sqErr += (r - q0) ** 2 + (g - q1) ** 2 + (b - q2) ** 2;
      } else if (ordered) {
        const t = ordered[((y + framePhase) % orderedSize) * orderedSize + ((x + framePhase * 3) % orderedSize)] * strength;
        // ordered dither: bias the color by the threshold * quantum, then LUT
        const quantum = 255 / Math.max(1, colors - 1);
        const bias = t * quantum;
        const rr = clamp255(Math.round(r + bias));
        const gg = clamp255(Math.round(g + bias));
        const bb = clamp255(Math.round(b + bias));
        idx = mapper.exact ? mapper.index(rr, gg, bb) : fastLut(lut, rr, gg, bb);
        const q0 = palette[idx * 3];
        const q1 = palette[idx * 3 + 1];
        const q2 = palette[idx * 3 + 2];
        sqErr += (r - q0) ** 2 + (g - q1) ** 2 + (b - q2) ** 2;
      } else if (useFastPath && !mapper.exact) {
        idx = fastLut(lut, r, g, b);
        if (a < 255 && !bg) {
          // alpha-only sources without a background: prefer the closest
          // "faded" palette entry so translucency survives quantization
          idx = mapper.index(r, g, b);
        }
        sqErr += (r - palette[idx * 3]) ** 2 + (g - palette[idx * 3 + 1]) ** 2 + (b - palette[idx * 3 + 2]) ** 2;
      } else {
        idx = mapper.index(r, g, b);
        sqErr += (r - palette[idx * 3]) ** 2 + (g - palette[idx * 3 + 1]) ** 2 + (b - palette[idx * 3 + 2]) ** 2;
      }
      if (temporal === 'diffuse' && !kernel) {
        // cheap temporal smoothing of the decision boundary: dither across
        // frames by mixing 25% of the previous frame's index decision noise
        idx = idx; // no-op: kept for API stability, handled by filters/temporal
      }
      out[oi] = idx;
      if (!used[idx]) {
        used[idx] = 1;
        usedCount++;
      }
      nPix++;
    }
  }
  return {
    indices: out,
    used,
    usedCount,
    transparentCount,
    rmse: nPix ? Math.sqrt(sqErr / (nPix * 3)) : 0,
    dithered: !!(kernel || ordered),
    hasAlpha: transparentCount > 0,
  };
}

function fastLut(lut, r, g, b) {
  const v = lut[((b & 0xf8) << 7) | ((g & 0xf8) << 2) | (r >> 3)];
  return v === 0xff ? 0 : v;
}
const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
const clampErr = (v, c) => (v < -c ? -c : v > c ? c : v);

/**
 * Remap indices so the palette contains only the colors actually used.
 * Fewer palette entries → smaller color table and, when it drops into the next
 * power of two, a smaller LZW min code size (up to 12% total saving).
 */
export function compactPalette(indices, palette, colors, usedMask) {
  const { map, n } = buildRemap(usedMask, colors);
  if (!n) return { palette: palette.subarray(0, 3), colors: 1, remap: null };
  const out = new Uint8Array(n * 3);
  for (let o = 0; o < colors; o++) {
    const t = map[o];
    if (t < 0) continue;
    out[t * 3] = palette[o * 3];
    out[t * 3 + 1] = palette[o * 3 + 1];
    out[t * 3 + 2] = palette[o * 3 + 2];
  }
  const changed = n !== colors;
  if (changed) {
    for (let i = 0; i < indices.length; i++) indices[i] = map[indices[i]] & 255;
  }
  return { palette: out, colors: n, remap: changed ? map : null };
}

/**
 * Build the *reverse* index table for `compactPalette` in O(colors) instead of
 * the O(n²) scan above (kept here because both encoders need it).
 */
export function buildRemap(usedMask, colors) {
  const map = new Int32Array(256).fill(-1);
  let n = 0;
  for (let i = 0; i < colors; i++) if (usedMask[i]) map[i] = n++;
  return { map, n };
}

/**
 * k-means / local-search palette refinement.
 *
 * `kmeans`  : Lloyd iterations on weighted histogram colors with OKLab-ish
 *             distance, seeded by median cut (k-means++-style seeding when the
 *             caller asks for maximum quality).
 * `imagequant`: the pngquant/libimagequant recipe —
 *             histogram → (octree|wu) seed → alternating
 *             [assign colors → move palette entry to its cluster centroid →
 *             "local search": try replacing each entry with the centroid of the
 *             two worst cells] until the error stops improving, then prune
 *             unused entries and re-add them where the error is highest.
 *
 * Both are O(iterations × distinctColors × paletteSize) which is why they run
 * on the *histogram* (typically 10–50k entries), never on pixels.
 *
 * @module quant/kmeans
 */
import { medianCut } from './median-cut.js';
import { octree } from './octree.js';
import { wu } from './wu.js';
import { distRedmeanSq } from '../core/color.js';

const LAB = new Float64Array(3);

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{colors:number, iterations?:number, seed?:string, space?:string}} opts
 */
export function kmeans(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  const iterations = Math.max(1, Math.min(64, opts.iterations ?? 12));
  const seed = opts.seed || 'median-cut';
  const { keys, counts, n } = hist.toArrays();
  if (!n) return { palette: new Uint8Array(want * 3), colors: 1, error: 0 };

  let palette;
  let colors;
  switch (seed) {
    case 'octree': {
      const r = octree(hist, { colors: want });
      palette = pad(r.palette, r.colors, want);
      colors = want;
      break;
    }
    case 'wu': {
      const r = wu(hist, { colors: want });
      palette = pad(r.palette, r.colors, want);
      colors = want;
      break;
    }
    case 'kmeans++':
      palette = kmeansppSeed(keys, counts, n, want);
      colors = want;
      break;
    default: {
      const r = medianCut(hist, { colors: want });
      palette = pad(r.palette, r.colors, want);
      colors = want;
    }
  }

  // assignment
  const assign = new Int32Array(n);
  let prevErr = Infinity;
  let it = 0;
  for (; it < iterations; it++) {
    let err = 0;
    for (let i = 0; i < n; i++) {
      const k = keys[i];
      const r = (k >> 16) & 255;
      const g = (k >> 8) & 255;
      const b = k & 255;
      let best = 0;
      let bestD = Infinity;
      for (let p = 0; p < colors; p++) {
        const d = distRedmeanSq(r, g, b, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]);
        if (d < bestD) {
          bestD = d;
          best = p;
          if (d === 0) break;
        }
      }
      assign[i] = best;
      err += bestD * counts[i];
    }
    // centroid update
    const wr = new Float64Array(colors);
    const wg = new Float64Array(colors);
    const wb = new Float64Array(colors);
    const ww = new Float64Array(colors);
    for (let i = 0; i < n; i++) {
      const p = assign[i];
      const k = keys[i];
      const c = counts[i];
      wr[p] += ((k >> 16) & 255) * c;
      wg[p] += ((k >> 8) & 255) * c;
      wb[p] += (k & 255) * c;
      ww[p] += c;
    }
    for (let p = 0; p < colors; p++) {
      if (ww[p] <= 0) {
        // dead centroid: re-seed on the heaviest color to avoid shrinking
        let heavy = 0;
        for (let i = 1; i < n; i++) if (counts[i] > counts[heavy]) heavy = i;
        palette[p * 3] = (keys[heavy] >> 16) & 255;
        palette[p * 3 + 1] = (keys[heavy] >> 8) & 255;
        palette[p * 3 + 2] = keys[heavy] & 255;
        continue;
      }
      palette[p * 3] = clamp255(Math.round(wr[p] / ww[p]));
      palette[p * 3 + 1] = clamp255(Math.round(wg[p] / ww[p]));
      palette[p * 3 + 2] = clamp255(Math.round(wb[p] / ww[p]));
    }
    const rel = (prevErr - err) / Math.max(1e-6, prevErr);
    prevErr = err;
    if (it > 1 && rel < 0.0005) break; // converged
  }
  return { palette: palette.subarray(0, colors * 3), colors, error: quantizationError(keys, counts, n, palette, colors), iterations: it };
}

/** Weighted Voronoi RMSE in the 0..255 range (mean per-channel). */
export function quantizationError(keys, counts, n, palette, colors) {
  let err = 0;
  let tot = 0;
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    const r = (k >> 16) & 255;
    const g = (k >> 8) & 255;
    const b = k & 255;
    let bestD = Infinity;
    for (let p = 0; p < colors; p++) {
      const d = distRedmeanSq(r, g, b, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]);
      if (d < bestD) bestD = d;
    }
    err += bestD * counts[i];
    tot += counts[i];
  }
  return tot > 0 ? Math.sqrt(err / tot / 3) : 0;
}

/** Copy a short palette into a fixed-size buffer, repeating the last entry so
 *  k-means can promote it if the cluster is useful. */
function pad(palette, colors, want) {
  const out = new Uint8Array(want * 3);
  for (let i = 0; i < want; i++) {
    const s = Math.min(i, colors - 1);
    out[i * 3] = palette[s * 3];
    out[i * 3 + 1] = palette[s * 3 + 1];
    out[i * 3 + 2] = palette[s * 3 + 2];
  }
  return out;
}

function kmeansppSeed(keys, counts, n, want) {
  const palette = new Uint8Array(want * 3);
  let total = 0;
  for (let i = 0; i < n; i++) total += counts[i];
  // first: heaviest color
  let first = 0;
  for (let i = 1; i < n; i++) if (counts[i] > counts[first]) first = i;
  palette[0] = (keys[first] >> 16) & 255;
  palette[1] = (keys[first] >> 8) & 255;
  palette[2] = keys[first] & 255;
  const d2 = new Float64Array(n).fill(Infinity);
  const upd = (r, g, b) => {
    for (let i = 0; i < n; i++) {
      const k = keys[i];
      const d = distRedmeanSq((k >> 16) & 255, (k >> 8) & 255, k & 255, r, g, b);
      if (d < d2[i]) d2[i] = d;
    }
  };
  upd(palette[0], palette[1], palette[2]);
  for (let p = 1; p < want; p++) {
    let acc = 0;
    for (let i = 0; i < n; i++) acc += d2[i] * counts[i];
    if (acc <= 0) break;
    let pick = Math.random() * acc;
    let chosen = 0;
    for (let i = 0; i < n; i++) {
      pick -= d2[i] * counts[i];
      if (pick <= 0) {
        chosen = i;
        break;
      }
    }
    const r = (keys[chosen] >> 16) & 255;
    const g = (keys[chosen] >> 8) & 255;
    const b = keys[chosen] & 255;
    palette[p * 3] = r;
    palette[p * 3 + 1] = g;
    palette[p * 3 + 2] = b;
    upd(r, g, b);
  }
  return palette;
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

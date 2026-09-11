/**
 * `imagequant` — the pngquant/libimagequant-style palette optimizer.
 *
 * Default for ≥64 colors because it is the only method here that optimizes an
 * explicit objective against the weighted histogram:
 *
 *   seed (Wu | octree | median-cut)
 *     └─ repeat:
 *         1. assign every histogram color to its nearest palette entry
 *         2. move each entry to the weighted centroid of its cluster
 *         3. variance reduction — split the clusters carrying the most residual
 *            error along their own widest axis, and recycle entries that lost
 *            all their members, until `colors` slots are actually in use
 *         4. (quality ≥ 55) local search: snap entries onto real histogram
 *            colors when that lowers error — this is what makes UI captures
 *            and pixel art come out with exact text/logo colors
 *
 * `quality` (1..100) scales iterations and which steps run. Error is reported
 * as weighted per-channel RMSE (0..255) so `colors:32` vs `colors:128` can be
 * compared honestly by the target-size search.
 *
 * @module quant/imagequant
 */
import { wu } from './wu.js';
import { medianCut } from './median-cut.js';
import { octree } from './octree.js';
import { distRedmeanSq } from '../core/color.js';

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{
 *   colors?:number, quality?:number, seed?:'wu'|'octree'|'median-cut',
 *   snap?:boolean, maxHistogram?:number, iterations?:number
 * }} [opts]
 */
export function imagequant(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  const quality = Math.max(1, Math.min(100, opts.quality ?? 80));
  const maxHist = opts.maxHistogram ?? (quality > 70 ? 48000 : 20000);
  const { keys, counts, n } = hist.toArrays();
  if (!n) return { palette: new Uint8Array(want * 3), colors: 1, error: 0, iterations: 0 };

  const sel = selectHeavy(keys, counts, n, maxHist);
  const N = sel.n;

  // ---- seed ----
  const seedName = opts.seed || (quality >= 50 ? 'wu' : 'median-cut');
  let seed;
  if (seedName === 'wu') seed = wu(hist, { colors: want });
  else if (seedName === 'octree') seed = octree(hist, { colors: want });
  else seed = medianCut(hist, { colors: want });
  const palette = new Uint8Array(want * 3);
  palette.set(seed.palette.subarray(0, Math.min(palette.length, seed.palette.length)));
  let colors = Math.max(2, Math.min(want, seed.colors || want));
  // Duplicate the tail so every slot starts distinct-ish; the split step below
  // replaces these with real clusters when it finds them useful.
  for (let i = colors; i < want; i++) {
    palette[i * 3] = clamp255(palette[(i - 1) * 3] + (i % 2 ? 6 : -6));
    palette[i * 3 + 1] = clamp255(palette[(i - 1) * 3 + 1] + (i % 3 ? 4 : -4));
    palette[i * 3 + 2] = clamp255(palette[(i - 1) * 3 + 2] + (i % 5 ? 3 : -3));
  }
  colors = want;

  const iterations = Math.max(1, Math.min(24, opts.iterations ?? 3 + Math.round((quality / 100) * 9)));
  const assign = new Int32Array(N);
  const errOf = new Float64Array(want);
  const wOf = new Float64Array(want);
  const cr = new Float64Array(want);
  const cg = new Float64Array(want);
  const cb = new Float64Array(want);
  const count = new Int32Array(want);
  const box = Array.from({ length: want }, () => newBox());

  let prevErr = Infinity;
  let itersRun = 0;

  for (let it = 0; it < iterations; it++) {
    errOf.fill(0);
    wOf.fill(0);
    cr.fill(0);
    cg.fill(0);
    cb.fill(0);
    count.fill(0);
    for (const b of box) resetBox(b);

    // ---- 1. assign ----
    let totalErr = 0;
    for (let i = 0; i < N; i++) {
      const k = sel.keys[i];
      const r = (k >> 16) & 255;
      const g = (k >> 8) & 255;
      const b = k & 255;
      const c = sel.counts[i];
      let best = 0;
      let bestD = Infinity;
      for (let p = 0; p < colors; p++) {
        const d = distRedmeanSq(r, g, b, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]);
        if (d < bestD) {
          bestD = d;
          best = p;
          if (!d) break;
        }
      }
      assign[i] = best;
      errOf[best] += bestD * c;
      wOf[best] += c;
      cr[best] += r * c;
      cg[best] += g * c;
      cb[best] += b * c;
      count[best]++;
      growBox(box[best], r, g, b);
      totalErr += bestD * c;
    }

    const improvement = prevErr === Infinity ? 1 : (prevErr - totalErr) / Math.max(1e-9, prevErr);
    prevErr = totalErr;
    itersRun = it + 1;
    if (it > 0 && improvement < 0.0004 && quality < 95) break;

    // ---- 2. centroids ----
    for (let p = 0; p < colors; p++) {
      if (wOf[p] <= 0) continue;
      palette[p * 3] = clamp255(Math.round(cr[p] / wOf[p]));
      palette[p * 3 + 1] = clamp255(Math.round(cg[p] / wOf[p]));
      palette[p * 3 + 2] = clamp255(Math.round(cb[p] / wOf[p]));
    }

    if (it === iterations - 1) break;

    // ---- 3. variance reduction: grow into unused slots by splitting the
    //          clusters that still carry the most error ----
    if (quality >= 12) {
      const order = [];
      for (let p = 0; p < colors; p++) if (count[p] > 1) order.push({ p, score: (errOf[p] / count[p]) * Math.sqrt(count[p]) });
      order.sort((a, b) => b.score - a.score);
      const free = [];
      for (let p = 0; p < colors; p++) if (count[p] === 0) free.push(p);
      for (const { p } of order) {
        let dst = free.length ? free.pop() : -1;
        if (dst < 0) break; // no slot left to steal
        const b = box[p];
        const ext = [Math.max(0, b.rmax - b.rmin), Math.max(0, b.gmax - b.gmin), Math.max(0, b.bmax - b.bmin)];
        const ax = ext[1] > ext[0] && ext[1] >= ext[2] ? 1 : ext[2] > ext[0] && ext[2] > ext[1] ? 2 : 0;
        if (ext[ax] < 2) continue;
        const off = Math.max(1, Math.round(ext[ax] / 4));
        palette[dst * 3] = palette[p * 3];
        palette[dst * 3 + 1] = palette[p * 3 + 1];
        palette[dst * 3 + 2] = palette[p * 3 + 2];
        if (ax === 0) {
          palette[p * 3] = clamp255(palette[p * 3] - ((off + 1) >> 1));
          palette[dst * 3] = clamp255(palette[dst * 3] + (off >> 1) + 1);
        } else if (ax === 1) {
          palette[p * 3 + 1] = clamp255(palette[p * 3 + 1] - ((off + 1) >> 1));
          palette[dst * 3 + 1] = clamp255(palette[dst * 3 + 1] + (off >> 1) + 1);
        } else {
          palette[p * 3 + 2] = clamp255(palette[p * 3 + 2] - ((off + 1) >> 1));
          palette[dst * 3 + 2] = clamp255(palette[dst * 3 + 2] + (off >> 1) + 1);
        }
      }
    }
  }

  // ---- 4. local search (snap to real colors) ----
  if (opts.snap !== false && quality >= 55) {
    for (let p = 0; p < colors; p++) {
      let cand = -1;
      let candC = -1;
      for (let i = 0; i < N; i++) {
        if (assign[i] !== p) continue;
        if (sel.counts[i] > candC) {
          candC = sel.counts[i];
          cand = i;
        }
      }
      if (cand < 0) continue;
      const k = sel.keys[cand];
      const nr = (k >> 16) & 255;
      const ng = (k >> 8) & 255;
      const nb = k & 255;
      const or = palette[p * 3];
      const og = palette[p * 3 + 1];
      const ob = palette[p * 3 + 2];
      if (nr === or && ng === og && nb === ob) continue;
      let delta = 0;
      for (let i = 0; i < N; i++) {
        if (assign[i] !== p) continue;
        const kk = sel.keys[i];
        const r = (kk >> 16) & 255;
        const g = (kk >> 8) & 255;
        const b = kk & 255;
        delta += (distRedmeanSq(r, g, b, nr, ng, nb) - distRedmeanSq(r, g, b, or, og, ob)) * sel.counts[i];
        if (delta > 0) break;
      }
      if (delta < 0) {
        palette[p * 3] = nr;
        palette[p * 3 + 1] = ng;
        palette[p * 3 + 2] = nb;
      }
    }
  }

  // ---- prune entries that ended up unused, recompute the honest metric ----
  const used = new Uint8Array(colors);
  let totalErr = 0;
  let totalW = 0;
  for (let i = 0; i < N; i++) {
    const p = assign[i];
    used[p] = 1;
    const k = sel.keys[i];
    totalErr += distRedmeanSq((k >> 16) & 255, (k >> 8) & 255, k & 255, palette[p * 3], palette[p * 3 + 1], palette[p * 3 + 2]) * sel.counts[i];
    totalW += sel.counts[i];
  }
  let keep = 0;
  for (let p = 0; p < colors; p++) if (used[p]) keep++;
  const out = new Uint8Array(Math.max(2, keep) * 3);
  let o = 0;
  for (let p = 0; p < colors; p++) {
    if (!used[p]) continue;
    out[o++] = palette[p * 3];
    out[o++] = palette[p * 3 + 1];
    out[o++] = palette[p * 3 + 2];
  }
  if (!o) {
    out[0] = palette[0];
    out[1] = palette[1];
    out[2] = palette[2];
    out[3] = palette[3] || 255;
    out[4] = palette[4] || 255;
    out[5] = palette[5] || 255;
    o = 6;
  }
  return {
    palette: out.subarray(0, o),
    colors: o / 3,
    error: totalW > 0 ? Math.sqrt(totalErr / totalW / 3) : 0,
    iterations: itersRun,
    histogramUsed: N,
    seed: seedName,
  };
}

function newBox() {
  return { rmin: 255, rmax: 0, gmin: 255, gmax: 0, bmin: 255, bmax: 0 };
}
function resetBox(b) {
  b.rmin = b.gmin = b.bmin = 255;
  b.rmax = b.gmax = b.bmax = 0;
}
function growBox(b, r, g, bl) {
  if (r < b.rmin) b.rmin = r;
  if (r > b.rmax) b.rmax = r;
  if (g < b.gmin) b.gmin = g;
  if (g > b.gmax) b.gmax = g;
  if (bl < b.bmin) b.bmin = bl;
  if (bl > b.bmax) b.bmax = bl;
}

/**
 * Keep the heaviest `max` histogram entries and fold the tail into nearby kept
 * cells (down-weighted), so refinement stays ~30 ms on 4K frames while rare but
 * visible colors still vote.
 */
function selectHeavy(keys, counts, n, max) {
  if (n <= max) return { keys, counts, n };
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => counts[b] - counts[a]);
  const kept = order.slice(0, max);
  const outKeys = new Int32Array(max);
  const outCounts = new Float64Array(max);
  for (let i = 0; i < kept.length; i++) {
    outKeys[i] = keys[kept[i]];
    outCounts[i] = counts[kept[i]];
  }
  const bucketOf = (k) => (((k >> 19) & 15) << 8) | (((k >> 11) & 15) << 4) | ((k >> 3) & 15);
  const keptBucket = new Map();
  for (let i = 0; i < kept.length; i++) {
    const bkt = bucketOf(keys[kept[i]]);
    const list = keptBucket.get(bkt);
    if (list) list.push(i);
    else keptBucket.set(bkt, [i]);
  }
  for (let j = max; j < n; j++) {
    const src = keys[order[j]];
    const bkt = bucketOf(src);
    let target = -1;
    for (let db = -1; db <= 1 && target < 0; db++) {
      for (let dg = -1; dg <= 1 && target < 0; dg++) {
        for (let dr = -1; dr <= 1 && target < 0; dr++) {
          const list = keptBucket.get(bkt + db * 256 + dg * 16 + dr);
          if (list) {
            target = list[0];
            if (list.length > 1 && counts[order[j]] > counts[kept[list[0]]]) target = list[1];
          }
        }
      }
    }
    if (target < 0) target = 0;
    outCounts[target] += counts[order[j]] * 0.05;
  }
  return { keys: outKeys, counts: outCounts, n: max };
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

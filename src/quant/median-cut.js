/**
 * Median-cut quantizer (histogram-weighted).
 *
 * Classic Heckbert with three improvements that matter for video:
 *  - split on the channel with the largest *weighted* range, and split at the
 *    weighted median (not the middle index) so a few bright pixels in an
 *    otherwise dark box don't steal a slot;
 *  - boxes are re-ranked by `range * sqrt(weight)` before each split, which is
 *    a cheap stand-in for variance-based splitting;
 *  - final colors are weighted centroids, snapped to full 8-bit.
 *
 * @module quant/median-cut
 */

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{colors:number, prune?:boolean}} opts
 * @returns {{palette:Uint8Array, colors:number, error:number}}
 */
export function medianCut(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  const { keys, counts, n } = hist.toArrays();
  if (!n) {
    const p = new Uint8Array(want * 3);
    return { palette: p, colors: 1, error: 0 };
  }

  // Sort once by luminance-descending per axis is impossible for all three, so
  // sort by packed key (R-major) and use index ranges + per-box sorting later.
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  sortByChannel(keys, idx, 0); // R-major initial order; each split re-sorts its own range

  /** @type {{lo:number, hi:number, w:number, rMin:number,rMax:number,gMin:number,gMax:number,bMin:number,bMax:number, r:number,g:number,b:number}[]} */
  let boxes = [{ lo: 0, hi: n, w: 0, rMin: 255, rMax: 0, gMin: 255, gMax: 0, bMin: 255, bMax: 0, r: 0, g: 0, b: 0 }];
  measure(keys, counts, idx, boxes[0]);

  const target = Math.min(want, n);
  while (boxes.length < target) {
    // pick the box with the biggest (range × sqrt(weight))
    let bi = -1;
    let bestScore = -1;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (b.hi - b.lo < 2) continue;
      const range = Math.max(b.rMax - b.rMin, b.gMax - b.gMin, b.bMax - b.bMin);
      const score = range * Math.sqrt(Math.max(1, b.w));
      if (score > bestScore) {
        bestScore = score;
        bi = i;
      }
    }
    if (bi < 0) break;
    const box = boxes[bi];
    const rRange = box.rMax - box.rMin;
    const gRange = box.gMax - box.gMin;
    const bRange = box.bMax - box.bMin;
    // channel index matches sortByChannel: 0 = R, 1 = G, 2 = B
    const ch = rRange >= gRange && rRange >= bRange ? 0 : gRange >= bRange ? 1 : 2;
    sortByChannel(keys, idx.subarray(box.lo, box.hi), ch);
    // weighted median split
    const half = box.w / 2;
    let acc = 0;
    let mid = box.lo + 1;
    for (let i = box.lo; i < box.hi - 1; i++) {
      acc += counts[idx[i]];
      if (acc >= half) {
        mid = i + 1;
        break;
      }
    }
    if (mid <= box.lo) mid = box.lo + 1;
    if (mid >= box.hi) mid = box.hi - 1;
    const a = { lo: box.lo, hi: mid, w: 0, rMin: 255, rMax: 0, gMin: 255, gMax: 0, bMin: 255, bMax: 0, r: 0, g: 0, b: 0 };
    const b = { lo: mid, hi: box.hi, w: 0, rMin: 255, rMax: 0, gMin: 255, gMax: 0, bMin: 255, bMax: 0, r: 0, g: 0, b: 0 };
    measure(keys, counts, idx, a);
    measure(keys, counts, idx, b);
    boxes.splice(bi, 1, a, b);
  }

  const palette = new Uint8Array(boxes.length * 3);
  let err = 0;
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    palette[i * 3] = clamp255(Math.round(b.r));
    palette[i * 3 + 1] = clamp255(Math.round(b.g));
    palette[i * 3 + 2] = clamp255(Math.round(b.b));
  }
  void err;
  return { palette, colors: boxes.length, error: 0 };
}

function measure(keys, counts, idx, box) {
  let w = 0;
  let rs = 0;
  let gs = 0;
  let bs = 0;
  box.rMin = box.gMin = box.bMin = 255;
  box.rMax = box.gMax = box.bMax = 0;
  for (let i = box.lo; i < box.hi; i++) {
    const j = idx[i];
    const k = keys[j];
    const c = counts[j];
    const r = (k >> 16) & 255;
    const g = (k >> 8) & 255;
    const b = k & 255;
    w += c;
    rs += r * c;
    gs += g * c;
    bs += b * c;
    if (r < box.rMin) box.rMin = r;
    if (r > box.rMax) box.rMax = r;
    if (g < box.gMin) box.gMin = g;
    if (g > box.gMax) box.gMax = g;
    if (b < box.bMin) box.bMin = b;
    if (b > box.bMax) box.bMax = b;
  }
  box.w = w;
  if (w > 0) {
    box.r = rs / w;
    box.g = gs / w;
    box.b = bs / w;
  } else {
    box.r = (box.rMin + box.rMax) / 2;
    box.g = (box.gMin + box.gMax) / 2;
    box.b = (box.bMin + box.bMax) / 2;
  }
}

/** In-place insertion+shell sort — boxes are small (<a few k), and a full
 *  qsort per split was measurably slower than sorting the sub-range directly. */
function sortByChannel(keys, idx, channel) {
  const shift = channel === 0 ? 16 : channel === 1 ? 8 : 0;
  const n = idx.length;
  let gap = 1;
  while (gap * 3 + 1 < n) gap = gap * 3 + 1;
  for (; gap >= 1; gap = (gap / 3) | 0) {
    for (let i = gap; i < n; i++) {
      const vi = idx[i];
      const vk = (keys[vi] >> shift) & 255;
      let j = i;
      while (j >= gap && ((keys[idx[j - gap]] >> shift) & 255) > vk) {
        idx[j] = idx[j - gap];
        j -= gap;
      }
      idx[j] = vi;
    }
  }
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/**
 * Xiaolin Wu's 3-D variance-minimizing quantizer, histogram-driven.
 *
 * The color cube is reduced to 32³ (5 bits/channel); six moment tables plus a
 * weight table are prefix-summed so any box's statistics are O(1). Boxes are
 * then split greedily: pick the box maximizing `variance·√weight` and cut it
 * where the summed child variance is smallest.
 *
 * Why you want it: for gradients, skies, skin and film grain it produces
 * visibly less banding than median cut at the same color count, because its
 * objective is actual squared error rather than box volume.
 *
 * Cost: ~6 × 32³ float64 (≈500 KB, reused) + O(colors × cube) split search —
 * typically 4–9 ms per palette at 128 colors.
 *
 * @module quant/wu
 */

const SIZE = 32;
const N = SIZE;
const CELLS = N * N * N;

/** Reusable tables (allocation per call dominated the sampling budget). */
let T = null;
function tables() {
  if (!T) {
    T = {
      weight: new Float64Array(CELLS + 1),
      r: new Float64Array(CELLS + 1),
      g: new Float64Array(CELLS + 1),
      b: new Float64Array(CELLS + 1),
      rr: new Float64Array(CELLS + 1),
      gg: new Float64Array(CELLS + 1),
      bb: new Float64Array(CELLS + 1),
    };
  }
  return T;
}

const at = (r, g, b) => (r * N + g) * N + b + 1;

/** Inclusive prefix sums along r, g, b (three 1-D passes). */
function prefixSums(m) {
  for (let g = 0; g < N; g++) {
    for (let b = 0; b < N; b++) {
      let s = 0;
      for (let r = 0; r < N; r++) {
        const i = at(r, g, b);
        s += m[i];
        m[i] = s;
      }
    }
  }
  for (let r = 0; r < N; r++) {
    for (let b = 0; b < N; b++) {
      let s = 0;
      for (let g = 0; g < N; g++) {
        const i = at(r, g, b);
        s += m[i];
        m[i] = s;
      }
    }
  }
  for (let r = 0; r < N; r++) {
    for (let g = 0; g < N; g++) {
      let s = 0;
      for (let b = 0; b < N; b++) {
        const i = at(r, g, b);
        s += m[i];
        m[i] = s;
      }
    }
  }
}

/** Inclusion–exclusion over the prefix tables. r1/g1/b1 may be 0. */
function boxAt(m, r1, g1, b1, r2, g2, b2) {
  const A = m[at(r2, g2, b2)];
  const B = r1 > 0 ? m[at(r1 - 1, g2, b2)] : 0;
  const C = g1 > 0 ? m[at(r2, g1 - 1, b2)] : 0;
  const D = b1 > 0 ? m[at(r2, g2, b1 - 1)] : 0;
  const E = r1 > 0 && g1 > 0 ? m[at(r1 - 1, g1 - 1, b2)] : 0;
  const F = r1 > 0 && b1 > 0 ? m[at(r1 - 1, g2, b1 - 1)] : 0;
  const G = g1 > 0 && b1 > 0 ? m[at(r2, g1 - 1, b1 - 1)] : 0;
  const H = r1 > 0 && g1 > 0 && b1 > 0 ? m[at(r1 - 1, g1 - 1, b1 - 1)] : 0;
  return A - B - C - D + E + F + G - H;
}

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{colors:number}} opts
 * @returns {{palette:Uint8Array, colors:number, error:number}}
 */
export function wu(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  const { keys, counts, n } = hist.toArrays();
  const t = tables();
  for (const m of [t.weight, t.r, t.g, t.b, t.rr, t.gg, t.bb]) m.fill(0);
  if (!n) return { palette: new Uint8Array(want * 3), colors: 1, error: 0 };

  // 5-bit bucket = round(i*31/255); `(i*31 + 127) / 255 | 0` avoids a LUT
  // lookup per channel while staying exactly right at the ends.
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    const r8 = (k >> 16) & 255;
    const g8 = (k >> 8) & 255;
    const b8 = k & 255;
    const c = counts[i];
    const slot = at(((r8 * 31 + 127) / 255) | 0, ((g8 * 31 + 127) / 255) | 0, ((b8 * 31 + 127) / 255) | 0);
    t.weight[slot] += c;
    t.r[slot] += r8 * c;
    t.g[slot] += g8 * c;
    t.b[slot] += b8 * c;
    t.rr[slot] += r8 * r8 * c;
    t.gg[slot] += g8 * g8 * c;
    t.bb[slot] += b8 * b8 * c;
  }
  for (const m of [t.weight, t.r, t.g, t.b, t.rr, t.gg, t.bb]) prefixSums(m);

  const stat = (m, x) => boxAt(m, x.r1, x.g1, x.b1, x.r2, x.g2, x.b2);
  const variance = (x) => {
    const w = stat(t.weight, x);
    if (w <= 0) return 0;
    return stat(t.rr, x) + stat(t.gg, x) + stat(t.bb, x) - (stat(t.r, x) ** 2 + stat(t.g, x) ** 2 + stat(t.b, x) ** 2) / w;
  };

  /** @type {{r1:number,g1:number,b1:number,r2:number,g2:number,b2:number}[]} */
  const boxes = [{ r1: 0, g1: 0, b1: 0, r2: N - 1, g2: N - 1, b2: N - 1 }];

  const dead = new Uint8Array(1 << 16);
  let boxesLen = 1;
  while (boxesLen < want) {
    let bi = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < boxes.length; i++) {
      if (dead[i]) continue;
      const x = boxes[i];
      const w = stat(t.weight, x);
      if (w <= 0) {
        dead[i] = 1;
        continue;
      }
      const v = variance(x);
      if (!(v > 1e-6)) {
        dead[i] = 1;
        continue;
      }
      const extent = x.r2 - x.r1 + (x.g2 - x.g1) + (x.b2 - x.b1);
      const score = v * Math.sqrt(w) * (1 + extent / 24);
      if (score > bestScore) {
        bestScore = score;
        bi = i;
      }
    }
    if (bi < 0) break; // nothing splittable left (few distinct colors → palette is exact)
    const box = boxes[bi];
    const dims = [box.r2 - box.r1, box.g2 - box.g1, box.b2 - box.b1];
    const axis = dims[1] > dims[0] && dims[1] >= dims[2] ? 1 : dims[2] > dims[0] && dims[2] > dims[1] ? 2 : 0;
    const lo = axis === 0 ? box.r1 : axis === 1 ? box.g1 : box.b1;
    const hi = axis === 0 ? box.r2 : axis === 1 ? box.g2 : box.b2;
    if (hi - lo < 1) {
      dead[bi] = 1;
      continue;
    }
    const cut = findCut(box, axis, lo, hi);
    if (cut < lo || cut >= hi) {
      // No split along the widest axis improves things: try the other two
      // before giving up on the box, otherwise a wide-but-thin box stalls.
      let alt = -1;
      for (const other of [0, 1, 2]) {
        if (other === axis) continue;
        const l2 = other === 0 ? box.r1 : other === 1 ? box.g1 : box.b1;
        const h2 = other === 0 ? box.r2 : other === 1 ? box.g2 : box.b2;
        if (h2 - l2 < 1) continue;
        const c2 = findCut(box, other, l2, h2);
        if (c2 >= l2 && c2 < h2) {
          alt = other * 1000 + c2;
          break;
        }
      }
      if (alt < 0) {
        dead[bi] = 1;
        continue;
      }
      const otherAxis = (alt / 1000) | 0;
      const c = alt % 1000;
      const left = { ...box };
      const right = { ...box };
      if (otherAxis === 0) {
        left.r2 = c;
        right.r1 = c + 1;
      } else if (otherAxis === 1) {
        left.g2 = c;
        right.g1 = c + 1;
      } else {
        left.b2 = c;
        right.b1 = c + 1;
      }
      boxes.splice(bi, 1, left, right);
      dead.splice ? dead.fill(0, 0, 0) : null; // (dead flags shift; rebuild below)
      remapDead(dead, boxes.length, bi);
      boxesLen++;
      continue;
    }
    const left = { ...box };
    const right = { ...box };
    if (axis === 0) {
      left.r2 = cut;
      right.r1 = cut + 1;
    } else if (axis === 1) {
      left.g2 = cut;
      right.g1 = cut + 1;
    } else {
      left.b2 = cut;
      right.b1 = cut + 1;
    }
    boxes.splice(bi, 1, left, right);
    remapDead(dead, boxes.length, bi);
    boxesLen++;
  }

  const colors = Math.min(want, boxes.length);
  const palette = new Uint8Array(colors * 3);
  let err = 0;
  let totalW = 0;
  const whole = { r1: 0, g1: 0, b1: 0, r2: N - 1, g2: N - 1, b2: N - 1 };
  totalW = stat(t.weight, whole) || 1;
  for (let i = 0; i < colors; i++) {
    const x = boxes[i];
    const w = stat(t.weight, x);
    let r;
    let g;
    let b;
    if (w > 0) {
      r = stat(t.r, x) / w;
      g = stat(t.g, x) / w;
      b = stat(t.b, x) / w;
    } else {
      r = (x.r1 + x.r2 + 1) * 8;
      g = (x.g1 + x.g2 + 1) * 8;
      b = (x.b1 + x.b2 + 1) * 8;
    }
    palette[i * 3] = clamp255(Math.round(r));
    palette[i * 3 + 1] = clamp255(Math.round(g));
    palette[i * 3 + 2] = clamp255(Math.round(b));
    err += Math.max(0, variance(x));
  }
  return { palette, colors, error: Math.sqrt(err / colors) / 442 * 100 };

  /**
   * Coarse-to-fine search for the cut minimizing summed child variance,
   * restricted to cuts that leave mass on both sides (an "empty side" cut is
   * what made the naive version stall forever on thin boxes).
   */
  function findCut(x, ax, l, h) {
    const span = h - l;
    const coarseStep = Math.max(1, Math.floor(span / 20));
    let bestP = -1;
    let bestV = Infinity;
    for (let p = l; p < h; p += coarseStep) {
      const v = cutVariance(x, ax, p);
      if (v < bestV) {
        bestV = v;
        bestP = p;
      }
    }
    if (bestP < 0) return -1;
    const from = Math.max(l, bestP - coarseStep);
    const to = Math.min(h - 1, bestP + coarseStep);
    for (let p = from; p <= to; p++) {
      const v = cutVariance(x, ax, p);
      if (v < bestV) {
        bestV = v;
        bestP = p;
      }
    }
    return bestP;
  }

  function cutVariance(x, ax, p) {
    const l = { ...x };
    const r = { ...x };
    if (ax === 0) {
      l.r2 = p;
      r.r1 = p + 1;
    } else if (ax === 1) {
      l.g2 = p;
      r.g1 = p + 1;
    } else {
      l.b2 = p;
      r.b1 = p + 1;
    }
    // both sides must hold mass, else the split is a no-op that eats a slot
    const wl = stat(t.weight, l);
    const wr = stat(t.weight, r);
    if (wl <= 0 || wr <= 0) return Infinity;
    return variance(l) + variance(r);
  }
}

/**
 * `dead` is indexed by position in `boxes`; a splice at `bi` shifts everything
 * after it by one. Carry the flags along (cheaper than an object field).
 */
function remapDead(dead, newLen, bi) {
  for (let i = newLen - 1; i > bi; i--) dead[i] = dead[i - 1];
  dead[bi] = 0;
  dead[bi + 1] = 0;
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

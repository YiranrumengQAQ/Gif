/**
 * Octree color reduction with level-biased merging.
 *
 * Linear in the number of *distinct* colors, memory-bounded, and unlike median
 * cut it never spends a palette slot on a box containing one outlier color.
 * `palette.method:'auto'` picks it when colors are plentiful but slots are few.
 *
 * Why it is written this way:
 *  - structure-of-arrays typed buffers; 200k nodes as JS objects costs tens of
 *    MB and turns the GC into the bottleneck;
 *  - weight and color sums are accumulated **on the path during insertion**, so
 *    merging a subtree is "forget its children" — no re-walk of colors;
 *  - leaf accounting is computed once after insertion, then maintained exactly
 *    during reduction. Naive incremental accounting is the classic source of
 *    off-by-N bugs (and non-monotonic quality) in octree quantizers;
 *  - node capacity is capped. On overflow a color attaches to the deepest node
 *    that exists: precision loss, never a crash.
 *
 * @module quant/octree
 */

const MAX_LEVEL = 7; // level l consumes 3 bits per channel: 21 bits at l = 6

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{colors:number, depth?:number}} opts
 */
export function octree(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  const depth = Math.max(1, Math.min(MAX_LEVEL, opts.depth ?? MAX_LEVEL));
  const { keys, counts, n } = hist.toArrays();
  if (!n) return { palette: new Uint8Array(want * 3), colors: 1, error: 0 };

  const capacity = Math.max(4096, Math.min(1 << 19, n * depth + 64));
  const child = new Int32Array(capacity * 8).fill(-1);
  const parent = new Int32Array(capacity).fill(-1);
  const lvl = new Int8Array(capacity);
  const weight = new Float64Array(capacity);
  const sr = new Float64Array(capacity);
  const sg = new Float64Array(capacity);
  const sb = new Float64Array(capacity);
  const childCount = new Int16Array(capacity);

  let nodes = 1; // node 0 = root
  let overflow = 0;

  /* ---------------------------- 1. insert ---------------------------- */
  for (let i = 0; i < n; i++) {
    const k = keys[i] >>> 0;
    const r = (k >> 16) & 255;
    const g = (k >> 8) & 255;
    const b = k & 255;
    const c = counts[i];
    let node = 0;
    for (let l = 0; l < depth; l++) {
      const bit = (k >>> (21 - l * 3)) & 7;
      const slot = node * 8 + bit;
      let next = child[slot];
      if (next < 0) {
        if (nodes >= capacity) {
          overflow++;
          break; // attach as deep as we can afford
        }
        next = nodes++;
        child[slot] = next;
        parent[next] = node;
        lvl[next] = l + 1;
      }
      node = next;
      weight[node] += c;
      sr[node] += r * c;
      sg[node] += g * c;
      sb[node] += b * c;
    }
    weight[0] += c;
    sr[0] += r * c;
    sg[0] += g * c;
    sb[0] += b * c;
  }

  /* --------------- 2. leaf accounting + per-level reducible lists --------------- */
  // Nodes were created in increasing depth order along each path, but not
  // globally; a reverse index walk is still correct here because a child is
  // always created after its parent (ids are monotone in creation time).
  const subtreeLeaves = new Int32Array(capacity);
  const levelHead = new Int32Array(MAX_LEVEL + 1).fill(-1);
  const nextInLevel = new Int32Array(capacity).fill(-1);
  let leaves = 0;
  for (let i = nodes - 1; i >= 1; i--) {
    let kids = 0;
    const base = i * 8;
    for (let c = 0; c < 8; c++) if (child[base + c] >= 0) kids++;
    childCount[i] = kids;
    if (kids === 0) {
      subtreeLeaves[i] = 1;
      if (weight[i] > 0) leaves++;
    } else {
      let s = 0;
      for (let c = 0; c < 8; c++) {
        const ch = child[base + c];
        if (ch >= 0) s += subtreeLeaves[ch];
      }
      subtreeLeaves[i] = s;
      // reducible = has children; indexed by its own level for the merge order
      nextInLevel[i] = levelHead[lvl[i]];
      levelHead[lvl[i]] = i;
    }
  }
  if (!leaves) return { palette: new Uint8Array([0, 0, 0]), colors: 1, error: 0, nodes, overflow };

  /* ------------------------------- 3. reduce ------------------------------- */
  // Deepest level first; inside a level, the least-populated subtree first.
  // The /(level+1) bias is Ostromoukhov's: it protects shallow, coarse splits
  // from being merged just because they happen to cover many pixels.
  for (let l = depth - 1; l >= 1 && leaves > want; l--) {
    const cand = [];
    for (let i = levelHead[l]; i >= 0; i = nextInLevel[i]) {
      if (childCount[i] > 0 && weight[i] > 0) cand.push(i);
    }
    if (!cand.length) continue;
    cand.sort((a, b) => weight[a] / (lvl[a] + 1) - weight[b] / (lvl[b] + 1));
    for (const node of cand) {
      if (leaves <= want) break;
      if (childCount[node] <= 0) continue;
      const base = node * 8;
      let swallowed = 0;
      for (let c = 0; c < 8; c++) {
        const ch = child[base + c];
        if (ch >= 0) {
          swallowed += subtreeLeaves[ch];
          child[base + c] = -1;
          childCount[ch] = 0; // tombstone
        }
      }
      childCount[node] = 0;
      subtreeLeaves[node] = 1;
      leaves += 1 - swallowed;
      let up = parent[node];
      while (up >= 0) {
        subtreeLeaves[up] = Math.max(1, subtreeLeaves[up] + 1 - swallowed);
        up = parent[up];
      }
    }
  }

  /* ------------------------------- 4. read out ------------------------------- */
  const keep = [];
  for (let i = 1; i < nodes; i++) if (childCount[i] === 0 && weight[i] > 0) keep.push(i);
  if (!keep.length) keep.push(0);
  keep.sort((a, b) => weight[b] - weight[a]);
  const chosen = keep.slice(0, want);
  const palette = new Uint8Array(chosen.length * 3);
  for (let i = 0; i < chosen.length; i++) {
    const node = chosen[i];
    const w = weight[node] || 1;
    palette[i * 3] = clamp255(Math.round(sr[node] / w));
    palette[i * 3 + 1] = clamp255(Math.round(sg[node] / w));
    palette[i * 3 + 2] = clamp255(Math.round(sb[node] / w));
  }
  return { palette, colors: chosen.length, error: 0, nodes, leaves: chosen.length, overflow };
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

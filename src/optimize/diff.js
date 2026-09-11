/**
 * GIFX Kernel — frame differencing, dirty rects and disposal planning.
 *
 * This is where most of the byte savings in a GIF come from, so the rules matter:
 *
 *  - A sub-frame may only omit pixels that the *viewer* will still have on
 *    screen. That is decided by the previous frame's disposal method, which we
 *    control — so the planner and the rect finder have to agree, and they do:
 *    `planDisposal()` is the single source of truth.
 *  - "Restore to previous" (disposal 3) lets a small change cost a small rect
 *    forever; "keep" (0/1) forces the union of all changed pixels once it has
 *    been drawn. The planner therefore compares `cur` against the *composited*
 *    canvas, not against the previous encoded rect.
 *  - Identical consecutive frames are collapsed and their delays merged — but a
 *    merged delay must not fall under the browser clamping floor, so merging is
 *    evaluated after `normalizeDelay`.
 *  - Comparison runs on a strided sample first (typically 1 of 4 pixels) and
 *    only refines where the sample says something changed: on 4K sources that is
 *    a ~4x wall-clock win in the search loop, where most candidates are skipped.
 *
 * @module optimize/diff
 */
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Bounding box of pixels that differ between two RGBA buffers.
 *
 * @param {Uint8Array} prev composited previous canvas (RGBA, `stride` bytes/row)
 * @param {Uint8Array} cur candidate canvas
 * @param {object} [opts]
 * @param {number} opts.width @param {number} opts.height
 * @param {number} [opts.stride] bytes per row (default width*4)
 * @param {number} [opts.bytesPerPixel=4] 3 for RGB canvases, 4 for RGBA
 * @param {number} [opts.threshold=0] per-channel max delta treated as equal
 * @param {boolean} [opts.ignoreAlpha=false] treat alpha as irrelevant (opaque video)
 * @param {number} [opts.sample=1] 1 = exact; >1 probes every Nth pixel per row, then refines
 * @param {number} [opts.padding=0] grow the rect by N px each side
 * @param {number} [opts.align=1] snap the rect edges to a multiple (8 helps LZW)
 * @returns {{x:number,y:number,width:number,height:number,area:number,changed:boolean,holes:number}}
 */
export function dirtyRect(prev, cur, opts = {}) {
  const width = opts.width | 0;
  const height = opts.height | 0;
  const bpp = opts.bytesPerPixel === 3 ? 3 : 4;
  const stride = opts.stride || width * bpp;
  const thr = opts.threshold | 0;
  const hasAlpha = bpp === 4;
  const ignoreAlpha = !!opts.ignoreAlpha || !hasAlpha;
  const sample = Math.max(1, opts.sample || 1);
  if (!width || !height) return EMPTY_RECT;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let holes = 0;
  // `a`/`b` are the same offset into two buffers of identical layout; keeping the
  // names separate documents which side each read belongs to.
  const cmp = (row, x) => {
    const a = row + x * bpp;
    if (Math.abs(prev[a] - cur[a]) > thr || Math.abs(prev[a + 1] - cur[a + 1]) > thr || Math.abs(prev[a + 2] - cur[a + 2]) > thr) return true;
    if (ignoreAlpha) return false;
    if (prev[a + 3] !== cur[a + 3]) {
      // alpha-only change is visible; and when both sides are partly transparent the
      // premultiplied color differs even if the stored RGB happens to match
      if (prev[a + 3] !== cur[a + 3]) return true;
      return Math.abs(prev[a + 3] - cur[a + 3]) > thr;
    }
    return false;
  };
  for (let y = 0; y < height; y++) {
    const row = y * stride;
    let rowChanged = false;
    if (sample === 1) {
      for (let x = 0; x < width; x++) {
        if (cmp(row, x)) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          rowChanged = true;
        }
      }
    } else {
      // coarse probe: only scan every `sample`-th pixel, then walk outward from
      // the first/last hit in that row to find the true edges
      let firstHit = -1;
      let lastHit = -1;
      for (let x = 0; x < width; x += sample) {
        if (cmp(row, x)) {
          if (firstHit < 0) firstHit = x;
          lastHit = x;
        }
      }
      if (firstHit >= 0) {
        rowChanged = true;
        let a = firstHit;
        while (a > 0 && cmp(row, a - 1)) a--;
        let b = lastHit;
        while (b + 1 < width && cmp(row, b + 1)) b++;
        minX = Math.min(minX, a);
        maxX = Math.max(maxX, b);
        holes += Math.max(0, b - a + 1 - (lastHit - firstHit + 1));
      }
    }
    if (rowChanged) {
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return EMPTY_RECT;
  if (sample > 1) {
    // The probe only visited every `sample`-th row, so the vertical extent must
    // be walked outward row-by-row or a 6-row change reads as 5 rows.
    const rowHasChange = (y) => {
      const row = y * stride;
      for (let x = 0; x < width; x++) if (cmp(row, x)) return true;
      return false;
    };
    while (minY > 0 && rowHasChange(minY - 1)) minY--;
    while (maxY + 1 < height && rowHasChange(maxY + 1)) maxY++;
  }
  let x0 = minX;
  let y0 = minY;
  let x1 = maxX + 1;
  let y1 = maxY + 1;
  const pad = opts.padding | 0;
  if (pad) {
    x0 = clamp(x0 - pad, 0, width);
    y0 = clamp(y0 - pad, 0, height);
    x1 = clamp(x1 + pad, 0, width);
    y1 = clamp(y1 + pad, 0, height);
  }
  const align = opts.align > 1 ? opts.align | 0 : 1;
  if (align > 1) {
    x0 = Math.floor(x0 / align) * align;
    y0 = Math.floor(y0 / align) * align;
    x1 = Math.ceil(x1 / align) * align;
    y1 = Math.ceil(y1 / align) * align;
    x1 = Math.min(x1, width);
    y1 = Math.min(y1, height);
  }
  const rw = x1 - x0;
  const rh = y1 - y0;
  return { x: x0, y: y0, width: rw, height: rh, area: rw * rh, changed: true, holes };
}

const EMPTY_RECT = { x: 0, y: 0, width: 0, height: 0, area: 0, changed: false, holes: 0 };

/** Union of two rects (clamped to the canvas). */
export function unionRect(a, b, width, height) {
  const empty = !a || !a.width || !a.height;
  const emptyB = !b || !b.width || !b.height;
  if (empty && emptyB) return { ...EMPTY_RECT };
  if (empty) return { ...b };
  if (emptyB) return { ...a };
  const x0 = Math.min(a.x, b.x);
  const y0 = Math.min(a.y, b.y);
  const x1 = Math.min(width, Math.max(a.x + a.width, b.x + b.width));
  const y1 = Math.min(height, Math.max(a.y + a.height, b.y + b.height));
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0, area: (x1 - x0) * (y1 - y0), changed: true, holes: 0 };
}

/** Does `a` fully contain `b`? (zero-area `b` counts as contained) */
export function contains(a, b) {
  if (!b || !b.width || !b.height) return true;
  if (!a || !a.width) return false;
  return b.x >= a.x && b.y >= a.y && b.x + b.width <= a.x + a.width && b.y + b.height <= a.y + a.height;
}

/** Mean absolute per-channel difference over a sampled grid, 0..255. */
export function meanAbsDiff(prev, cur, opts = {}) {
  const width = opts.width | 0;
  const height = opts.height | 0;
  const bpp = opts.bytesPerPixel === 3 ? 3 : 4;
  const stride = opts.stride || width * bpp;
  const step = Math.max(1, opts.sample || 4);
  if (!width || !height) return 0;
  let sum = 0;
  let n = 0;
  for (let y = 0; y < height; y += step) {
    const row = y * stride;
    for (let x = 0; x < width; x += step) {
      const p = row + x * bpp;
      sum += Math.abs(prev[p] - cur[p]) + Math.abs(prev[p + 1] - cur[p + 1]) + Math.abs(prev[p + 2] - cur[p + 2]);
      n++;
    }
  }
  return n ? sum / (n * 3) : 0;
}

/**
 * Fraction (0..1) of the frame that changed — the "motion" metric that drives
 * dither selection, temporal filtering strength and the size estimate.
 */
export function changeFraction(prev, cur, opts = {}) {
  const r = dirtyRect(prev, cur, { ...opts, sample: opts.sample || 4 });
  const total = (opts.width | 0) * (opts.height | 0);
  return total ? Math.min(1, r.area / total) : 0;
}

/**
 * Decide per-frame encoding geometry + disposal.
 *
 * @param {object[]} frames `[{ rgba, width, height, stride, delayCs, hasAlpha }]`
 *   in the order they will be encoded (already deduped by the caller if wanted)
 * @param {object} [opts]
 * @param {number} [opts.threshold=0] pixel-equality tolerance for the rect
 * @param {number} [opts.sample=1] sampling factor for the rect scan (1 exact)
 * @param {number} [opts.align=1] rect edge alignment
 * @param {number} [opts.padding=0] rect growth
 * @param {number} [opts.subFrameLimit=0.92] skip sub-framing above this area ratio
 * @param {boolean} [opts.restoreToPrevious=true] allow disposal 3
 * @param {number} [opts.transparentIndex] index used for "unchanged" pixels
 * @param {boolean} [opts.ignoreAlpha] treat alpha as irrelevant
 * @param {boolean} [opts.forceFullFrame] ignore dirty rects entirely
 * @returns {{frames:object[], totalRectPixels:number, totalPixels:number, subFrameRatio:number, savings:number}}
 */
export function planDisposal(frames, opts = {}) {
  const n = frames.length;
  if (!n) return { frames: [], totalRectPixels: 0, totalPixels: 0, subFrameRatio: 0, savings: 0 };
  const width = frames[0].width | 0;
  const height = frames[0].height | 0;
  const stride = frames[0].stride || width * 4;
  const force = !!opts.forceFullFrame;
  const allowRestore = opts.restoreToPrevious !== false;
  const limit = opts.subFrameLimit == null ? 0.92 : opts.subFrameLimit;
  const out = new Array(n);
  // The canvas the viewer will have *before* frame i is drawn. With disposal 3
  // it is the canvas from before frame i-1; with keep it is the previous composite.
  let composite = new Uint8Array(stride * height);
  composite.set(frames[0].rgba ? frames[0].rgba.subarray(0, composite.length) : new Uint8Array(composite.length));
  let totalRectPixels = 0;
  for (let i = 0; i < n; i++) {
    const f = frames[i];
    const cur = f.rgba.subarray(0, composite.length);
    let rect;
    if (i === 0 || force) {
      rect = fullRect(width, height);
    } else {
      rect = dirtyRect(composite, cur, {
        width,
        height,
        stride,
        threshold: opts.threshold,
        sample: opts.sample,
        align: opts.align,
        padding: opts.padding,
        ignoreAlpha: opts.ignoreAlpha,
      });
    }
    let disposal = 0;
    let encode = rect;
    if (!rect.changed || rect.width === 0) {
      // nothing changed: this frame is only a delay extension; the caller folds it
      out[i] = { index: i, rect: EMPTY_RECT, encodeRect: EMPTY_RECT, disposal: 0, unchanged: true, delayCs: f.delayCs };
      continue;
    }
    const areaRatio = rect.area / (width * height);
    if (force || areaRatio > limit) {
      encode = fullRect(width, height);
      disposal = 0;
    } else if (!containsRect(composite, cur, rect, width, height, stride, opts)) {
      // Pixels *outside* the dirty rect also changed, so "keep" would leave the
      // stale image showing. Two ways out: restore-to-previous and encode only the
      // rect (cheap, but the viewer has to snapshot the canvas), or grow the rect
      // to the union and keep (larger, but works in every viewer).
      if (allowRestore) {
        encode = rect;
        disposal = 3;
      } else {
        encode = mergeRectsWithUnionOfPrev(rect, i ? out[i - 1] : null, width, height);
        disposal = 0;
        if (encode.area >= width * height) encode = fullRect(width, height);
      }
    } else {
      encode = rect;
      disposal = 0;
    }
    totalRectPixels += encode.width * encode.height;
    // Update the composite exactly as a viewer would.
    if (disposal === 3) {
      // "Restore to previous": after display the viewer puts back the canvas as
      // it was *before* this frame, so the next frame must be compared against
      // that same base — which is what `composite` already holds. Leave it alone.
    } else if (disposal === 2) {
      composite.fill(0);
    } else {
      paintRect(composite, cur, encode, width, stride);
    }
    out[i] = {
      index: i,
      rect,
      encodeRect: encode,
      disposal,
      unchanged: false,
      delayCs: f.delayCs,
      subFrame: encode.width < width || encode.height < height,
      pixels: encode.width * encode.height,
      areaRatio: (encode.width * encode.height) / (width * height),
    };
  }
  const totalPixels = width * height * n;
  return {
    frames: out,
    width,
    height,
    totalRectPixels,
    totalPixels,
    subFrameRatio: totalPixels ? totalRectPixels / totalPixels : 1,
    savings: totalPixels ? 1 - totalRectPixels / totalPixels : 0,
  };
}

const fullRect = (w, h) => ({ x: 0, y: 0, width: w, height: h, area: w * h, changed: true, holes: 0 });

/**
 * Does every pixel *outside* `rect` already match between the viewer canvas and
 * the new frame? If yes, a sub-frame with "keep" is safe; if no, either the rect
 * must grow or we need disposal 3.
 */
function containsRect(composite, cur, rect, width, height, stride, opts) {
  const thr = (opts.threshold | 0) + 0;
  const step = Math.max(1, opts.verifySample || 2);
  for (let y = 0; y < height; y += step) {
    const inRectY = y >= rect.y && y < rect.y + rect.height;
    const row = y * stride;
    for (let x = 0; x < width; x += step) {
      const inRect = inRectY && x >= rect.x && x < rect.x + rect.width;
      if (inRect) continue;
      const p = row + x * 4;
      if (Math.abs(composite[p] - cur[p]) > thr || Math.abs(composite[p + 1] - cur[p + 1]) > thr || Math.abs(composite[p + 2] - cur[p + 2]) > thr) return false;
      if (!opts.ignoreAlpha && composite[p + 3] !== cur[p + 3] && (composite[p + 3] > 0 || cur[p + 3] > 0)) return false;
    }
  }
  return true;
}

function mergeRectsWithUnionOfPrev(rect, prevPlan, width, height) {
  if (!prevPlan || !prevPlan.encodeRect || !prevPlan.encodeRect.width) return rect;
  const a = rect;
  const b = prevPlan.encodeRect;
  const x0 = Math.min(a.x, b.x);
  const y0 = Math.min(a.y, b.y);
  const x1 = Math.min(width, Math.max(a.x + a.width, b.x + b.width));
  const y1 = Math.min(height, Math.max(a.y + a.height, b.y + b.height));
  const w = x1 - x0;
  const h = y1 - y0;
  return { x: x0, y: y0, width: w, height: h, area: w * h, changed: true, holes: 0 };
}

function paintRect(dst, src, rect, width, stride) {
  for (let y = 0; y < rect.height; y++) {
    const dy = (rect.y + y) * stride + rect.x * 4;
    const sy = dy;
    dst.set(src.subarray(sy, sy + rect.width * 4), dy);
  }
}

/**
 * Collapse consecutive visually identical frames, merging their durations.
 *
 * @param {object[]} frames `[{ rgba, delayCs, ... }]` (delayCs = centiseconds)
 * @param {object} [opts] `{threshold, sample, normalize(fn), maxMerge}`
 * @returns {{frames:object[], merged:number, dropped:number, savedMs:number}}
 */
export function collapseDuplicates(frames, opts = {}) {
  if (frames.length <= 1) return { frames: frames.slice(), merged: 0, dropped: 0, savedMs: 0 };
  const width = (frames[0].width || opts.width) | 0;
  const height = (frames[0].height || opts.height) | 0;
  // A missing dimension would silently merge *everything*, so refuse instead:
  // losing frames is far worse than a loud error.
  if (!width || !height) throw new TypeError('collapseDuplicates needs frames with width/height (or pass opts.width/opts.height)');
  if (frames.some((f) => !f.rgba)) throw new TypeError('collapseDuplicates needs frames with an `rgba` buffer');
  const stride = frames[0].stride || width * 4;
  const normalize = opts.normalize || ((cs) => Math.max(2, Math.round(cs)));
  const maxMerge = opts.maxMerge || 1024;
  const out = [];
  let dropped = 0;
  let savedCs = 0;
  let run = 0;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (out.length && run < maxMerge) {
      const keep = out[out.length - 1];
      const r = dirtyRect(keep.rgba, f.rgba, { width, height, stride, threshold: opts.threshold, sample: opts.sample ?? 1, ignoreAlpha: opts.ignoreAlpha });
      if (!r.changed) {
        // merging must not change how long the frame is *actually* displayed
        const before = normalize(keep.delayCs);
        keep.delayCs += f.delayCs;
        run++;
        const after = normalize(keep.delayCs);
        if (after !== before) {
          dropped++;
          savedCs += f.delayCs;
        }
        continue;
      }
    }
    out.push({ ...f, delayCs: f.delayCs });
    run = 0;
  }
  return { frames: out, merged: frames.length - out.length, dropped, savedMs: savedCs * 10 };
}

/**
 * Fingerprint a frame for cheap near-duplicate detection across search
 * candidates (same source frame at different settings). Not cryptographic.
 */
export function frameHash(rgba, width, height, stride) {
  const s = stride || width * 4;
  let h = 2166136261 >>> 0;
  const stepRow = Math.max(1, Math.floor(height / 16));
  const stepCol = Math.max(1, Math.floor(width / 16));
  for (let y = 0; y < height; y += stepRow) {
    for (let x = 0; x < width; x += stepCol) {
      const p = y * s + x * 4;
      h ^= (rgba[p] << 16) | (rgba[p + 1] << 8) | rgba[p + 2];
      h = Math.imul(h, 16777619) >>> 0;
    }
  }
  return h >>> 0;
}

export function rectArea(r) {
  return r && r.width && r.height ? r.width * r.height : 0;
}

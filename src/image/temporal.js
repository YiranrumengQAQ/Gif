/**
 * GIFX Kernel — temporal (cross-frame) effects.
 *
 * These are the filters that only make sense in an *animation* pipeline, and
 * they all follow one rule: the engine hands each frame to `apply(frame, state)`
 * in decode order, so any state is a small per-stream object created by the
 * matching `create*` factory. No filter may require a second pass — the engine
 * is single-pass streaming by design (memory), except where `needsAllFrames`
 * is set (then the collector keeps rasters and the engine opts in explicitly).
 *
 * @module image/temporal
 */
import { Raster } from '../core/buffers.js';
import { clamp255, luma709 } from '../core/color.js';

/**
 * Resample to a target fps with nearest-frame selection and *exact* output
 * timestamps. Returns the schedule the engine walks: `frames[i] = {srcIndex, pts}`.
 *
 * Why not "drop every Nth frame"? Variable-framerate screen recordings and
 * phone video have irregular gaps; naive decimation makes motion stutter and
 * (worse) mis-times each frame. We pick, for each output slot, the source frame
 * whose *midpoint* is closest to the slot centre, then assign the output delay
 * from the slot boundaries — so total duration is preserved to the millisecond.
 */
export function planFrameRate(sourceTimestamps, targetFps, opts = {}) {
  const n = sourceTimestamps.length;
  if (!n) return { schedule: [], durationMs: 0 };
  const start = opts.startMs ?? sourceTimestamps[0] ?? 0;
  const end = opts.endMs ?? (n > 1 ? sourceTimestamps[n - 1] : start + (opts.lastDurationMs || 1000 / (opts.sourceFps || 30)));
  const fps = Math.max(1, Math.min(70, targetFps || 15));
  const outCount = Math.max(1, Math.round(((end - start) / 1000) * fps));
  const step = (end - start) / outCount;
  const schedule = new Array(outCount);
  let cursor = 0;
  for (let i = 0; i < outCount; i++) {
    const slot = start + step * (i + 0.5);
    // advance while the next frame is closer to the slot than the current one
    while (cursor + 1 < n && Math.abs(sourceTimestamps[cursor + 1] - slot) < Math.abs(sourceTimestamps[cursor] - slot)) cursor++;
    // `dupPolicy: 'skip'` avoids emitting the same source frame twice in a row
    schedule[i] = { index: cursor, ptsMs: start + step * i, durationMs: step, duplicate: false };
    if (opts.dupPolicy === 'skip' && i > 0 && schedule[i - 1].index === cursor) schedule[i].duplicate = true;
  }
  if (opts.dupPolicy === 'skip') {
    const kept = schedule.filter((s) => !s.duplicate);
    if (kept.length) return { schedule: kept, durationMs: end - start, outFps: (kept.length * 1000) / Math.max(1, end - start) };
  }
  return { schedule, durationMs: end - start, outFps: fps };
}

/**
 * Frame-difference statistics, used for (a) duplicate-frame detection before
 * encoding, (b) the `motion` field in progress events, (c) auto-dither.
 */
export function frameDiff(a, b, opts = {}) {
  if (!a || !b) return { mean: 255, max: 255, changed: 1, score: 1, pixels: 0 };
  const w = Math.min(a.width, b.width);
  const h = Math.min(a.height, b.height);
  const da = a.data;
  const db = b.data;
  const step = Math.max(1, opts.step || (w * h > 200000 ? 2 : 1));
  let sum = 0;
  let max = 0;
  let changed = 0;
  let n = 0;
  let sumSq = 0;
  for (let y = 0; y < h; y += step) {
    const ra = y * a.stride;
    const rb = y * b.stride;
    for (let x = 0; x < w; x += step) {
      const ia = ra + x * 4;
      const ib = rb + x * 4;
      const dr = Math.abs(da[ia] - db[ib]);
      const dg = Math.abs(da[ia + 1] - db[ib + 1]);
      const dbb = Math.abs(da[ia + 2] - db[ib + 2]);
      const v = dr > dg ? (dr > dbb ? dr : dbb) : dg > dbb ? dg : dbb;
      // alpha changes count as full changes (transparency flicker is visible)
      const da3 = Math.abs(da[ia + 3] - db[ib + 3]) >> 2;
      const t = v > da3 ? v : da3;
      sum += t;
      sumSq += t * t;
      if (t > max) max = t;
      if (t > (opts.ignore ?? 2)) changed++;
      n++;
    }
  }
  const mean = n ? sum / n : 0;
  return {
    mean,
    max,
    rms: n ? Math.sqrt(sumSq / n) : 0,
    changed: n ? changed / n : 0,
    // 0..1 perceptual-ish score: mean delta normalized, boosted by coverage
    score: Math.min(1, (mean / 40) * 0.65 + (changed / n) * 0.6),
    pixels: n,
    width: w,
    height: h,
  };
}

/** Global-motion estimate (dense-ish, tiny search) for stabilize / trails. */
export function estimateMotion(prev, cur, opts = {}) {
  if (!prev || !cur) return { dx: 0, dy: 0, score: 0, reliable: false };
  const w = Math.min(prev.width, cur.width);
  const h = Math.min(prev.height, cur.height);
  // A search larger than the frame is meaningless (and would skip the loop
  // entirely, silently reporting "no motion").
  const maxShift = Math.max(1, Math.min(opts.maxShift || 16, Math.floor(Math.min(w, h) / 4) || 1, 48));
  const coarse = Math.max(1, Math.round(Math.sqrt((w * h) / 24000)));
  const lumA = lumaSample(prev, w, h, coarse);
  const lumB = lumaSample(cur, w, h, coarse);
  let best = { dx: 0, dy: 0, err: Infinity };
  for (let dy = -maxShift; dy <= maxShift; dy++) {
    for (let dx = -maxShift; dx <= maxShift; dx++) {
      let err = 0;
      let n = 0;
      for (let y = maxShift; y < h - maxShift; y += 2) {
        const ry = y * w;
        const ty = (y + dy) * w;
        for (let x = maxShift; x < w - maxShift; x += 2) {
          const d = lumA[ry + x] - lumB[ty + x + dx];
          err += d * d;
          n++;
        }
      }
      err = n ? err / n : Infinity;
      if (err < best.err) best = { dx, dy, err };
    }
  }
  const varA = variance(lumA);
  return {
    dx: best.dx,
    dy: best.dy,
    score: best.err,
    // flat/blurry frames give garbage vectors; say so instead of jittering
    reliable: varA > 40 && best.err < varA * 0.98,
  };
}

function lumaSample(img, w, h, step) {
  const out = new Float32Array(w * h);
  const d = img.data;
  for (let y = 0; y < h; y++) {
    const row = y * img.stride;
    for (let x = 0; x < w; x++) out[y * w + x] = luma709(d[row + x * 4], d[row + x * 4 + 1], d[row + x * 4 + 2]);
  }
  void step;
  return out;
}

function variance(a) {
  let s = 0;
  let s2 = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i];
    s2 += a[i] * a[i];
  }
  const m = s / a.length;
  return s2 / a.length - m * m;
}

/**
 * Image stabilization (translate-only, low-pass filtered motion). `crop` shrinks
 * the frame to hide the shake-compensation borders.
 */
export function createStabilizer(opts = {}) {
  const strength = opts.strength ?? 0.65;
  const maxShift = opts.maxShift ?? 24;
  return {
    name: 'stabilize',
    prev: null,
    cx: 0,
    cy: 0,
    apply(frame) {
      const m = estimateMotion(this.prev, frame, { maxShift });
      this.prev = frame;
      if (!m.reliable) {
        m.dx = 0;
        m.dy = 0;
      }
      this.cx += (m.dx - this.cx) * strength;
      this.cy += (m.dy - this.cy) * strength;
      // undo the measured motion
      const dx = Math.round(-this.cx + m.dx * 0);
      const dy = Math.round(-this.cy + m.dy * 0);
      const inset = opts.crop === false ? 0 : Math.max(2, Math.ceil(maxShift * 0.6));
      const w = frame.width - inset * 2;
      const h = frame.height - inset * 2;
      if (w < 4 || h < 4) return frame;
      const out = new Raster(w, h, undefined, frame.arena);
      shiftCopy(frame, out, dx + inset, dy + inset);
      return out;
    },
  };
}

function shiftCopy(src, dst, ox, oy) {
  const sd = src.data;
  const dd = dst.data;
  for (let y = 0; y < dst.height; y++) {
    const sy = y + oy;
    if (sy < 0 || sy >= src.height) {
      dd.fill(0, y * dst.stride, y * dst.stride + dst.width * 4);
      continue;
    }
    const srow = sy * src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < dst.width; x++) {
      const sx = x + ox;
      const di = drow + x * 4;
      if (sx < 0 || sx >= src.width) {
        dd[di] = dd[di + 1] = dd[di + 2] = 0;
        dd[di + 3] = 0;
        continue;
      }
      const si = srow + sx * 4;
      dd[di] = sd[si];
      dd[di + 1] = sd[si + 1];
      dd[di + 2] = sd[si + 2];
      dd[di + 3] = sd[si + 3];
    }
  }
}

/**
 * Ghost/trail/echo — the classic GIF "long exposure" look.
 * `decay` <1 per frame; `mode` blends for motion-blur-ish trails, `max` keeps
 * hard echoes (good for cursor trails).
 */
export function createTrails(opts = {}) {
  const decay = opts.decay ?? 0.7;
  const frames = opts.frames ?? 3;
  const mode = opts.mode || 'blend';
  const hist = [];
  return {
    name: 'trails',
    apply(frame) {
      hist.push(frame.clone());
      while (hist.length > frames + 1) hist.shift().release();
      const out = new Raster(frame.width, frame.height, undefined, frame.arena);
      out.data.set(frame.data.subarray(0, out.data.length));
      if (mode === 'max') {
        for (let i = hist.length - 2, k = 1; i >= 0; i--, k++) {
          const wgt = Math.pow(decay, k);
          const hd = hist[i].data;
          const od = out.data;
          for (let j = 0; j < od.length; j += 4) {
            if (hd[j + 3] < 8) continue;
            const lum = luma709(hd[j], hd[j + 1], hd[j + 2]) * wgt;
            const ol = luma709(od[j], od[j + 1], od[j + 2]);
            if (lum > ol) {
              od[j] = clamp255(hd[j]);
              od[j + 1] = clamp255(hd[j + 1]);
              od[j + 2] = clamp255(hd[j + 2]);
            }
          }
        }
        return out;
      }
      let total = 1;
      for (let i = hist.length - 2, k = 1; i >= 0; i--, k++) total += Math.pow(decay, k);
      const od = out.data;
      for (let i = hist.length - 2, k = 1; i >= 0; i--, k++) {
        const wgt = Math.pow(decay, k) / total;
        const hd = hist[i].data;
        for (let j = 0; j < od.length; j += 4) {
          od[j] += hd[j] * wgt;
          od[j + 1] += hd[j + 1] * wgt;
          od[j + 2] += hd[j + 2] * wgt;
          od[j + 3] = clamp255(od[j + 3] + hd[j + 3] * wgt * 0.4);
        }
      }
      return out;
    },
    reset() {
      for (const h of hist) h.release();
      hist.length = 0;
    },
  };
}

/** Frame blend/accumulate — smooths noise ("focus stacking" style). */
export function createAccumulate(weight = 0.35) {
  let prev = null;
  return {
    name: 'accumulate',
    apply(frame) {
      if (!prev) {
        prev = frame.clone();
        return frame;
      }
      const d = frame.data;
      const p = prev.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp255(d[i] * (1 - weight) + p[i] * weight);
        d[i + 1] = clamp255(d[i + 1] * (1 - weight) + p[i + 1] * weight);
        d[i + 2] = clamp255(d[i + 2] * (1 - weight) + p[i + 2] * weight);
        d[i + 3] = clamp255(d[i + 3] * (1 - weight) + p[i + 3] * weight);
      }
      p.set(d.subarray(0, p.length));
      return frame;
    },
    reset() {
      if (prev) prev.release();
      prev = null;
    },
  };
}

/**
 * Crossfade between two clips of *equal* length, or a dissolve into a still.
 * @returns {function(Raster, Raster, number): Raster}
 */
export function crossfade(a, b, t, opts = {}) {
  const w = Math.min(a.width, b.width);
  const h = Math.min(a.height, b.height);
  const out = new Raster(w, h, undefined, a.arena);
  const k = Math.max(0, Math.min(1, t));
  const ease = opts.ease === false ? k : k * k * (3 - 2 * k); // smoothstep
  const ad = a.data;
  const bd = b.data;
  const od = out.data;
  for (let y = 0; y < h; y++) {
    const ra = y * a.stride;
    const rb = y * b.stride;
    const ro = y * out.stride;
    for (let x = 0; x < w; x++) {
      const ia = ra + x * 4;
      const ib = rb + x * 4;
      const io = ro + x * 4;
      od[io] = clamp255(ad[ia] + (bd[ib] - ad[ia]) * ease);
      od[io + 1] = clamp255(ad[ia + 1] + (bd[ib + 1] - ad[ia + 1]) * ease);
      od[io + 2] = clamp255(ad[ia + 2] + (bd[ib + 2] - ad[ia + 2]) * ease);
      od[io + 3] = clamp255(ad[ia + 3] + (bd[ib + 3] - ad[ia + 3]) * ease);
    }
  }
  return out;
}

/**
 * Wipe transition shapes (rect / circle / diamond / blinds / clock). Cheaper and
 * more "GIF-y" than crossfades, and great for product demos.
 */
export function wipe(a, b, t, shape = 'circle', opts = {}) {
  const w = Math.min(a.width, b.width);
  const h = Math.min(a.height, b.height);
  const out = new Raster(w, h, undefined, a.arena);
  const k = Math.max(0, Math.min(1, t));
  const cx = (opts.cx ?? 0.5) * w;
  const cy = (opts.cy ?? 0.5) * h;
  const maxR = Math.hypot(Math.max(cx, w - cx), Math.max(cy, h - cy));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let use;
      switch (shape) {
        case 'rect':
        case 'iris-rect':
          use = Math.abs(x - cx) / Math.max(1, Math.max(cx, w - cx)) <= k && Math.abs(y - cy) / Math.max(1, Math.max(cy, h - cy)) <= k;
          break;
        case 'diamond':
          use = (Math.abs(x - cx) / (w * k || 1) + Math.abs(y - cy) / (h * k || 1)) <= 1;
          break;
        case 'blinds':
          use = (y % Math.max(4, opts.band || 12)) / Math.max(4, opts.band || 12) <= k;
          break;
        case 'left':
          use = x / w <= k;
          break;
        case 'right':
          use = 1 - x / w <= k;
          break;
        case 'up':
          use = 1 - y / h <= k;
          break;
        case 'down':
          use = y / h <= k;
          break;
        case 'clock': {
          const ang = Math.atan2(y - cy, x - cx) + Math.PI;
          use = ang / (2 * Math.PI) <= k;
          break;
        }
        case 'circle':
        default:
          use = Math.hypot(x - cx, y - cy) <= maxR * k;
      }
      const src = use ? b : a;
      const si = y * src.stride + x * 4;
      const di = y * out.stride + x * 4;
      out.data[di] = src.data[si];
      out.data[di + 1] = src.data[si + 1];
      out.data[di + 2] = src.data[si + 2];
      out.data[di + 3] = src.data[si + 3];
    }
  }
  return out;
}

/**
 * Ken Burns (slow zoom/pan) over a *still* — turns screenshots into video.
 * `at` is 0..1 across the clip.
 */
export function kenBurns(still, at, opts = {}) {
  const from = opts.from || { x: 0.5, y: 0.5, zoom: 1 };
  const to = opts.to || { x: 0.5, y: 0.5, zoom: 1.25 };
  const k = Math.max(0, Math.min(1, at));
  const zoom = from.zoom + (to.zoom - from.zoom) * k;
  const cx = from.x + (to.x - from.x) * k;
  const cy = from.y + (to.y - from.y) * k;
  const w = Math.max(1, Math.round(still.width / zoom));
  const h = Math.max(1, Math.round(still.height / zoom));
  const x = Math.max(0, Math.min(still.width - w, Math.round(cx * still.width - w / 2)));
  const y = Math.max(0, Math.min(still.height - h, Math.round(cy * still.height - h / 2)));
  const out = new Raster(still.width, still.height, undefined, still.arena);
  const src = new Raster(w, h, undefined, still.arena);
  shiftCopyCrop(still, src, x, y);
  scaleInto(src, out);
  if (src.arena) src.arena.release(src);
  return out;
}

function shiftCopyCrop(src, dst, ox, oy) {
  for (let y = 0; y < dst.height; y++) {
    const sy = Math.min(src.height - 1, y + oy) * src.stride + ox * 4;
    dst.data.set(src.data.subarray(sy, sy + dst.width * 4), y * dst.stride);
  }
}

function scaleInto(src, dst) {
  const fx = src.width / dst.width;
  const fy = src.height / dst.height;
  for (let y = 0; y < dst.height; y++) {
    const sy = Math.min(src.height - 1, (y * fy) | 0);
    const srow = sy * src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < dst.width; x++) {
      const sx = Math.min(src.width - 1, (x * fx) | 0) * 4;
      dst.data[drow + x * 4] = src.data[srow + sx];
      dst.data[drow + x * 4 + 1] = src.data[srow + sx + 1];
      dst.data[drow + x * 4 + 2] = src.data[srow + sx + 2];
      dst.data[drow + x * 4 + 3] = src.data[srow + sx + 3];
    }
  }
}

/**
 * Boomerang: play forward then reverse. Exposed as a schedule transform so the
 * engine never has to hold the frames.
 */
export function boomerangSchedule(count, opts = {}) {
  const hold = opts.holdFrames || 0;
  const out = [];
  for (let i = 0; i < count; i++) out.push(i);
  for (let i = 0; i < hold; i++) out.push(count - 1);
  for (let i = count - 2; i >= 0; i--) out.push(i);
  for (let i = 0; i < hold; i++) out.push(0);
  if (opts.loopTwice) for (let i = 0, n = out.length; i < n; i++) out.push(out[i]);
  return out;
}

/** Slow-motion / speed-ramp on a per-frame delay basis (no interpolation). */
export function speedSchedule(count, rate = 0.5, opts = {}) {
  if (rate <= 0) throw new Error('speed rate must be > 0');
  const out = [];
  if (rate >= 1) {
    // speed-up: keep every 1/rate-th frame
    const every = rate;
    for (let i = 0; i < count; i++) if (i % Math.max(1, Math.round(every)) === 0 || opts.keepAll) out.push(i);
    return out;
  }
  for (let i = 0; i < count; i++) {
    const repeats = Math.max(1, Math.round(1 / rate));
    for (let r = 0; r < repeats; r++) out.push(i);
  }
  void opts;
  return out;
}

/** Interpolated slow motion (optical-flow-ish: bilinear blend, no mesh). */
export function interpolateFrame(a, b, t) {
  return crossfade(a, b, t, { ease: false });
}

/**
 * Repeat a region (motion-echo for cursors) — cheap effect that looks great in
 * small GIFs: the last N frames' brightest difference is echoed at low alpha.
 */
export function createCursorEcho(opts = {}) {
  const alpha = opts.alpha ?? 0.25;
  const every = opts.every ?? 2;
  let i = 0;
  let prev = null;
  return {
    name: 'cursorEcho',
    apply(frame) {
      i++;
      if (i % every !== 0 || !prev) {
        prev = frame.clone();
        return frame;
      }
      const out = frame.clone();
      const od = out.data;
      const pd = prev.data;
      const fd = frame.data;
      for (let j = 0; j < od.length; j += 4) {
        const d = Math.abs(fd[j] - pd[j]) + Math.abs(fd[j + 1] - pd[j + 1]) + Math.abs(fd[j + 2] - pd[j + 2]);
        if (d < (opts.minDelta || 40)) continue;
        od[j] = clamp255(od[j] + (pd[j] - od[j]) * alpha);
        od[j + 1] = clamp255(od[j + 1] + (pd[j + 1] - od[j + 1]) * alpha);
        od[j + 2] = clamp255(od[j + 2] + (pd[j + 2] - od[j + 2]) * alpha);
      }
      prev.release();
      prev = frame.clone();
      return out;
    },
    reset() {
      if (prev) prev.release();
      prev = null;
    },
  };
}

/** Fade in/out over the whole clip (frame index based). */
export function fadeEnvelope(index, count, fadeIn = 0.1, fadeOut = 0.1, opts = {}) {
  const a = Math.max(0, Math.min(1, index / Math.max(1, count * fadeIn)));
  const b = Math.max(0, Math.min(1, (count - 1 - index) / Math.max(1, count * fadeOut)));
  const k = Math.min(a, b);
  const color = opts.color || null;
  return { alpha: k, color };
}

/** Apply fade on a raster in place. */
export function applyFade(raster, alpha, color) {
  if (alpha >= 1) return raster;
  const d = raster.data;
  const n = raster.height * raster.stride;
  const k = Math.max(0, alpha);
  const ik = 1 - k;
  if (color) {
    for (let i = 0; i < n; i += 4) {
      d[i] = clamp255(d[i] * k + color[0] * ik);
      d[i + 1] = clamp255(d[i + 1] * k + color[1] * ik);
      d[i + 2] = clamp255(d[i + 2] * k + color[2] * ik);
    }
  } else {
    for (let i = 0; i < n; i += 4) d[i + 3] = clamp255(d[i + 3] * k);
  }
  return raster;
}

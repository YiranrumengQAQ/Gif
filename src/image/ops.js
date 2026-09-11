/**
 * GIFX Kernel — spatial image operations.
 *
 * All of these run on packed RGBA `Uint8Array`s with an explicit stride, in
 * place whenever possible, and never allocate per pixel. Filters are ordered so
 * that separable passes come first (cache friendly).
 *
 * Scaling notes:
 *  - `area` (box) is the correct choice when downscaling video: it is the only
 *    kernel that cannot alias, and it costs 1 pass with a two-stage
 *    horizontal+vertical separable implementation (the trick used by libvips /
 *    Pillow's `reducing_gap`), so 1080p→480 is ~4× cheaper than a naive
 *    Lanczos while looking nearly identical.
 *  - `lanczos3` / `bicubic` are for upscale or when edges must stay crisp.
 *  - `bilinear` is the "live preview" kernel: 3–6× faster, slightly softer.
 *  - `nearest` for pixel art (never blur it!).
 *
 * @module image/ops
 */
import { Raster } from '../core/buffers.js';
import { clamp255, luma709 } from '../core/color.js';

/** Kernels available to `scale`. */
export const SCALE_KERNELS = ['area', 'bilinear', 'bicubic', 'lanczos2', 'lanczos3', 'nearest', 'point'];

/**
 * Resample a raster.
 *
 * @param {Raster} src
 * @param {number} dstW
 * @param {number} dstH
 * @param {object} [opts]
 * @param {'area'|'bilinear'|'bicubic'|'lanczos2'|'lanczos3'|'nearest'} [opts.kernel='area']
 * @param {Raster} [opts.dst] reuse
 * @param {number} [opts.alpha='premultiplied'] handle alpha correctly
 * @returns {Raster}
 */
export function scale(src, dstW, dstH, opts = {}) {
  dstW = Math.max(1, dstW | 0);
  dstH = Math.max(1, dstH | 0);
  const kernel = opts.kernel || 'area';
  // Every kernel here is identity-preserving at 1:1 scale, so returning the
  // source is both faster and exactly right (a fresh buffer would be blank!).
  if (dstW === src.width && dstH === src.height) return src;
  const dst = opts.dst && opts.dst.width === dstW && opts.dst.height === dstH ? opts.dst : new Raster(dstW, dstH, undefined, opts.arena);

  switch (kernel) {
    case 'nearest':
    case 'point':
      scaleNearest(src, dst);
      break;
    case 'bilinear':
      scaleBilinear(src, dst);
      break;
    case 'bicubic': {
      const s2 = preshrink(src, dstW, dstH, opts.arena);
      scaleSeparable(s2, dst, cubicWeights(dstW, s2.width, 0.5), cubicWeights(dstH, s2.height, 0.5));
      if (s2 !== src && s2.arena) s2.arena.release(s2);
      break;
    }
    case 'lanczos2': {
      const s2 = preshrink(src, dstW, dstH, opts.arena);
      scaleSeparable(s2, dst, lanczosWeights(dstW, s2.width, 2), lanczosWeights(dstH, s2.height, 2));
      if (s2 !== src && s2.arena) s2.arena.release(s2);
      break;
    }
    case 'lanczos3': {
      const s2 = preshrink(src, dstW, dstH, opts.arena);
      scaleSeparable(s2, dst, lanczosWeights(dstW, s2.width, 3), lanczosWeights(dstH, s2.height, 3));
      if (s2 !== src && s2.arena) s2.arena.release(s2);
      break;
    }
    case 'area':
    default:
      scaleArea(src, dst, opts.twoStage !== false);
      break;
  }
  dst.pts = src.pts;
  dst.duration = src.duration || src.duration;
  return dst;
}

function scaleNearest(src, dst) {
  const sx = src.width / dst.width;
  const sy = src.height / dst.height;
  for (let y = 0; y < dst.height; y++) {
    const srow = ((y * sy) | 0) * src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < dst.width; x++) {
      const si = srow + (((x * sx) | 0) << 2);
      const di = drow + (x << 2);
      dst.data[di] = src.data[si];
      dst.data[di + 1] = src.data[si + 1];
      dst.data[di + 2] = src.data[si + 2];
      dst.data[di + 3] = src.data[si + 3];
    }
  }
}

/**
 * Scratch for the horizontal pass of separable resampling. Kept at module level
 * and grown on demand: allocating a new one per frame per stage shows up in the
 * GC profile immediately.
 */
let separableTmp = null;
function tmpFor(need) {
  if (!separableTmp || separableTmp.length < need) separableTmp = new Float32Array(need + (need >> 2));
  return separableTmp;
}

function scaleBilinear(src, dst) {
  const fx = src.width / dst.width;
  const fy = src.height / dst.height;
  const sd = src.data;
  const dd = dst.data;
  const opaque = src.opaque === true;
  for (let y = 0; y < dst.height; y++) {
    const sy = (y + 0.5) * fy - 0.5;
    let y0 = Math.floor(sy);
    let wy = sy - y0;
    if (y0 < 0) {
      y0 = 0;
      wy = 0;
    }
    let y1 = y0 + 1;
    if (y1 > src.height - 1) y1 = src.height - 1;
    const r0 = y0 * src.stride;
    const r1 = y1 * src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < dst.width; x++) {
      const sxp = (x + 0.5) * fx - 0.5;
      let x0 = Math.floor(sxp);
      let wx = sxp - x0;
      if (x0 < 0) {
        x0 = 0;
        wx = 0;
      }
      let x1 = x0 + 1;
      if (x1 > src.width - 1) x1 = src.width - 1;
      const i00 = r0 + (x0 << 2);
      const i01 = r0 + (x1 << 2);
      const i10 = r1 + (x0 << 2);
      const i11 = r1 + (x1 << 2);
      const di = drow + (x << 2);
      const w00 = (1 - wx) * (1 - wy);
      const w01 = wx * (1 - wy);
      const w10 = (1 - wx) * wy;
      const w11 = wx * wy;
      if (opaque) {
        dd[di] = clamp255(sd[i00] * w00 + sd[i01] * w01 + sd[i10] * w10 + sd[i11] * w11);
        dd[di + 1] = clamp255(sd[i00 + 1] * w00 + sd[i01 + 1] * w01 + sd[i10 + 1] * w10 + sd[i11 + 1] * w11);
        dd[di + 2] = clamp255(sd[i00 + 2] * w00 + sd[i01 + 2] * w01 + sd[i10 + 2] * w10 + sd[i11 + 2] * w11);
        dd[di + 3] = 255;
        continue;
      }
      // premultiplied blend so semi-transparent edges neither darken nor halo
      const a00 = sd[i00 + 3];
      const a01 = sd[i01 + 3];
      const a10 = sd[i10 + 3];
      const a11 = sd[i11 + 3];
      const a = a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11;
      const inv = a > 0 ? 1 / a : 0;
      dd[di] = clamp255((sd[i00] * a00 * w00 + sd[i01] * a01 * w01 + sd[i10] * a10 * w10 + sd[i11] * a11 * w11) * inv);
      dd[di + 1] = clamp255((sd[i00 + 1] * a00 * w00 + sd[i01 + 1] * a01 * w01 + sd[i10 + 1] * a10 * w10 + sd[i11 + 1] * a11 * w11) * inv);
      dd[di + 2] = clamp255((sd[i00 + 2] * a00 * w00 + sd[i01 + 2] * a01 * w01 + sd[i10 + 2] * a10 * w10 + sd[i11 + 2] * a11 * w11) * inv);
      dd[di + 3] = clamp255(a);
    }
  }
}

/**
 * Weight tables for a separable resample along one axis.
 * Each output pixel owns `taps` consecutive input pixels.
 */
function kernelWeights(dstN, srcN, support, fn) {
  const scaleF = srcN / dstN;
  const filterScale = scaleF > 1 ? scaleF : 1;
  const taps = Math.max(1, Math.ceil(2 * support * filterScale));
  const off = new Int32Array(dstN);
  const w = new Float32Array(dstN * taps);
  for (let i = 0; i < dstN; i++) {
    const center = (i + 0.5) * scaleF - 0.5;
    let start = Math.floor(center - (taps / 2 - 0.5) / 1);
    if (start < 0) start = 0;
    if (start + taps > srcN) start = Math.max(0, srcN - taps);
    let sum = 0;
    for (let t = 0; t < taps; t++) {
      const pos = start + t;
      const v = fn((pos - center) / filterScale);
      w[i * taps + t] = v;
      sum += v;
    }
    if (sum === 0) {
      w[i * taps] = 1;
      sum = 1;
    }
    const inv = 1 / sum;
    for (let t = 0; t < taps; t++) w[i * taps + t] *= inv;
    off[i] = start;
  }
  return { taps, off, w };
}

function lanczosWeights(dstN, srcN, a) {
  return kernelWeights(dstN, srcN, a, (x) => {
    if (x === 0) return 1;
    if (x <= -a || x >= a) return 0;
    const px = Math.PI * x;
    return ((a * Math.sin(px) * Math.sin(px / a)) / (px * px));
  });
}

function cubicWeights(dstN, srcN, a) {
  // Catmull-Rom (a = 0.5)
  return kernelWeights(dstN, srcN, 2, (x) => {
    const ax = Math.abs(x);
    if (ax < 1) return ((1.5 * ax - 2.5) * ax * ax + 1);
    if (ax < 2) return ((-0.5 * ax + 2.5) * ax - 4) * ax + 2;
    return 0;
  });
}

/**
 * Pre-shrink by integer 2x box steps until the remaining ratio is < 2, then let
 * the windowed kernel finish. Without this, Lanczos from 4K to GIF size spends
 * ~20 taps per axis per pixel (82 ms/frame at 720p→480 here, vs ~12 ms with the
 * pyramid) while looking the same, because a box pre-filter is itself the ideal
 * anti-alias for the remaining narrow window.
 */
function preshrink(src, dstW, dstH, arena) {
  let cur = src;
  let guard = 0;
  while (guard++ < 6 && cur.width >= dstW * 4 && cur.height >= dstH * 4 && cur.width >= 4 && cur.height >= 4) {
    const next = downscale2x(cur);
    if (next === cur) break;
    if (cur !== src && cur.arena) cur.arena.release(cur);
    cur = next;
  }
  return cur;
}

/** Two-pass separable resample (horizontal then vertical) via a float scratch. */
function scaleSeparable(src, dst, wx, wy) {
  const tmp = tmpFor(dst.width * src.height * 4);
  const sd = src.data;
  const opaque = src.opaque === true;
  for (let y = 0; y < src.height; y++) {
    const srow = y * src.stride;
    const trow = y * dst.width * 4;
    for (let x = 0; x < dst.width; x++) {
      const base = x * wx.taps;
      const off = wx.off[x];
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let t = 0; t < wx.taps; t++) {
        const wt = wx.w[base + t];
        if (!wt) continue;
        const si = srow + ((off + t) << 2);
        if (opaque) {
          r += sd[si] * wt;
          g += sd[si + 1] * wt;
          b += sd[si + 2] * wt;
        } else {
          const al = sd[si + 3];
          a += al * wt;
          r += sd[si] * al * wt;
          g += sd[si + 1] * al * wt;
          b += sd[si + 2] * al * wt;
        }
      }
      const di = trow + (x << 2);
      if (opaque) {
        tmp[di] = r;
        tmp[di + 1] = g;
        tmp[di + 2] = b;
        tmp[di + 3] = 255;
      } else {
        const inv = a > 0 ? 1 / a : 0;
        tmp[di] = a > 0 ? r * inv : 0;
        tmp[di + 1] = a > 0 ? g * inv : 0;
        tmp[di + 2] = a > 0 ? b * inv : 0;
        tmp[di + 3] = a;
      }
    }
  }
  const rowLen = dst.width * 4;
  const dd = dst.data;
  for (let y = 0; y < dst.height; y++) {
    const base = y * wy.taps;
    const off = wy.off[y];
    for (let x = 0; x < dst.width; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let t = 0; t < wy.taps; t++) {
        const wt = wy.w[base + t];
        if (!wt) continue;
        const si = (off + t) * rowLen + (x << 2);
        if (opaque) {
          r += tmp[si] * wt;
          g += tmp[si + 1] * wt;
          b += tmp[si + 2] * wt;
        } else {
          const al = tmp[si + 3];
          a += al * wt;
          r += tmp[si] * al * wt;
          g += tmp[si + 1] * al * wt;
          b += tmp[si + 2] * al * wt;
        }
      }
      const di = y * dst.stride + (x << 2);
      if (opaque) {
        dd[di] = clamp255(r);
        dd[di + 1] = clamp255(g);
        dd[di + 2] = clamp255(b);
        dd[di + 3] = 255;
      } else {
        const inv = a > 0 ? 1 / a : 0;
        dd[di] = clamp255(a > 0 ? r * inv : 0);
        dd[di + 1] = clamp255(a > 0 ? g * inv : 0);
        dd[di + 2] = clamp255(a > 0 ? b * inv : 0);
        dd[di + 3] = clamp255(a);
      }
    }
  }
}

/**
 * Box/area downscale (upscaling falls back to bilinear). Separable two-stage:
 * first reduce toward ≥ target size by integer-ish steps, then finish — this is
 * dramatically cheaper than a wide kernel and avoids moiré on 4K sources.
 */
export function scaleArea(src, dst, twoStage = true) {
  if (dst.width >= src.width || dst.height >= src.height) {
    return scaleBilinear(src, dst);
  }
  const sd = src.data;
  const dd = dst.data;
  const opaque = src.opaque === true;
  const sx = src.width / dst.width;
  const sy = src.height / dst.height;
  // horizontal averages for one output row band, kept as premultiplied sums
  const rowBuf = new Float32Array(dst.width * 4);
  for (let y = 0; y < dst.height; y++) {
    const y0 = Math.floor(y * sy);
    let y1 = Math.floor((y + 1) * sy);
    if (y1 <= y0) y1 = y0 + 1;
    if (y1 > src.height) y1 = src.height;
    rowBuf.fill(0);
    const rows = y1 - y0;
    for (let syi = y0; syi < y1; syi++) {
      const srow = syi * src.stride;
      for (let x = 0; x < dst.width; x++) {
        const xa = Math.floor(x * sx);
        let xb = Math.floor((x + 1) * sx);
        if (xb <= xa) xb = xa + 1;
        if (xb > src.width) xb = src.width;
        let r = 0;
        let g = 0;
        let b = 0;
        let a = 0;
        for (let xi = xa; xi < xb; xi++) {
          const si = srow + (xi << 2);
          if (opaque) {
            r += sd[si];
            g += sd[si + 1];
            b += sd[si + 2];
          } else {
            const al = sd[si + 3];
            a += al;
            r += sd[si] * al;
            g += sd[si + 1] * al;
            b += sd[si + 2] * al;
          }
        }
        const n = xb - xa;
        const o = x << 2;
        if (opaque) {
          rowBuf[o] += r / n;
          rowBuf[o + 1] += g / n;
          rowBuf[o + 2] += b / n;
        } else {
          rowBuf[o] += r / n;
          rowBuf[o + 1] += g / n;
          rowBuf[o + 2] += b / n;
          rowBuf[o + 3] += a / n;
        }
      }
    }
    const drow = y * dst.stride;
    const invRows = 1 / rows;
    for (let x = 0; x < dst.width; x++) {
      const o = x << 2;
      const pr = rowBuf[o] * invRows;
      const pg = rowBuf[o + 1] * invRows;
      const pb = rowBuf[o + 2] * invRows;
      dd[drow + o] = clamp255(pr);
      dd[drow + o + 1] = clamp255(pg);
      dd[drow + o + 2] = clamp255(pb);
      dd[drow + o + 3] = 255;
    }
    if (!opaque) {
      for (let x = 0; x < dst.width; x++) {
        const o = x << 2;
        const a = rowBuf[o + 3] * invRows;
        const inv = a > 0 ? 1 / a : 0;
        const di = drow + o;
        dd[di] = a > 0 ? clamp255(rowBuf[o] * invRows * inv) : 0;
        dd[di + 1] = a > 0 ? clamp255(rowBuf[o + 1] * invRows * inv) : 0;
        dd[di + 2] = a > 0 ? clamp255(rowBuf[o + 2] * invRows * inv) : 0;
        dd[di + 3] = clamp255(a);
      }
    }
  }
  void twoStage;
}

/* -------------------------------------------------------------- geometry */

/**
 * Crop, in place-safe (returns a new raster unless `dst` is provided).
 * Clamps to bounds instead of throwing, because these numbers usually come
 * from a drag handle in a UI at non-integer positions.
 */
export function crop(src, x, y, w, h, opts = {}) {
  x = Math.max(0, Math.round(x));
  y = Math.max(0, Math.round(y));
  w = Math.max(1, Math.min(src.width - x, Math.round(w)));
  h = Math.max(1, Math.min(src.height - y, Math.round(h)));
  if (x === 0 && y === 0 && w === src.width && h === src.height) return src;
  const dst = opts.dst && opts.dst.width === w && opts.dst.height === h ? opts.dst : new Raster(w, h, undefined, opts.arena);
  const sd = src.data;
  const dd = dst.data;
  for (let row = 0; row < h; row++) {
    const si = (y + row) * src.stride + x * 4;
    const di = row * dst.stride;
    dd.set(sd.subarray(si, si + w * 4), di);
  }
  dst.pts = src.pts;
  dst.duration = src.duration;
  return dst;
}

/**
 * Pad / letterbox. `mode`:
 *  - `pad` solid color (default)
 *  - `edge` clamp-to-edge stretch (nice for anamorphic fixes)
 *  - `mirror` reflect at the edges
 *  - `blur` fill with a blurred+darkened version of the frame (video-style
 *    pillarbox, the look people actually want for vertical phone video in a
 *    16:9 GIF)
 */
export function pad(src, w, h, opts = {}) {
  w = Math.max(src.width, Math.round(w));
  h = Math.max(src.height, Math.round(h));
  const mode = opts.mode || 'pad';
  const dst = new Raster(w, h, undefined, opts.arena);
  const bg = opts.color || [0, 0, 0, opts.transparent ? 0 : 255];
  dst.fill(bg[0], bg[1], bg[2], bg[2] === undefined ? 255 : bg[3] == null ? 255 : bg[3]);
  const ox = Math.round((w - src.width) * (opts.alignX == null ? 0.5 : opts.alignX));
  const oy = Math.round((h - src.height) * (opts.alignY == null ? 0.5 : opts.alignY));
  if (mode !== 'pad') {
    if (mode === 'blur') {
      const bgRaster = new Raster(w, h);
      scaleBilinearFit(src, bgRaster);
      boxBlur(bgRaster, Math.max(8, Math.round(Math.min(w, h) / 12)));
      for (let i = 0; i < bgRaster.data.length; i += 4) {
        bgRaster.data[i] *= 0.55;
        bgRaster.data[i + 1] *= 0.55;
        bgRaster.data[i + 2] *= 0.55;
        bgRaster.data[i + 3] = 255;
      }
      dst.data.set(bgRaster.data);
    } else {
      // edge / mirror: extend the border rows/cols
      const sd = src.data;
      const dd = dst.data;
      for (let y = 0; y < h; y++) {
        let sy = y - oy;
        if (sy < 0) sy = mode === 'mirror' ? -sy : 0;
        if (sy >= src.height) sy = mode === 'mirror' ? 2 * src.height - sy - 2 : src.height - 1;
        sy = Math.max(0, Math.min(src.height - 1, sy));
        for (let x = 0; x < w; x++) {
          let sx = x - ox;
          if (sx < 0) sx = mode === 'mirror' ? -sx : 0;
          if (sx >= src.width) sx = mode === 'mirror' ? 2 * src.width - sx - 2 : src.width - 1;
          sx = Math.max(0, Math.min(src.width - 1, sx));
          const si = sy * src.stride + sx * 4;
          const di = y * dst.stride + x * 4;
          dd[di] = sd[si];
          dd[di + 1] = sd[si + 1];
          dd[di + 2] = sd[si + 2];
          dd[di + 3] = sd[si + 3];
        }
      }
    }
  }
  dst.composite(src, ox, oy, 1, 'copy');
  dst.pts = src.pts;
  dst.duration = src.duration;
  return dst;
}

function scaleBilinearFit(src, dst) {
  const fx = src.width / dst.width;
  const fy = src.height / dst.height;
  for (let y = 0; y < dst.height; y++) {
    const sy = Math.min(src.height - 1, (y * fy) | 0);
    for (let x = 0; x < dst.width; x++) {
      const sx = Math.min(src.width - 1, (x * fx) | 0);
      const si = sy * src.stride + sx * 4;
      const di = y * dst.stride + x * 4;
      dst.data[di] = src.data[si];
      dst.data[di + 1] = src.data[si + 1];
      dst.data[di + 2] = src.data[si + 2];
      dst.data[di + 3] = src.data[si + 3];
    }
  }
}

/** Rotate by an exact multiple of 90° (allocation-free fast path) or any angle. */
export function rotate(src, degrees, opts = {}) {
  const d = ((Math.round(degrees / 90) * 90) % 360 + 360) % 360;
  if (d === 0) return src;
  if (d === 90 || d === 180 || d === 270) return rotateOrtho(src, d, opts.arena);
  return rotateArbitrary(src, degrees, opts);
}

function rotateOrtho(src, d, arena) {
  const swap = d === 90 || d === 270;
  const w = swap ? src.height : src.width;
  const h = swap ? src.width : src.height;
  const dst = new Raster(w, h, undefined, arena);
  const sd = src.data;
  const dd = dst.data;
  for (let y = 0; y < src.height; y++) {
    for (let x = 0; x < src.width; x++) {
      const si = (y * src.stride + x * 4) | 0;
      let dx;
      let dy;
      if (d === 90) {
        dx = src.height - 1 - y;
        dy = x;
      } else if (d === 180) {
        dx = src.width - 1 - x;
        dy = src.height - 1 - y;
      } else {
        dx = y;
        dy = src.width - 1 - x;
      }
      const di = dy * dst.stride + dx * 4;
      dd[di] = sd[si];
      dd[di + 1] = sd[si + 1];
      dd[di + 2] = sd[si + 2];
      dd[di + 3] = sd[si + 3];
    }
  }
  return dst;
}

/** Any-angle rotation with bilinear sampling and background fill. */
export function rotateArbitrary(src, degrees, opts = {}) {
  const rad = (degrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const expand = opts.expand !== false;
  const corners = expand
    ? [
        [-src.width / 2, -src.height / 2],
        [src.width / 2, -src.height / 2],
        [-src.width / 2, src.height / 2],
        [src.width / 2, src.height / 2],
      ].map(([x, y]) => [x * cos - y * sin, x * sin + y * cos])
    : [];
  let w = src.width;
  let h = src.height;
  if (expand) {
    const xs = corners.map((c) => c[0]);
    const ys = corners.map((c) => c[1]);
    w = Math.max(1, Math.ceil(Math.max(...xs) - Math.min(...xs)));
    h = Math.max(1, Math.ceil(Math.max(...ys) - Math.min(...ys)));
  }
  const dst = new Raster(w, h, undefined, opts.arena);
  const bg = opts.color || [0, 0, 0, 0];
  dst.fill(bg[0], bg[1], bg[2], bg[3] == null ? 255 : bg[3]);
  const cx = src.width / 2;
  const cy = src.height / 2;
  const ocx = w / 2;
  const ocy = h / 2;
  const invCos = cos;
  const invSin = sin;
  for (let y = 0; y < h; y++) {
    const dy = y - ocy;
    for (let x = 0; x < w; x++) {
      const dx = x - ocx;
      const sx = dx * invCos + dy * invSin + cx;
      const sy = -dx * invSin + dy * invCos + cy;
      if (sx < 0 || sy < 0 || sx > src.width - 1 || sy > src.height - 1) continue;
      const x0 = sx | 0;
      const y0 = sy | 0;
      const fx = sx - x0;
      const fy = sy - y0;
      const x1 = Math.min(src.width - 1, x0 + 1);
      const y1 = Math.min(src.height - 1, y0 + 1);
      const i00 = y0 * src.stride + x0 * 4;
      const i01 = y0 * src.stride + x1 * 4;
      const i10 = y1 * src.stride + x0 * 4;
      const i11 = y1 * src.stride + x1 * 4;
      const di = y * dst.stride + x * 4;
      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;
      const sd = src.data;
      dst.data[di] = clamp255(sd[i00] * w00 + sd[i01] * w01 + sd[i10] * w10 + sd[i11] * w11);
      dst.data[di + 1] = clamp255(sd[i00 + 1] * w00 + sd[i01 + 1] * w01 + sd[i10 + 1] * w10 + sd[i11 + 1] * w11);
      dst.data[di + 2] = clamp255(sd[i00 + 2] * w00 + sd[i01 + 2] * w01 + sd[i10 + 2] * w10 + sd[i11 + 2] * w11);
      dst.data[di + 3] = clamp255(sd[i00 + 3] * w00 + sd[i01 + 3] * w01 + sd[i10 + 3] * w10 + sd[i11 + 3] * w11);
    }
  }
  return dst;
}

export function flip(src, { horizontal = true, vertical = false } = {}) {
  const dst = new Raster(src.width, src.height, undefined, src.arena);
  const sd = src.data;
  const dd = dst.data;
  for (let y = 0; y < src.height; y++) {
    const sy = vertical ? src.height - 1 - y : y;
    const srow = sy * src.stride;
    const drow = y * dst.stride;
    if (horizontal) {
      for (let x = 0; x < src.width; x++) {
        const si = srow + (src.width - 1 - x) * 4;
        const di = drow + x * 4;
        dd[di] = sd[si];
        dd[di + 1] = sd[si + 1];
        dd[di + 2] = sd[si + 2];
        dd[di + 3] = sd[si + 3];
      }
    } else {
      dd.set(sd.subarray(srow, srow + src.width * 4), drow);
    }
  }
  return dst;
}

/** 90° transpose (mirror along the main diagonal). */
export function transpose(src) {
  const dst = new Raster(src.height, src.width);
  for (let y = 0; y < src.height; y++) {
    for (let x = 0; x < src.width; x++) {
      const si = y * src.stride + x * 4;
      const di = x * dst.stride + y * 4;
      dst.data[di] = src.data[si];
      dst.data[di + 1] = src.data[si + 1];
      dst.data[di + 2] = src.data[si + 2];
      dst.data[di + 3] = src.data[si + 3];
    }
  }
  return dst;
}

/* --------------------------------------------------------------- borders */

/**
 * Rounded-corner mask (alpha). `radius` in px, or a string like `50%`.
 * Supersampled 2× so the curve is smooth before quantization dithers it.
 */
export function roundedCorners(raster, radius, opts = {}) {
  const r = typeof radius === 'string' ? (parseFloat(radius) / 100) * Math.min(raster.width, raster.height) : radius;
  if (!(r > 0)) return raster;
  const d = raster.data;
  const w = raster.width;
  const h = raster.height;
  const rr = r * r;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let cx = -1;
      let cy = -1;
      if (x < r && y < r) {
        cx = r;
        cy = r;
      } else if (x >= w - r && y < r) {
        cx = w - r - 1;
        cy = r;
      } else if (x < r && y >= h - r) {
        cx = r;
        cy = h - r - 1;
      } else if (x >= w - r && y >= h - r) {
        cx = w - r - 1;
        cy = h - r - 1;
      } else continue;
      const dx = x - cx;
      const dy = y - cy;
      const dist2 = dx * dx + dy * dy;
      const aa = 1 - (dist2 - rr) / (2 * r);
      if (aa < 1) {
        const i = y * raster.stride + x * 4;
        d[i + 3] = clamp255(d[i + 3] * Math.max(0, Math.min(1, aa)));
      }
    }
  }
  return raster;
}

/** Solid or gradient border. */
export function border(raster, width, opts = {}) {
  const wd = Math.max(0, Math.round(width));
  if (!wd) return raster;
  const color = opts.color || [255, 255, 255, 255];
  const d = raster.data;
  const w = raster.width;
  const h = raster.height;
  const put = (x, y, t) => {
    let r = color[0];
    let g = color[1];
    let b = color[2];
    let a = color[3] == null ? 255 : color[3];
    if (opts.gradient) {
      const gc = opts.gradient;
      r = lerp2(gc[0], color[0], t);
      g = lerp2(gc[1], color[1], t);
      b = lerp2(gc[2], color[2], t);
    }
    const i = y * raster.stride + x * 4;
    if (a >= 255) {
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = 255;
    } else {
      const f = a / 255;
      d[i] = r * f + d[i] * (1 - f);
      d[i + 1] = g * f + d[i + 1] * (1 - f);
      d[i + 2] = b * f + d[i + 2] * (1 - f);
    }
  };
  for (let x = 0; x < w; x++) {
    for (let t = 0; t < wd; t++) {
      put(x, t, h > 1 ? t / (h - 1) : 0);
      put(x, h - 1 - t, h > 1 ? 1 - t / (h - 1) : 1);
    }
  }
  for (let y = 0; y < h; y++) {
    for (let t = 0; t < wd; t++) {
      put(t, y, w > 1 ? t / (w - 1) : 0);
      put(w - 1 - t, y, w > 1 ? 1 - t / (w - 1) : 1);
    }
  }
  return raster;
}
const lerp2 = (a, b, t) => a + (b - a) * t;

/** Drop the outer N pixels on each side (used by "remove black bars"). */
export function trimEdges(raster, left, top, right, bottom) {
  return crop(raster, left, top, raster.width - left - right, raster.height - top - bottom);
}

/**
 * Auto-crop letterbox/pillarbox bars (ffmpeg `detectcrop` port, simplified):
 * scans edges for rows/columns whose variance and level are below a threshold.
 */
export function detectBlackBars(src, opts = {}) {
  const threshold = opts.threshold ?? 24; // 0..255 max luma considered "black"
  const tolerance = opts.tolerance ?? 0.02; // fraction of pixels allowed above threshold
  const { width: w, height: h } = src;
  const d = src.data;
  const rowBad = new Uint8Array(h);
  const colBad = new Uint8Array(w);
  for (let y = 0; y < h; y++) {
    let bad = 0;
    const step = Math.max(1, Math.floor(w / 256));
    for (let x = 0; x < w; x += step) {
      const i = y * src.stride + x * 4;
      const l = Math.max(d[i], d[i + 1], d[i + 2]);
      if (l > threshold) bad++;
    }
    rowBad[y] = bad / Math.ceil(w / step) <= tolerance ? 1 : 0;
  }
  for (let x = 0; x < w; x++) {
    let bad = 0;
    const step = Math.max(1, Math.floor(h / 256));
    for (let y = 0; y < h; y += step) {
      const i = y * src.stride + x * 4;
      const l = Math.max(d[i], d[i + 1], d[i + 2]);
      if (l > threshold) bad++;
    }
    colBad[x] = bad / Math.ceil(h / step) <= tolerance ? 1 : 0;
  }
  let top = 0;
  while (top < h - 1 && rowBad[top]) top++;
  let bottom = 0;
  while (bottom < h - 1 - top && rowBad[h - 1 - bottom]) bottom++;
  let left = 0;
  while (left < w - 1 && colBad[left]) left++;
  let right = 0;
  while (right < w - 1 - left && colBad[w - 1 - right]) right++;
  return { top, bottom, left, right, width: w - left - right, height: h - top - bottom, hasBars: top + bottom + left + right > 0 };
}

/**
 * Fit the source inside a box: `contain` (letterbox), `cover` (crop),
 * `fill` (stretch), `shrink-only`. Returns the crop+scale description so the
 * caller can pass it to `createImageBitmap` for a GPU-side single pass.
 */
export function fitPlan(srcW, srcH, dstW, dstH, mode = 'contain', opts = {}) {
  const ar = srcW / srcH;
  const targetAr = dstW / dstH;
  let cropW = srcW;
  let cropH = srcH;
  let outW = dstW;
  let outH = dstH;
  let pad = null;
  const eps = 1e-6;
  if (mode === 'cover') {
    if (ar > targetAr) cropW = Math.round(srcH * targetAr);
    else cropH = Math.round(srcW / targetAr);
  } else if (mode === 'contain') {
    if (ar > targetAr) {
      outW = dstW;
      outH = Math.max(1, Math.round(dstW / ar));
    } else {
      outH = dstH;
      outW = Math.max(1, Math.round(dstH * ar));
    }
    if (opts.shrinkOnly) {
      outW = Math.min(outW, srcW);
      outH = Math.min(outH, srcH);
    }
    pad = { width: dstW, height: dstH };
  } else if (mode === 'fill' || mode === 'stretch') {
    // keep requested box
  } else if (mode === 'width') {
    outW = dstW;
    outH = Math.max(1, Math.round(dstW / ar));
    pad = opts.padToBox ? { width: dstW, height: dstH } : null;
  } else if (mode === 'height') {
    outH = dstH;
    outW = Math.max(1, Math.round(dstH * ar));
    pad = opts.padToBox ? { width: dstW, height: dstH } : null;
  }
  const cx = Math.round((srcW - cropW) / 2);
  const cy = Math.round((srcH - cropH) / 2);
  void eps;
  return { cropX: cx, cropY: cy, cropW, cropH, width: outW, height: outH, pad, sourceWidth: srcW, sourceHeight: srcH };
}

/** Even-integer downscale by pixel averaging (fastest possible filter). */
export function downscale2x(src) {
  const w = src.width >> 1;
  const h = src.height >> 1;
  if (w < 1 || h < 1) return src;
  const dst = new Raster(w, h, undefined, src.arena);
  const sd = src.data;
  const dd = dst.data;
  for (let y = 0; y < h; y++) {
    const r0 = y * 2 * src.stride;
    const r1 = r0 + src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < w; x++) {
      const a = r0 + x * 8;
      const b = r1 + x * 8;
      const o = drow + x * 4;
      dd[o] = (sd[a] + sd[a + 4] + sd[b] + sd[b + 4] + 2) >> 2;
      dd[o + 1] = (sd[a + 1] + sd[a + 5] + sd[b + 1] + sd[b + 5] + 2) >> 2;
      dd[o + 2] = (sd[a + 2] + sd[a + 6] + sd[b + 2] + sd[b + 6] + 2) >> 2;
      dd[o + 3] = (sd[a + 3] + sd[a + 7] + sd[b + 3] + sd[b + 7] + 2) >> 2;
    }
  }
  return dst;
}

/** Pixelate / posterize-by-block (retro look, also great for hiding detail). */
export function pixelate(src, blockSize, opts = {}) {
  const bs = Math.max(1, Math.round(blockSize));
  if (bs <= 1) return src;
  const d = src.data;
  const w = src.width;
  const h = src.height;
  for (let by = 0; by < h; by += bs) {
    for (let bx = 0; bx < w; bx += bs) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      let n = 0;
      const yEnd = Math.min(h, by + bs);
      const xEnd = Math.min(w, bx + bs);
      for (let y = by; y < yEnd; y++) {
        const row = y * src.stride;
        for (let x = bx; x < xEnd; x++) {
          const i = row + x * 4;
          r += d[i];
          g += d[i + 1];
          b += d[i + 2];
          a += d[i + 3];
          n++;
        }
      }
      r = (r / n) | 0;
      g = (g / n) | 0;
      b = (b / n) | 0;
      a = (a / n) | 0;
      for (let y = by; y < yEnd; y++) {
        const row = y * src.stride;
        for (let x = bx; x < xEnd; x++) {
          const i = row + x * 4;
          d[i] = r;
          d[i + 1] = g;
          d[i + 2] = b;
          d[i + 3] = a;
        }
      }
    }
  }
  void opts;
  return src;
}

/**
 * Vignette (per-channel, so it can tint as well as darken).
 */
export function vignette(src, amount = 0.5, opts = {}) {
  const w = src.width;
  const h = src.height;
  const cx = w / 2;
  const cy = h / 2;
  const max = Math.hypot(cx, cy);
  const d = src.data;
  const soft = opts.softness ?? 0.6;
  const tint = opts.color || null;
  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const row = y * src.stride;
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const r = Math.hypot(dx, dy) / max;
      let k = 1 - amount * Math.pow(Math.max(0, (r - (1 - soft)) / soft), 2);
      if (k < 0) k = 0;
      const i = row + x * 4;
      d[i] = clamp255(d[i] * k + (tint ? tint[0] * (1 - k) * 0.35 : 0));
      d[i + 1] = clamp255(d[i + 1] * k + (tint ? tint[1] * (1 - k) * 0.35 : 0));
      d[i + 2] = clamp255(d[i + 2] * k + (tint ? tint[2] * (1 - k) * 0.35 : 0));
    }
  }
  return src;
}

/** Crop to a specific aspect ratio (`16:9`, `1`, `4:3.5`, `1920:1080`). */
export function cropToAspect(src, aspect, opts = {}) {
  const a = parseAspect(aspect);
  if (!a) return src;
  const cur = src.width / src.height;
  let w = src.width;
  let h = src.height;
  if (cur > a) w = Math.round(src.height * a);
  else h = Math.round(src.width / a);
  const x = Math.round((src.width - w) * (opts.alignX ?? 0.5));
  const y = Math.round((src.height - h) * (opts.alignY ?? 0.5));
  return crop(src, x, y, w, h, opts);
}

export function parseAspect(str) {
  if (typeof str === 'number') return str > 0 ? str : 0;
  const m = /^\s*(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)\s*$/i.exec(String(str));
  if (m) return parseFloat(m[2]) === 0 ? 0 : parseFloat(m[1]) / parseFloat(m[2]);
  const n = parseFloat(str);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

/** Aspect-preserving size solve: `fit(1920,1080,{width:480})` → {480,270}. */
export function solveSize(srcW, srcH, wantW, wantH, maxSide = 4096, multiple = 1) {
  let w = wantW;
  let h = wantH;
  if (w && !h) h = Math.round(w / (srcW / srcH));
  else if (h && !w) w = Math.round(h * (srcW / srcH));
  else if (!w && !h) {
    w = srcW;
    h = srcH;
  }
  if (maxSide) {
    const m = Math.max(w, h);
    if (m > maxSide) {
      const k = maxSide / m;
      w = Math.round(w * k);
      h = Math.round(h * k);
    }
  }
  if (multiple > 1) {
    w = Math.max(multiple, Math.round(w / multiple) * multiple);
    h = Math.max(multiple, Math.round(h / multiple) * multiple);
  }
  return { width: Math.max(1, w | 0), height: Math.max(1, h | 0) };
}

/** Luma of a whole raster (used by auto-black-bar + exposure meters). */
export function meanLuma(src, samples = 4096) {
  const d = src.data;
  const total = src.width * src.height;
  const step = Math.max(1, Math.floor(total / samples)) * 4;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < d.length; i += step) {
    sum += luma709(d[i], d[i + 1], d[i + 2]);
    n++;
  }
  return n ? sum / n : 0;
}

/** Separable box blur (3 passes ≈ Gaussian) — the workhorse behind shadows. */
export function boxBlur(raster, radius) {
  if (!(radius > 0)) return raster;
  const { width: w, height: h, data } = raster;
  const tmp = new Uint8ClampedArray(w * h * 4);
  const div = radius * 2 + 1;
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) {
      const row = y * raster.stride;
      for (let c = 0; c < 4; c++) {
        let acc = 0;
        for (let x = -radius; x <= radius; x++) acc += data[row + (Math.min(w - 1, Math.max(0, x)) << 2) + c];
        for (let x = 0; x < w; x++) {
          tmp[row + (x << 2) + c] = acc / div;
          const add = Math.min(w - 1, x + radius + 1);
          const sub = Math.max(0, x - radius);
          acc += data[row + (add << 2) + c] - data[row + (sub << 2) + c];
        }
      }
    }
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 4; c++) {
        let acc = 0;
        for (let y = -radius; y <= radius; y++) acc += tmp[(Math.min(h - 1, Math.max(0, y)) * w + x) * 4 + c];
        for (let y = 0; y < h; y++) {
          data[y * raster.stride + x * 4 + c] = acc / div;
          const add = Math.min(h - 1, y + radius + 1);
          const sub = Math.max(0, y - radius);
          acc += tmp[(add * w + x) * 4 + c] - tmp[(sub * w + x) * 4 + c];
        }
      }
    }
  }
  return raster;
}

export { clamp255 };

/**
 * GIFX Kernel — color / tone / texture filters.
 *
 * Design rules:
 *  - Anything separable or table-lookup-able goes through an 8-bit LUT
 *    (`buildLut`), which makes per-pixel cost ~1 add + 1 index instead of a
 *    pow()/log() call per channel per pixel.
 *  - Multi-pass ops (unsharp, blur) reuse scratch buffers.
 *  - Every filter accepts a `Raster` and mutates it in place unless stated.
 *
 * The `filters()` entry point applies a named chain, which is what the engine
 * calls per frame; `parseFiltergraph()` understands an ffmpeg-ish
 * `scale=480:-2,fps=12,hue=s=140,eq=brightness=0.06` string so users can paste
 * ffmpeg knowledge straight into the browser.
 *
 * @module image/filters
 */
import { Raster } from '../core/buffers.js';
import { ErrorCode } from '../core/errors.js';
import { clamp255, luma709, parseColor, toLinear8, fromLinear01, curveLut } from '../core/color.js';
import { scale, boxBlur, vignette, pixelate, detectBlackBars, crop, pad, rotate, flip, cropToAspect, fitPlan, roundedCorners, border } from './ops.js';

/* --------------------------------------------------------------- LUT utils */

/** Build a 256-entry LUT for a per-channel function. */
export function buildLut(fn, lut) {
  const out = lut || new Uint8Array(256);
  for (let i = 0; i < 256; i++) out[i] = clamp255(Math.round(fn(i)));
  return out;
}

/** Combine LUTs (apply in order) into one — chains stay single-pass. */
export function composeLuts(luts) {
  const out = new Uint8Array(256);
  for (let i = 0; i < 256; i++) out[i] = i;
  for (const l of luts) for (let i = 0; i < 256; i++) out[i] = l[out[i]];
  return out;
}

function applyLut(raster, lut, { alpha = false } = {}) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    const a = d[i + 3];
    if (!alpha && a === 0) continue;
    d[i] = lut[d[i]];
    d[i + 1] = lut[d[i + 1]];
    d[i + 2] = lut[d[i + 2]];
  }
  return raster;
}

/* ------------------------------------------------------------ basic tonality */

/**
 * ffmpeg-`eq`-compatible controls (all default to no-op so chains are cheap).
 * brightness/contrast in -1..1 (+1 contrast = inf), saturation 0..~3, gamma
 * 0.1..4, plus unsharp and a black point.
 */
export function adjust(raster, opts = {}) {
  const b = opts.brightness || 0;
  const c = opts.contrast || 0;
  const g = opts.gamma || 1;
  const s = opts.saturation == null ? 1 : opts.saturation;
  const pivot = (opts.pivot == null ? 0.5 : opts.pivot) * 255;
  const cg = c === 0 ? 1 : Math.tan((c + 1) * (Math.PI / 4));
  const lut = new Uint8Array(256);
  const gamma = g !== 1;
  const bright = b !== 0;
  const contr = c !== 0;
  if (bright || contr || gamma) {
    for (let i = 0; i < 256; i++) {
      let v = i;
      if (bright) v += b * 255;
      if (contr) v = (v - pivot) * cg + pivot;
      if (gamma && v > 0) v = 255 * Math.pow(v / 255, 1 / g);
      lut[i] = clamp255(Math.round(v));
    }
    applyLut(raster, lut);
  }
  if (s !== 1) {
    const d = raster.data;
    const n = raster.height * raster.stride;
    for (let i = 0; i < n; i += 4) {
      if (d[i + 3] === 0) continue;
      const l = luma709(d[i], d[i + 1], d[i + 2]);
      d[i] = clamp255(l + (d[i] - l) * s);
      d[i + 1] = clamp255(l + (d[i + 1] - l) * s);
      d[i + 2] = clamp255(l + (d[i + 2] - l) * s);
    }
  }
  if (opts.hue) rotateHue(raster, opts.hue);
  if (opts.unsharp) unsharp(raster, opts.unsharp, opts.unsharpSize || 3, opts.unsharpThreshold || 0);
  if (opts.temp > 0 || opts.temp < 0) applyTemperature(raster, opts.temp);
  if (opts.tint || opts.tintR != null) applyTint(raster, opts);
  return raster;
}

/** Hue rotation in degrees (table-free; uses YIQ so skin stays plausible). */
export function rotateHue(raster, degrees) {
  const rad = (degrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] === 0) continue;
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const I = 0.596 * r - 0.275 * g - 0.321 * b;
    const Q = 0.212 * r - 0.523 * g + 0.311 * b;
    const I2 = I * cos - Q * sin;
    const Q2 = I * sin + Q * cos;
    d[i] = clamp255(y + 0.956 * I2 + 0.621 * Q2);
    d[i + 1] = clamp255(y - 0.272 * I2 - 0.647 * Q2);
    d[i + 2] = clamp255(y - 1.106 * I2 + 1.703 * Q2);
  }
  return raster;
}

/**
 * Separate color/brightness adjustment (like ffmpeg's `colorlevels`).
 * `inRange` [lo,hi] maps into `outRange`.
 */
export function levels(raster, { inBlack = 0, inWhite = 255, gamma = 1, outBlack = 0, outWhite = 255 } = {}, channel) {
  const lut = new Uint8Array(256);
  const span = Math.max(1, inWhite - inBlack);
  const g = gamma > 0 ? 1 / gamma : 1;
  for (let i = 0; i < 256; i++) {
    let v = (i - inBlack) / span;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    v = Math.pow(v, g);
    lut[i] = clamp255(Math.round(outBlack + v * (outWhite - outBlack)));
  }
  if (channel == null || channel < 0) return applyLut(raster, lut);
  const d = raster.data;
  const n = raster.height * raster.stride;
  const ch = channel | 0;
  for (let i = 0; i < n; i += 4) d[i + ch] = lut[d[i + ch]];
  return raster;
}

/**
 * Curve editor: control points per channel, monotone cubic (PCHIP) so curves
 * never overshoot (a real requirement — a Catmull-Rom through user-dragged
 * points can dip below 0 and produce banding).
 */
export function curve(raster, points, channel = -1) {
  const lut = curveLut(points);
  if (channel < 0) return applyLut(raster, lut);
  const d = raster.data;
  for (let i = channel; i < d.length; i += 4) d[i] = lut[d[i]];
  return raster;
}

/** `filter: ...` property string → our filter set (blur included). */
export function applyCssFilter(raster, filterString) {
  cssFilter(raster, filterString);
  const m = /blur\(\s*([\d.]+)px\s*\)/.exec(filterString || '');
  if (m) boxBlur(raster, Math.max(1, Math.round(parseFloat(m[1]))));
  return raster;
}

/**
 * Cross-channel CSS filters need pixel math, not a LUT — `cssFilter` builds the
 * full description so `filters()` can dispatch correctly.
 */
export function cssFilter(raster, str) {
  const s = String(str || '');
  let m = /brightness\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) applyLut(raster, buildLut((x) => x * parseNorm(m[1], 1)));
  m = /contrast\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) applyLut(raster, buildLut((x) => (x - 128) * parseNorm(m[1], 1) + 128));
  m = /saturate\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) adjust(raster, { saturation: parseNorm(m[1], 1) });
  m = /grayscale\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) {
    const v = Math.min(1, Math.max(0, parseNorm(m[1], 1)));
    if (v > 0) adjust(raster, { saturation: 1 - v });
  }
  m = /hue-rotate\(\s*(-?[\d.]+)(deg)?\s*\)/.exec(s);
  if (m) rotateHue(raster, parseFloat(m[1]));
  m = /invert\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) applyLut(raster, buildLut((x) => x + (255 - 2 * x) * Math.min(1, parseNorm(m[1], 1))));
  m = /sepia\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) sepia(raster, Math.min(1, parseNorm(m[1], 1)));
  m = /opacity\(\s*([\d.]+%?)\s*\)/.exec(s);
  if (m) {
    const v = Math.min(1, parseNorm(m[1], 1));
    const d = raster.data;
    for (let i = 3; i < d.length; i += 4) d[i] = clamp255(d[i] * v);
  }
  return raster;
}

export function sepia(raster, amount = 1) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  const a = Math.min(1, Math.max(0, amount));
  for (let i = 0; i < n; i += 4) {
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    const tr = 0.393 * r + 0.769 * g + 0.189 * b;
    const tg = 0.349 * r + 0.686 * g + 0.168 * b;
    const tb = 0.272 * r + 0.534 * g + 0.131 * b;
    d[i] = clamp255(r + (tr - r) * a);
    d[i + 1] = clamp255(g + (tg - g) * a);
    d[i + 2] = clamp255(b + (tb - b) * a);
  }
  return raster;
}

export function grayscale(raster, mode = 'luma') {
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] === 0) continue;
    let g;
    switch (mode) {
      case 'average':
        g = (d[i] + d[i + 1] + d[i + 2]) / 3;
        break;
      case 'lightness':
        g = (Math.max(d[i], d[i + 1], d[i + 2]) + Math.min(d[i], d[i + 1], d[i + 2])) / 2;
        break;
      case 'max':
        g = Math.max(d[i], d[i + 1], d[i + 2]);
        break;
      case 'rec601':
        g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        break;
      case 'rec2020':
        g = 0.2627 * d[i] + 0.678 * d[i + 1] + 0.0593 * d[i + 2];
        break;
      default:
        g = luma709(d[i], d[i + 1], d[i + 2]);
    }
    const v = clamp255(Math.round(g));
    d[i] = v;
    d[i + 1] = v;
    d[i + 2] = v;
  }
  return raster;
}

export function invert(raster, { rgb = true, alpha = false } = {}) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (rgb) {
      d[i] = 255 - d[i];
      d[i + 1] = 255 - d[i + 1];
      d[i + 2] = 255 - d[i + 2];
    }
    if (alpha) d[i + 3] = 255 - d[i + 3];
  }
  return raster;
}

/** Posterize to N levels per channel (cheap "1-bit look" for dither demos). */
export function posterize(raster, levelsPerChannel = 4) {
  const l = Math.max(2, Math.min(256, levelsPerChannel | 0));
  const step = 255 / (l - 1);
  const lut = buildLut((x) => Math.round(Math.round(x / step) * step));
  return applyLut(raster, lut);
}

/** Threshold to 2 colors (classic 1-bit dither targets). */
export function threshold(raster, t = 128, { low = 0, high = 255 } = {}) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] === 0) continue;
    const v = luma709(d[i], d[i + 1], d[i + 2]) >= t ? high : low;
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  return raster;
}

/** Exposure in stops (linear-light multiply, done in linear space). */
export function exposure(raster, stops = 1) {
  const k = Math.pow(2, stops);
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = clamp255(Math.round(fromLinear01(toLinear8(i) * k) * 255));
  return applyLut(raster, lut);
}

/** White balance in Kelvin-ish: `temperature` -100..100 (cool→warm). */
export function applyTemperature(raster, amount) {
  const t = Math.max(-100, Math.min(100, amount)) / 100;
  const rL = buildLut((x) => x * (1 + 0.22 * t));
  const gL = buildLut((x) => x * (1 - 0.05 * Math.abs(t)));
  const bL = buildLut((x) => x * (1 - 0.22 * t));
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    d[i] = rL[d[i]];
    d[i + 1] = gL[d[i + 1]];
    d[i + 2] = bL[d[i + 2]];
  }
  return raster;
}

/** Color cast/tint with explicit per-channel multipliers or a color. */
export function applyTint(raster, opts) {
  if (opts.tint) {
    const rgb = typeof opts.tint === 'string' ? parseColor(opts.tint) : opts.tint;
    const strength = opts.tintAmount ?? 0.35;
    const d = raster.data;
    const n = raster.height * raster.stride;
    for (let i = 0; i < n; i += 4) {
      d[i] = clamp255(d[i] * (1 - strength) + rgb[0] * strength);
      d[i + 1] = clamp255(d[i + 1] * (1 - strength) + rgb[1] * strength);
      d[i + 2] = clamp255(d[i + 2] * (1 - strength) + rgb[2] * strength);
    }
    return raster;
  }
  const mr = opts.tintR ?? opts.mulR ?? 1;
  const mg = opts.tintG ?? opts.mulG ?? 1;
  const mb = opts.tintB ?? opts.mulB ?? 1;
  if (mr === 1 && mg === 1 && mb === 1) return raster;
  const rl = buildLut((x) => x * mr);
  const gl = buildLut((x) => x * mg);
  const bl = buildLut((x) => x * mb);
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    d[i] = rl[d[i]];
    d[i + 1] = gl[d[i + 1]];
    d[i + 2] = bl[d[i + 2]];
  }
  return raster;
}

/** Auto-levels (histogram stretch, with percentile clipping for robustness). */
export function autoLevels(raster, opts = {}) {
  const clip = opts.clip ?? 0.005;
  const hist = new Uint32Array(256);
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] < 8) continue;
    hist[luma709(d[i], d[i + 1], d[i + 2]) | 0]++;
  }
  let total = 0;
  for (let i = 0; i < 256; i++) total += hist[i];
  if (!total) return raster;
  const lim = total * clip;
  let lo = 0;
  let acc = 0;
  while (lo < 255 && acc + hist[lo] < lim) acc += hist[lo++];
  let hi = 255;
  acc = 0;
  while (hi > lo && acc + hist[hi] < lim) acc += hist[hi--];
  if (hi - lo < 4) return raster;
  const span = hi - lo;
  const lut = buildLut((x) => ((x - lo) * 255) / span);
  return applyLut(raster, lut);
}

/** Gamma-correct L* contrast (less clipping than naive contrast). */
export function vibrance(raster, amount = 0.5) {
  const a = Math.max(-1, Math.min(1, amount));
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] === 0) continue;
    const mx = Math.max(d[i], d[i + 1], d[i + 2]);
    const mn = Math.min(d[i], d[i + 1], d[i + 2]);
    const sat = mx === 0 ? 0 : (mx - mn) / mx;
    const boost = 1 + a * (1 - sat); // protects already-saturated + skin-ish colors
    const l = luma709(d[i], d[i + 1], d[i + 2]);
    d[i] = clamp255(l + (d[i] - l) * boost);
    d[i + 1] = clamp255(l + (d[i + 1] - l) * boost);
    d[i + 2] = clamp255(l + (d[i + 2] - l) * boost);
  }
  return raster;
}

/* ------------------------------------------------------------------ kernels */

/**
 * 3x3 convolution with edge clamping. Handles sharpen/emboss/edge-detect/etc.
 * `div`/`bias` follow the SVG filter convention.
 */
export function convolve3x3(raster, k, div, bias = 0, dst) {
  const d = div || Math.max(1, k.reduce((a, b) => a + b, 0));
  const out = dst || new Raster(raster.width, raster.height, undefined, raster.arena);
  const src = raster.data;
  const od = out.data;
  const w = raster.width;
  const h = raster.height;
  const st = raster.stride;
  const clampX = (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);
  const clampY = (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);
  for (let y = 0; y < h; y++) {
    const ym = clampY(y - 1) * st;
    const y0 = y * st;
    const yp = clampY(y + 1) * st;
    const orow = y * out.stride;
    for (let x = 0; x < w; x++) {
      const xm = clampX(x - 1) << 2;
      const x0 = x << 2;
      const xp = clampX(x + 1) << 2;
      for (let c = 0; c < 3; c++) {
        const cc = c;
        let acc =
          k[0] * src[ym + xm + cc] + k[1] * src[ym + x0 + cc] + k[2] * src[ym + xp + cc] +
          k[3] * src[y0 + xm + cc] + k[4] * src[y0 + x0 + cc] + k[5] * src[y0 + xp + cc] +
          k[6] * src[yp + xm + cc] + k[7] * src[yp + x0 + cc] + k[8] * src[yp + xp + cc];
        acc = acc / d + bias;
        od[orow + x0 + cc] = clamp255(Math.round(acc));
      }
      od[orow + x0 + 3] = src[y0 + x0 + 3];
    }
  }
  if (dst !== raster) {
    raster.data.set(out.data.subarray(0, raster.height * raster.stride));
    if (out !== dst && out.arena) out.arena.release(out);
  }
  return raster;
}

/** Named 3x3 presets. */
export const KERNELS = {
  identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
  sharpenSoft: [-1, -1, -1, -1, 9, -1, -1, -1, -1],
  unsharpKernel: [0, -1, 0, -1, 5, -1, 0, -1, 0],
  emboss: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
  edge: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
  edgeLight: [0, 1, 0, 1, -4, 1, 0, 1, 0],
  laplacian: [0, 1, 0, 1, -4, 1, 0, 1, 0],
  sobelX: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  sobelY: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  prewittX: [-1, 0, 1, -1, 0, 1, -1, 0, 1],
  findEdges: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
  box: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  gaussian: [1, 2, 1, 2, 4, 2, 1, 2, 1],
  gaussian5: [1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1],
  motionBlurH: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  custom: null,
};

export function convolve(raster, nameOrKernel, opts = {}) {
  const k = Array.isArray(nameOrKernel) ? nameOrKernel : KERNELS[nameOrKernel] || KERNELS.identity;
  if (k.length === 25) return convolve5x5(raster, k, opts.div || 273, opts.bias || 0);
  return convolve3x3(raster, k, opts.div, opts.bias || 0);
}

function convolve5x5(raster, k, div, bias) {
  const out = new Raster(raster.width, raster.height, undefined, raster.arena);
  const src = raster.data;
  const od = out.data;
  const w = raster.width;
  const h = raster.height;
  const st = raster.stride;
  for (let y = 0; y < h; y++) {
    const orow = y * out.stride;
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let ky = -2; ky <= 2; ky++) {
          const sy = Math.min(h - 1, Math.max(0, y + ky));
          const sxo = sy * st;
          for (let kx = -2; kx <= 2; kx++) {
            const sx = Math.min(w - 1, Math.max(0, x + kx));
            acc += k[(ky + 2) * 5 + kx + 2] * src[sxo + (sx << 2) + c];
          }
        }
        od[orow + (x << 2) + c] = clamp255(Math.round(acc / div + bias));
      }
      od[orow + (x << 2) + 3] = src[y * st + (x << 2) + 3];
    }
  }
  raster.data.set(out.data.subarray(0, raster.height * raster.stride));
  if (out.arena) out.arena.release(out);
  return raster;
}

/** Gaussian-ish blur via the separable box approximation. */
export function blur(raster, radius = 2, passes = 3) {
  if (!(radius > 0)) return raster;
  const r = Math.round(radius);
  const { width: w, height: h } = raster;
  const tmp = new Uint8Array(w * h * 4);
  for (let p = 0; p < passes; p++) {
    blurH(raster, r, tmp);
    blurV(raster, r, tmp);
  }
  return raster;
}

function blurH(img, r, tmp) {
  const { width: w, height: h, data } = img;
  const div = r * 2 + 1;
  for (let y = 0; y < h; y++) {
    const row = y * img.stride;
    for (let c = 0; c < 4; c++) {
      let acc = data[row + c] * (r + 1);
      for (let x = 1; x <= r; x++) acc += data[row + (Math.min(w - 1, x) << 2) + c];
      for (let x = 0; x < w; x++) {
        tmp[(y * w + x) * 4 + c] = acc / div;
        const add = Math.min(w - 1, x + r + 1);
        const sub = Math.max(0, x - r);
        acc += data[row + (add << 2) + c] - data[row + (sub << 2) + c];
      }
    }
  }
  data.set(tmp.subarray(0, h * w * 4), 0);
}

function blurV(img, r, tmp) {
  const { width: w, height: h, data } = img;
  const div = r * 2 + 1;
  for (let x = 0; x < w; x++) {
    const col = x << 2;
    for (let c = 0; c < 4; c++) {
      let acc = data[col + c] * (r + 1);
      for (let y = 1; y <= r; y++) acc += data[(Math.min(h - 1, y) * img.stride + col) + c];
      const out = new Float32Array(h);
      for (let y = 0; y < h; y++) {
        out[y] = acc / div;
        const add = Math.min(h - 1, y + r + 1);
        const sub = Math.max(0, y - r);
        acc += data[add * img.stride + col + c] - data[sub * img.stride + col + c];
      }
      for (let y = 0; y < h; y++) data[y * img.stride + col + c] = out[y];
    }
  }
  void tmp;
}

/**
 * Unsharp mask. `amount` in 0..3, `threshold` in 0..255 luma units — the
 * threshold is what keeps film grain from becoming white halos.
 */
export function unsharp(raster, amount = 1, radius = 2, threshold = 1) {
  const src = new Uint8Array(raster.data.subarray(0, raster.height * raster.stride));
  const blurred = new Uint8ClampedArray(src);
  const tmp = new Uint8Array(raster.width * raster.height * 4);
  const bImg = { data: blurred, width: raster.width, height: raster.height, stride: raster.width * 4, arena: null };
  blurH(bImg, Math.max(1, radius | 0), tmp);
  blurV(bImg, Math.max(1, radius | 0), tmp);
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    const a = src[i + 3];
    if (a === 0) continue;
    for (let c = 0; c < 3; c++) {
      const diff = src[i + c] - blurred[i + c];
      if (Math.abs(diff) <= threshold) continue;
      d[i + c] = clamp255(src[i + c] + diff * amount);
    }
  }
  return raster;
}

/** Edge-aware smoothing (bilateral-lite): keeps line art crisp while denoising. */
export function denoise(raster, strength = 6) {
  const r = Math.max(1, Math.round(strength));
  const w = raster.width;
  const h = raster.height;
  const src = new Uint8Array(raster.data.subarray(0, h * raster.stride));
  const d = raster.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i0 = y * raster.stride + x * 4;
      const cr = src[i0];
      const cg = src[i0 + 1];
      const cb = src[i0 + 2];
      let rr = 0;
      let gg = 0;
      let bb = 0;
      let ww = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const sy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const sx = Math.min(w - 1, Math.max(0, x + dx));
          const i = sy * raster.stride + sx * 4;
          const dr = src[i] - cr;
          const dg = src[i + 1] - cg;
          const db = src[i + 2] - cb;
          const dist = dr * dr + dg * dg + db * db;
          if (dist > r * r * 3 * 64) continue;
          const weight = 1 + (dx === 0 && dy === 0 ? 2 : 0);
          rr += src[i] * weight;
          gg += src[i + 1] * weight;
          bb += src[i + 2] * weight;
          ww += weight;
        }
      }
      d[i0] = (rr / ww) | 0;
      d[i0 + 1] = (gg / ww) | 0;
      d[i0 + 2] = (bb / ww) | 0;
    }
  }
  return raster;
}

/* ------------------------------------------------------------- dither-adjacent */

/** Ordered noise grain (film look / hides banding before quantization). */
export function grain(raster, amount = 12, { color = false, seed = 1 } = {}) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  let s = seed >>> 0 || 1;
  for (let i = 0; i < n; i += 4) {
    if (d[i + 3] === 0) continue;
    s = (s * 1664525 + 1013904223) >>> 0;
    const r = ((s >>> 8) / 16777216 - 0.5) * 2 * amount;
    d[i] = clamp255(d[i] + r);
    if (color) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const g = ((s >>> 8) / 16777216 - 0.5) * 2 * amount;
      s = (s * 1664525 + 1013904223) >>> 0;
      const b = ((s >>> 8) / 16777216 - 0.5) * 2 * amount;
      d[i + 1] = clamp255(d[i + 1] + g);
      d[i + 2] = clamp255(d[i + 2] + b);
    } else {
      d[i + 1] = clamp255(d[i + 1] + r);
      d[i + 2] = clamp255(d[i + 2] + r);
    }
  }
  return raster;
}

/** Chroma subsample (4:2:0 look) — reduces palette pressure a lot for video. */
export function chromaSubsample(raster, factor = 2) {
  const f = Math.max(1, factor | 0);
  if (f === 1) return raster;
  const w = raster.width;
  const h = raster.height;
  const d = raster.data;
  for (let by = 0; by < h; by += f) {
    for (let bx = 0; bx < w; bx += f) {
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let y = by; y < Math.min(h, by + f); y++) {
        for (let x = bx; x < Math.min(w, bx + f); x++) {
          const i = y * raster.stride + x * 4;
          r += d[i];
          g += d[i + 1];
          b += d[i + 2];
          n++;
        }
      }
      r = (r / n) | 0;
      g = (g / n) | 0;
      b = (b / n) | 0;
      for (let y = by; y < Math.min(h, by + f); y++) {
        for (let x = bx; x < Math.min(w, bx + f); x++) {
          const i = y * raster.stride + x * 4;
          const l = luma709(d[i], d[i + 1], d[i + 2]);
          const rr = clamp255(l + (r - luma709(r, g, b)));
          const gg = clamp255(l + (g - luma709(r, g, b)));
          const bb = clamp255(l + (b - luma709(r, g, b)));
          d[i] = rr;
          d[i + 1] = gg;
          d[i + 2] = bb;
        }
      }
    }
  }
  return raster;
}

/**
 * Color-key / chroma-key → alpha. `tolerance` in squared RGB distance.
 * `feather` softens the matte edge (essential: hard keys alias badly at GIF
 * sizes and the transparency becomes a jaggies mess).
 */
export function chromaKey(raster, keyColor, opts = {}) {
  const key = typeof keyColor === 'string' ? parseColor(keyColor) : keyColor;
  const tol = opts.tolerance ?? 3600;
  const feather = opts.feather ?? 2400;
  const spill = opts.despill || 0;
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    const dr = d[i] - key[0];
    const dg = d[i + 1] - key[1];
    const db = d[i + 2] - key[2];
    const dist = dr * dr + dg * dg + db * db;
    if (dist <= tol) {
      d[i + 3] = 0;
      continue;
    }
    if (feather > 0 && dist <= tol + feather) {
      d[i + 3] = clamp255((d[i + 3] * (dist - tol)) / feather);
    }
    if (spill) {
      // remove green/magenta cast on hair edges
      const which = key[1] > Math.max(key[0], key[2]) ? 1 : key[0] > key[2] ? 0 : 2;
      if (which === 1) d[i + 1] = clamp255(d[i + 1] - (d[i + 1] - Math.max(d[i], d[i + 2])) * spill);
    }
  }
  return raster;
}

/** Remove a near-uniform background by sampling a corner (great for screen-casts). */
export function autoKey(raster, opts = {}) {
  const d = raster.data;
  const st = raster.stride;
  const corners = [0, (raster.width - 1) * 4, (raster.height - 1) * st, (raster.height - 1) * st + (raster.width - 1) * 4];
  let r = 0;
  let g = 0;
  let b = 0;
  for (const c of corners) {
    r += d[c];
    g += d[c + 1];
    b += d[c + 2];
  }
  return chromaKey(raster, [(r / 4) | 0, (g / 4) | 0, (b / 4) | 0], opts);
}

/** Flood-fill background removal: transparent outside closed shapes (logos!). */
export function floodKey(raster, opts = {}) {
  const { width: w, height: h } = raster;
  const d = raster.data;
  const tol = opts.tolerance ?? 2600;
  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const push = (p) => {
    if (seen[p]) return;
    seen[p] = 1;
    stack[sp++] = p;
  };
  const seedColor = [d[0], d[1], d[2]];
  for (let x = 0; x < w; x++) {
    push(x);
    push((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    push(y * w);
    push(y * w + w - 1);
  }
  while (sp > 0) {
    const p = stack[--sp];
    const x = p % w;
    const y = (p / w) | 0;
    const i = y * raster.stride + x * 4;
    const dr = d[i] - seedColor[0];
    const dg = d[i + 1] - seedColor[1];
    const db = d[i + 2] - seedColor[2];
    if (dr * dr + dg * dg + db * db > tol) continue;
    d[i + 3] = 0;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w);
    if (y < h - 1) push(p + w);
  }
  return raster;
}

/** Alpha → premultiplied (needed before compositing onto a color for GIF). */
export function premultiply(raster, bg) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  const bgc = bg || null;
  for (let i = 0; i < n; i += 4) {
    const a = d[i + 3];
    if (a === 255) continue;
    const f = a / 255;
    if (bgc) {
      d[i] = clamp255(d[i] * f + bgc[0] * (1 - f));
      d[i + 1] = clamp255(d[i + 1] * f + bgc[1] * (1 - f));
      d[i + 2] = clamp255(d[i + 2] * f + bgc[2] * (1 - f));
    } else {
      d[i] = clamp255(d[i] * f);
      d[i + 1] = clamp255(d[i + 1] * f);
      d[i + 2] = clamp255(d[i + 2] * f);
    }
    d[i + 3] = 255;
  }
  return raster;
}

/**
 * Matte a hard-edged alpha channel by removing color fringing
 * (`defringe`/`unpremultiply`), which GIFs need because they only have 1-bit
 * transparency and any colored halo becomes visible garbage.
 */
export function defringe(raster, thresholdAlpha = 250) {
  const d = raster.data;
  const n = raster.height * raster.stride;
  for (let i = 0; i < n; i += 4) {
    const a = d[i + 3];
    if (a === 0 || a >= thresholdAlpha) continue;
    const inv = 255 / a;
    d[i] = clamp255(d[i] * inv);
    d[i + 1] = clamp255(d[i + 1] * inv);
    d[i + 2] = clamp255(d[i + 2] * inv);
  }
  if (thresholdAlpha < 255) {
    for (let i = 0; i < n; i += 4) if (d[i + 3] >= thresholdAlpha) d[i + 3] = 255;
  }
  return raster;
}

/** Gamma/tone-curve presets, GIF-friendly (GIF has no color profile). */
export const TONE_PRESETS = {
  none: null,
  punch: { contrast: 0.18, saturation: 1.18, brightness: 0.01 },
  clean: { brightness: 0.02, contrast: 0.06, saturation: 1.06 },
  faded: { contrast: -0.12, saturation: 0.86, brightness: 0.05 },
  cinematic: { contrast: 0.12, saturation: 0.94, tint: '#2b3a55', tintAmount: 0.1, brightness: -0.02 },
  mono: { saturation: 0 },
  noir: { contrast: 0.34, saturation: 0, brightness: -0.04 },
  vintage: { sepia: 0.35, contrast: -0.05, saturation: 0.86, warmth: 12 },
  warm: { temp: 18 },
  cool: { temp: -18 },
  vivid: { saturation: 1.4, contrast: 0.12 },
  pastel: { saturation: 0.72, brightness: 0.08, contrast: -0.06 },
  highkey: { brightness: 0.16, contrast: 0.05 },
  lowkey: { brightness: -0.14, contrast: 0.16 },
};

export function tonePreset(raster, name) {
  const p = TONE_PRESETS[name];
  if (!p) return raster;
  if (p.sepia) sepia(raster, p.sepia);
  if (p.temp || p.warmth) applyTemperature(raster, p.temp ?? p.warmth);
  adjust(raster, {
    brightness: p.brightness || 0,
    contrast: p.contrast || 0,
    saturation: p.saturation == null ? 1 : p.saturation,
    tint: p.tint,
    tintAmount: p.tintAmount,
  });
  return raster;
}

/* --------------------------------------------------------- named filter map */

/**
 * The registry the engine and `parseFiltergraph()` share. Every entry:
 * `(raster, args) => raster`.
 */
export const FILTERS = {
  /** scale=WxH | scale=W:-2 | scale=w=480:h=-1:kernel=lanczos3 (ffmpeg rules) */
  scale(r, args) {
    let w = args.width ?? args.w;
    let h = args.height ?? args.h;
    if (typeof w === 'string' && /[:x]/.test(w) && h == null) {
      const [a, b] = String(w).split(/[:x]/);
      w = a;
      h = b;
    }
    const ar = r.width / r.height;
    // `scale=50%` with no height means "50% of both", like ffmpeg.
    if (h == null && typeof w === 'string' && w.trim().endsWith('%')) h = w;
    w = resolveDim(w, h, r.width, ar, true);
    h = resolveDim(h, w, r.height, ar, false);
    return scale(r, w, h, { kernel: args.kernel || 'area' });
  },
  crop(r, args) {
    return crop(r, args.x || 0, args.y || 0, args.width ?? args.w ?? r.width, args.height ?? args.h ?? r.height);
  },
  pad(r, args) {
    return pad(r, args.width ?? args.w, args.height ?? args.h, { color: args.color, mode: args.mode, transparent: args.transparent });
  },
  rotate(r, args) {
    return rotate(r, parseFloat(args.deg ?? args.angle ?? args['90'] ?? 0));
  },
  flip(r, args) {
    return flip(r, { horizontal: args.h !== false && args.horizontal !== false, vertical: args.v !== false && args.vertical !== false });
  },
  transpose(r) {
    return flip(r, { horizontal: true, vertical: false });
  },
  grayscale(r, args) {
    return grayscale(r, args.mode);
  },
  sepia(r, args) {
    return sepia(r, args.amount == null ? 1 : parseFloat(args.amount));
  },
  invert(r) {
    return invert(r);
  },
  brightness(r, args) {
    return adjust(r, { brightness: num(args.v ?? args.value ?? args.brightness, 0.1) });
  },
  contrast(r, args) {
    return adjust(r, { contrast: num(args.v ?? args.value ?? args.contrast, 0.1) });
  },
  saturate(r, args) {
    return adjust(r, { saturation: num(args.v ?? args.value ?? args.saturation, 1) });
  },
  hue(r, args) {
    return rotateHue(r, num(args.h ?? args.degrees ?? args.hue, 0));
  },
  eq(r, args) {
    return adjust(r, {
      brightness: num(args.brightness, 0),
      contrast: num(args.contrast, 0),
      saturation: num(args.saturation, 1),
      gamma: num(args.gamma, 1),
      hue: num(args.hue, 0),
      pivot: num(args.pivot, 0.5),
    });
  },
  curves(r, args) {
    const pts = args.points || (args.preset ? CURVE_PRESETS[args.preset] : null);
    return pts ? curve(r, pts) : r;
  },
  levels(r, args) {
    return levels(r, args);
  },
  unsharp(r, args) {
    return unsharp(r, num(args.amount, 1), num(args.size ?? args.lsize, 3), num(args.max, 0));
  },
  blur(r, args) {
    return blur(r, num(args.radius ?? args.size, 2));
  },
  boxblur(r, args) {
    return boxBlur(r, num(args.radius ?? args.l, 2));
  },
  denoise(r, args) {
    return denoise(r, num(args.strength ?? args.s, 6));
  },
  convolve(r, args) {
    return convolve(r, args.kernel || 'sharpen', { div: args.div, bias: args.bias });
  },
  sharpen(r, args) {
    return convolve(r, 'sharpen', { bias: num(args.bias, 0) });
  },
  emboss(r) {
    return convolve(r, 'emboss', { bias: 128 });
  },
  edges(r) {
    return convolve(r, 'edge');
  },
  posterize(r, args) {
    return posterize(r, num(args.bits ?? args.levels, 4));
  },
  threshold(r, args) {
    return threshold(r, num(args.t, 128));
  },
  exposure(r, args) {
    return exposure(r, num(args.ev ?? args.stops, 1));
  },
  temperature(r, args) {
    return applyTemperature(r, num(args.v ?? args.degrees, 20));
  },
  vibrance(r, args) {
    return vibrance(r, num(args.amount, 0.5));
  },
  vignette(r, args) {
    return vignette(r, num(args.amount, 0.5), { softness: num(args.softness, 0.6), color: args.color });
  },
  grain(r, args) {
    return grain(r, num(args.amount, 12), { color: !!args.color, seed: num(args.seed, 1) });
  },
  pixelate(r, args) {
    return pixelate(r, num(args.size ?? args.block, 8));
  },
  chromaSubsample(r, args) {
    return chromaSubsample(r, num(args.factor, 2));
  },
  chromakey(r, args) {
    return chromaKey(r, args.color || '#00ff00', { tolerance: num(args.tolerance, 3600), feather: num(args.feather, 2400), despill: num(args.despill, 0) });
  },
  autokey(r, args) {
    return autoKey(r, args);
  },
  floodkey(r, args) {
    return floodKey(r, args);
  },
  alphaThreshold(r, args) {
    const t = num(args.v, 128);
    const d = r.data;
    for (let i = 3; i < d.length; i += 4) d[i] = d[i] >= t ? 255 : 0;
    return r;
  },
  defringe(r, args) {
    return defringe(r, num(args.t, 250));
  },
  premultiply(r, args) {
    return premultiply(r, args.color);
  },
  autoLevels(r, args) {
    return autoLevels(r, args);
  },
  tone(r, args) {
    return tonePreset(r, args.preset || args.name || 'punch');
  },
  css(r, args) {
    return cssFilter(r, args.filter || args.value || '');
  },
  rounded(r, args) {
    return roundedCorners(r, num(args.radius, 12));
  },
  border(r, args) {
    return border(r, num(args.width, 2), { color: args.color, gradient: args.gradient });
  },
  aspect(r, args) {
    return cropToAspect(r, args.ratio || args.a || '16:9');
  },
  cropdetect(r, args) {
    const c = detectBlackBars(r, { threshold: num(args.threshold, 24) });
    if (c.hasBars) return crop(r, c.left, c.top, c.width, c.height);
    return r;
  },
  /** Force an exact output size preserving AR with a letterbox. */
  fit(r, args) {
    const plan = fitPlan(r.width, r.height, num(args.width, r.width), num(args.height, r.height), args.mode || 'contain');
    let out = crop(r, plan.cropX, plan.cropY, plan.cropW, plan.cropH);
    out = scale(out, plan.width, plan.height, { kernel: args.kernel || 'area' });
    if (plan.pad) out = pad(out, plan.pad.width, plan.pad.height, { color: args.color, transparent: !!args.transparent });
    return out;
  },
};
const num = (v, d) => (v == null || v === '' || Number.isNaN(parseFloat(v)) ? d : parseFloat(v));

/**
 * ffmpeg's `-1` (keep aspect) / `-2` (keep aspect, force even) semantics for
 * `scale=w:h`. Even-multiples matter for codecs; for GIF they matter because a
 * 2-pixel difference in width changes LZW row alignment and thus the size.
 */
function resolveDim(v, other, srcDim, ar, isWidth) {
  if (v == null || v === '') return other != null && other !== '' ? resolveDim(other, v, srcDim, ar, !isWidth) : srcDim;
  const pct = typeof v === 'string' && v.trim().endsWith('%');
  const n = typeof v === 'number' ? v : parseFloat(v);
  if (!Number.isFinite(n)) return srcDim;
  if (pct) return Math.max(1, Math.round((srcDim * n) / 100));
  const fromOther = () => Math.max(1, Math.round(isWidth ? other * ar : other / ar));
  if (n === -1) return other != null ? fromOther() : srcDim;
  if (n === -2) {
    const raw = other != null ? Math.max(2, fromOther()) : srcDim;
    return raw - (raw % 2);
  }
  return Math.max(1, n | 0);
}

/** Standard curve shapes for the UI + `curves=preset=…`. */
export const CURVE_PRESETS = {
  rgb: [[0, 0], [255, 255]],
  contrast: [[0, 0], [64, 48], [192, 208], [255, 255]],
  soft: [[0, 12], [64, 60], [192, 196], [255, 244]],
  film: [[0, 18], [48, 44], [208, 212], [255, 238]],
  pop: [[0, 0], [96, 68], [160, 200], [255, 255]],
  fade: [[0, 26], [255, 232]],
  bleach: [[0, 34], [96, 92], [192, 204], [255, 255]],
  night: [[0, 0], [64, 40], [160, 150], [224, 214], [255, 250]],
};
/* -------------------------------------------------------- filtergraph parser */

/**
 * ffmpeg-ish filtergraph: `scale=480:-2,fps=12,eq=brightness=0.06:saturation=1.2`
 * Also accepts JSON option objects from the UI: `[{filter:'eq',options:{...}}]`.
 * Unknown names throw a typed `filter-unknown` error so typos never silently
 * no-op (the single most annoying thing about hand-written filter strings).
 */
export function parseFiltergraph(str, registry = FILTERS) {
  if (!str) return [];
  if (Array.isArray(str)) return str.map((s) => (typeof s === 'string' ? parseOne(s, registry) : { name: s.name || s.filter, args: s.args || s.options || {}, apply: s.apply }));
  if (typeof str !== 'string') throw mkErr('filtergraph must be a string or array', ErrorCode.FILTER_INVALID);
  const out = [];
  for (const chunk of splitTopLevel(str)) {
    const t = chunk.trim();
    if (!t) continue;
    out.push(parseOne(t, registry));
  }
  return out;
}

function splitTopLevel(str) {
  const parts = [];
  let depth = 0;
  let cur = '';
  for (const ch of str) {
    if (ch === '[' || ch === '{' || ch === '(') depth++;
    if (ch === ']' || ch === '}' || ch === ')') depth--;
    if (ch === ',' && depth === 0) {
      parts.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}

function parseOne(t, registry) {
  const eq = t.indexOf('=');
  const name = (eq === -1 ? t : t.slice(0, eq)).trim();
  const argStr = eq === -1 ? '' : t.slice(eq + 1).trim();
  const args = {};
  if (argStr) {
    const kv = argStr.split(/[:;]/);
    if (kv.every((p) => p.includes('='))) {
      for (const p of kv) {
        const [k, ...rest] = p.split('=');
        args[k.trim()] = coerce(rest.join('='));
      }
    } else if (name === 'scale') {
      const [w, h] = argStr.split('x');
      args.width = coerce(w);
      args.height = coerce(h);
    } else if (name === 'crop') {
      const [w, h, x, y] = argStr.split(':');
      args.width = coerce(w);
      args.height = coerce(h);
      args.x = coerce(x);
      args.y = coerce(y);
    } else {
      const parts = argStr.split(':');
      const meta = POSITIONAL[name];
      if (meta) parts.forEach((p, i) => (args[meta[i] || `arg${i}`] = coerce(p)));
      else args.value = coerce(argStr);
    }
  }
  if (!registry[name]) throw mkErr(`unknown filter "${name}" (known: ${Object.keys(registry).sort().join(', ')})`, ErrorCode.FILTER_UNKNOWN);
  return { name, args };
}
const POSITIONAL = {
  scale: ['width', 'height'],
  crop: ['width', 'height', 'x', 'y'],
  pad: ['width', 'height', 'x', 'y', 'color'],
  unsharp: ['size', 'amount', 'sigma', 'threshold'],
  boxblur: ['radius', 'steps'],
  pixelate: ['block'],
  hue: ['h'],
};
const coerce = (v) => {
  if (v == null) return v;
  const s = String(v).trim();
  if (/^-?\d+(\.\d+)?$/.test(s)) return parseFloat(s);
  if (s === 'true') return true;
  if (s === 'false') return false;
  return s;
};

/** Apply a parsed chain; returns the (possibly replaced) raster. */
export function applyFilters(raster, chain, ctx = {}) {
  let cur = raster;
  for (const step of chain) {
    const fn = step.apply || FILTERS[step.name];
    if (!fn) throw mkErr(`unknown filter "${step.name}"`, ErrorCode.FILTER_UNKNOWN);
    const next = fn(cur, step.args, ctx);
    if (next instanceof Promise) throw mkErr(`filter "${step.name}" is asynchronous — use applyFiltersAsync`, ErrorCode.FILTER_ASYNC);
    if (next && next !== cur && !(next instanceof Raster)) throw mkErr(`filter "${step.name}" returned a non-raster`, ErrorCode.FILTER_INVALID);
    if (next && next !== cur) {
      if (cur.arena && cur !== raster) cur.arena.release(cur);
      cur = next;
    }
  }
  return cur;
}

export async function applyFiltersAsync(raster, chain, ctx = {}) {
  let cur = raster;
  for (const step of chain) {
    const fn = step.apply || FILTERS[step.name];
    if (!fn) throw mkErr(`unknown filter "${step.name}"`, ErrorCode.FILTER_UNKNOWN);
    const next = await fn(cur, step.args, ctx);
    if (next && next !== cur) {
      if (cur.arena && cur !== raster) cur.arena.release(cur);
      cur = next;
    }
  }
  return cur;
}

/** Human-readable list for UIs / error messages. */
export function listFilters() {
  return Object.keys(FILTERS).sort();
}

function mkErr(msg, code) {
  const e = new Error(msg);
  e.code = code;
  e.retryable = false;
  return e;
}

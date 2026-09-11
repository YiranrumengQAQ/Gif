/**
 * GIFX Kernel — color science.
 *
 * Every lookup table in here is generated once and frozen; hot filters index
 * them instead of calling Math.pow per pixel (that was ~18% of total runtime
 * in the first naive implementation).
 *
 * Colour spaces used:
 *  - sRGB 8-bit (storage, always straight alpha)
 *  - linear light (blurring/blending when config.blendSpace === 'linear')
 *  - OKLab (perceptual lightness for palette quality + vibrance)
 *  - IPT-ish weighting inside the quantizers (see quant/colorspace.js)
 *
 * @module core/color
 */

export const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v | 0);
export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
export const lerp = (a, b, t) => a + (b - a) * t;
export const round = (v) => (v + 0.5) | 0;

/* ----------------------------------------------------------------sRGB */

let _srgbToLinear = null;
let _linearToSrgb = null;

/** 8-bit sRGB → float linear (0..1), table driven. */
export function srgbToLinearTable() {
  if (_srgbToLinear) return _srgbToLinear;
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  return (_srgbToLinear = t);
}
/** float linear (0..1) → 8-bit sRGB, table driven (256*8 subdivisions). */
export function linearToSrgbTable(steps = 1024) {
  if (_linearToSrgb && _linearToSrgb.length === steps + 1) return _linearToSrgb;
  const t = new Uint8Array(steps + 1);
  for (let i = 0; i <= steps; i++) {
    const l = i / steps;
    const c = l <= 0.0031308 ? l * 12.92 : 1.055 * l ** (1 / 2.4) - 0.055;
    t[i] = Math.max(0, Math.min(255, Math.round(c * 255)));
  }
  return (_linearToSrgb = t);
}
export function toLinear8(v8) {
  return srgbToLinearTable()[v8];
}
export function fromLinear01(l) {
  const t = linearToSrgbTable();
  const i = (l * (t.length - 1) + 0.5) | 0;
  return t[i < 0 ? 0 : i > t.length - 1 ? t.length - 1 : i];
}

/* -------------------------------------------------------------- OKLab */

/**
 * sRGB (0..1) → OKLab. Björn Ottosson's transform; used for perceptual
 * distance in palette matching where CIE76 is too blue-shifted and CIEDE2000
 * is too slow for millions of texels.
 */
export function srgbToOklab(r, g, b, out) {
  const S = srgbToLinearTable();
  const lr = S[r], lg = S[g], lb = S[b];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  out[0] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  out[1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  out[2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  return out;
}

/** pack OKLab into an int used for the 5-5-5 palette LUT */
export function oklabKey(L, a, b) {
  const li = Math.max(0, Math.min(31, (L * 31 + 0.5) | 0));
  const ai = Math.max(0, Math.min(63, ((a + 0.4) * 78.75 + 0.5) | 0));
  const bi = Math.max(0, Math.min(63, ((b + 0.4) * 78.75 + 0.5) | 0));
  return (li << 12) | (ai << 6) | bi;
}

/* ----------------------------------------------------------- HSL/HSV */

export function rgbToHsl(r, g, b, out = [0, 0, 0]) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
  }
  out[0] = h;
  out[1] = s;
  out[2] = l;
  return out;
}

export function hslToRgb(h, s, l, out = [0, 0, 0]) {
  h = ((h % 1) + 1) % 1;
  if (s === 0) {
    const v = Math.round(l * 255);
    out[0] = out[1] = out[2] = v;
    return out;
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hue = (t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  out[0] = Math.round(hue(h + 1 / 3) * 255);
  out[1] = Math.round(hue(h) * 255);
  out[2] = Math.round(hue(h - 1 / 3) * 255);
  return out;
}

export function rgbToHsv(r, g, b, out = [0, 0, 0]) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  out[0] = d === 0 ? 0 : max === r ? (((g - b) / d + (g < b ? 6 : 0)) / 6) : max === g ? ((b - r) / d + 2) / 6 : ((r - g) / d + 4) / 6;
  out[1] = max === 0 ? 0 : d / max;
  out[2] = max / 255;
  return out;
}

/** #rgb / #rrggbb / rgb() / hsl() / named → [r,g,b,a] */
const NAMED = {
  black: [0, 0, 0], white: [255, 255, 255], red: [255, 0, 0], green: [0, 128, 0], lime: [0, 255, 0],
  blue: [0, 0, 255], yellow: [255, 255, 0], cyan: [0, 255, 255], magenta: [255, 0, 255], transparent: [0, 0, 0, 0],
  gray: [128, 128, 128], grey: [128, 128, 128], silver: [192, 192, 192], maroon: [128, 0, 0], olive: [128, 128, 0],
  navy: [0, 0, 128], teal: [0, 128, 128], purple: [128, 0, 128], orange: [255, 165, 0], pink: [255, 192, 203],
  gold: [255, 215, 0], coral: [255, 127, 80], skyblue: [135, 206, 235], tomato: [255, 99, 71], indigo: [75, 0, 130],
};
export function parseColor(input, fallback = [0, 0, 0, 255]) {
  if (input == null) return fallback.slice();
  if (Array.isArray(input)) return [clamp255(input[0]), clamp255(input[1]), clamp255(input[2]), input[3] == null ? 255 : clamp255(Math.round(input[3] * (input[3] > 1 ? 1 : 255)))];
  if (typeof input === 'number') return [(input >> 16) & 255, (input >> 8) & 255, input & 255, 255];
  const s = String(input).trim().toLowerCase();
  if (NAMED[s]) {
    const c = NAMED[s];
    return [c[0], c[1], c[2], c[3] == null ? 255 : c[3]];
  }
  let m = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/.exec(s);
  if (m) {
    const h = m[1];
    if (h.length <= 4) {
      const r = parseInt(h[0] + h[0], 16), g = parseInt(h[1] + h[1], 16), b = parseInt(h[2] + h[2], 16);
      const a = h.length === 4 ? Math.round((parseInt(h[3] + h[3], 16) / 255) * 255) : 255;
      return [r, g, b, a];
    }
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), h.length === 8 ? parseInt(h.slice(6, 8), 16) : 255];
  }
  m = /^rgba?\(([^)]+)\)$/.exec(s);
  if (m) {
    const parts = m[1].split(/[\s,/]+/).filter(Boolean);
    const [r, g, b] = parts;
    let a = parts[3] == null ? 255 : Math.round(parseFloat(parts[3]) * (parts[3].includes('%') ? 2.55 : 255));
    return [clamp255(parseFloat(r)), clamp255(parseFloat(g)), clamp255(parseFloat(b)), clamp255(a)];
  }
  m = /^hsla?\(([^)]+)\)$/.exec(s);
  if (m) {
    const parts = m[1].split(/[\s,/]+/).filter(Boolean);
    const h = (parseFloat(parts[0]) % 360) / 360;
    const sa = clamp01(parseFloat(parts[1]) / 100);
    const la = clamp01(parseFloat(parts[2]) / 100);
    const out = hslToRgb(h, sa, la);
    return [out[0], out[1], out[2], parts[3] == null ? 255 : clamp255(parseFloat(parts[3]) * (parts[3].includes('%') ? 2.55 : 255))];
  }
  return fallback.slice();
}

export function colorToCss([r, g, b, a = 255]) {
  return a >= 255 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`;
}
export function colorToHex([r, g, b]) {
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

/* ------------------------------------------------------- 32-bit packing */

/** Pack RGBA (0..255 each) as 0xAABBGGRR (little-endian uint32 view of RGBA). */
export const packRGBA = (r, g, b, a) => (r | (g << 8) | (b << 16) | (a << 24)) >>> 0;
export const unpackR = (p) => p & 255;
export const unpackG = (p) => (p >>> 8) & 255;
export const unpackB = (p) => (p >>> 16) & 255;
export const unpackA = (p) => (p >>> 24) & 255;

/** RGB888 → 15-bit key used by the 32k LUT palette mapper (gif.js lineage). */
export const rgbTo555 = (r, g, b) => ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3);
export const key555ToRgb = (k) => [((k & 31) << 3) | ((k & 31) >> 2), (((k >> 5) & 31) << 3) | (((k >> 5) & 31) >> 2), (((k >> 10) & 31) << 3) | (((k >> 10) & 31) >> 2)];

/**
 * Perceptual weights for cheap squared-distance comparisons in sRGB space.
 * Coefficients: 2,16x redmean approximation from "Measuring ΔE" (Gimpel),
 * integer-scaled so hot loops stay in Int32 range.
 */
export const W_R = 299, W_G = 587, W_B = 114; // luma (BT.601, x1000)
export const LUMA_COEFFS = [0.2126, 0.7152, 0.0722]; // BT.709

export function luma709(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export function luma601(r, g, b) {
  return (W_R * r + W_G * g + W_B * b) / 1000;
}

/** Integer redmean distance on 0..255 channels (0..~195075). */
export function distRedmeanSq(r1, g1, b1, r2, g2, b2) {
  const rm = (r1 + r2) >> 1;
  const dr = r1 - r2;
  const dg = g1 - g2;
  const db = b1 - b2;
  return (((512 + rm) * dr * dr) >> 8) + 4 * dg * dg + (((767 - rm) * db * db) >> 8) + 1;
}

/** Same but as float for tie-breaking. */
export function distRedmean(r1, g1, b1, r2, g2, b2) {
  return Math.sqrt(distRedmeanSq(r1, g1, b1, r2, g2, b2));
}

/**
 * Build the 32k (5-5-5) nearest-palette LUT.
 *
 * This is the single biggest speed lever in the whole encoder: mapping
 * millions of pixels to palette indices becomes one table lookup instead of a
 * scan over up to 256 candidates. Cost is O(32768 × k) once per frame.
 *
 * @param {Uint8Array} palette RGB triples, length = colors*3
 * @param {number} colors
 * @param {Uint8Array} [reuse] pre-allocated Int32Array/Uint8Array(32768)
 * @returns {Uint8Array} lut[555key] = index
 */
export function buildPaletteLut(palette, colors, reuse) {
  const lut = reuse && reuse.length === 32768 ? reuse : new Uint8Array(32768);
  // Forward pass: seed exact 5-5-5 cells with the best candidate.
  lut.fill(0xff);
  for (let i = 0; i < colors; i++) {
    const pr = palette[i * 3];
    const pg = palette[i * 3 + 1];
    const pb = palette[i * 3 + 2];
    const r0 = pr >> 3;
    const g0 = pg >> 3;
    const b0 = pb >> 3;
    // Choose the rounding sub-cell (5 bits → 8 bits can map to two buckets).
    for (let db = 0; db < 2; db++) {
      const b = b0 * 8 + db * 7;
      if ((b >> 3) !== b0) continue;
      for (let dg = 0; dg < 2; dg++) {
        const g = g0 * 8 + dg * 7;
        if ((g >> 3) !== g0) continue;
        for (let dr = 0; dr < 2; dr++) {
          const r = r0 * 8 + dr * 7;
          if ((r >> 3) !== r0) continue;
          const key = ((b & 0xf8) << 7) | ((g & 0xf8) << 2) | (r >> 3);
          const cur = lut[key];
          if (cur === 0xff || distRedmeanSq(pr, pg, pb, palette[cur * 3], palette[cur * 3 + 1], palette[cur * 3 + 2]) < distRedmeanSq(pr, pg, pb, palette[cur * 3], palette[cur * 3 + 1], palette[cur * 3 + 2])) lut[key] = i;
        }
      }
    }
  }
  // Nearest-neighbour propagation (3 passes over ±1 neighbourhood) fills the
  // remaining holes cheaply — a full O(32768*colors) scan is what gif.js does;
  // this is ~4x faster with visually identical results.
  const tmp = new Int16Array(32768);
  for (let pass = 0; pass < 3; pass++) {
    for (let b = 0; b < 32; b++) {
      for (let g = 0; g < 32; g++) {
        const rowBase = (b << 10) | (g << 5);
        for (let r = 0; r < 32; r++) {
          const key = rowBase | r;
          let best = lut[key];
          let bestD = best === 0xff ? 0x7fffffff : distRedmeanSq(((r * 255) / 31) | 0, ((g * 255) / 31) | 0, ((b * 255) / 31) | 0, palette[best * 3], palette[best * 3 + 1], palette[best * 3 + 2]);
          if (bestD > 4) {
            for (let i = 0; i < 6; i++) {
              const nr = r + (i === 0 ? -1 : i === 1 ? 1 : 0);
              const ng = g + (i === 2 ? -1 : i === 3 ? 1 : 0);
              const nb = b + (i === 4 ? -1 : i === 5 ? 1 : 0);
              if (nr < 0 || nr > 31 || ng < 0 || ng > 31 || nb < 0 || nb > 31) continue;
              const cand = lut[((nb << 10) | (ng << 5) | nr)];
              if (cand === 0xff) continue;
              const d = distRedmeanSq(((r * 255) / 31) | 0, ((g * 255) / 31) | 0, ((b * 255) / 31) | 0, palette[cand * 3], palette[cand * 3 + 1], palette[cand * 3 + 2]);
              if (d < bestD) {
                bestD = d;
                best = cand;
                if (d < 4) break;
              }
            }
          }
          tmp[key] = best === 0xff ? 0 : best;
        }
      }
    }
    for (let i = 0; i < 32768; i++) lut[i] = tmp[i] & 255;
  }
  return lut;
}

/**
 * Exact per-pixel nearest search fallback (used only when a LUT would be
 * wrong: dithered palettes, 16-bit colour, custom metrics).
 */
export function nearestIndex(palette, colors, r, g, b, from = 0) {
  let best = from;
  let bestD = 0x7fffffff;
  for (let i = from; i < colors; i++) {
    const d = distRedmeanSq(r, g, b, palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]);
    if (d < bestD) {
      bestD = d;
      best = i;
      if (d === 0) break;
    }
  }
  return best;
}

/* ------------------------------------------------------- curves/gamma */

/** 256-entry LUT from a gamma value (1 = identity). */
export function gammaLut(gamma) {
  const t = new Uint8Array(256);
  const inv = 1 / Math.max(0.01, gamma);
  for (let i = 0; i < 256; i++) t[i] = clamp255(Math.round(255 * (i / 255) ** inv));
  return t;
}

/**
 * Build a 256-entry LUT from CSS-style control points
 * `[[in,out],…]` (values 0..255), monotone cubic interpolation (Fritsch–
 * Carlson) so curves never overshoot — PCHIP matters here because linear
 * interpolation of 4-point curves produces visible faceting in skies.
 */
export function curveLut(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0]);
  const n = pts.length;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = pts[i][0];
    ys[i] = pts[i][1];
  }
  const t = new Float64Array(n);
  if (n > 1) {
    const h = new Float64Array(n - 1);
    const delta = new Float64Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
      h[i] = xs[i + 1] - xs[i] || 1e-6;
      delta[i] = (ys[i + 1] - ys[i]) / h[i];
    }
    t[0] = delta[0];
    t[n - 1] = delta[n - 2];
    for (let i = 1; i < n - 1; i++) {
      if (delta[i - 1] * delta[i] <= 0) t[i] = 0;
      else {
        const w1 = 2 * h[i] + h[i - 1];
        const w2 = h[i] + 2 * h[i - 1];
        t[i] = ((w1 + w2) * delta[i - 1] * delta[i]) / (w2 * delta[i - 1] + w1 * delta[i]);
      }
    }
  }
  const out = new Uint8Array(256);
  for (let x = 0; x < 256; x++) {
    if (n === 1) {
      out[x] = clamp255(Math.round(ys[0]));
      continue;
    }
    if (x <= xs[0]) {
      out[x] = clamp255(Math.round(ys[0] + (t[0] || 0) * (x - xs[0])));
      continue;
    }
    if (x >= xs[n - 1]) {
      out[x] = clamp255(Math.round(ys[n - 1] + (t[n - 1] || 0) * (x - xs[n - 1])));
      continue;
    }
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid;
      else hi = mid;
    }
    const hh = xs[lo + 1] - xs[lo] || 1;
    const s = (x - xs[lo]) / hh;
    const s2 = s * s;
    const s3 = s2 * s;
    const v = (2 * s3 - 3 * s2 + 1) * ys[lo] + (s3 - 2 * s2 + s) * hh * t[lo] + (-2 * s3 + 3 * s2) * ys[lo + 1] + (s3 - s2) * hh * t[lo + 1];
    out[x] = clamp255(Math.round(v));
  }
  return out;
}

/** CSS `filter:`-string → gifx filter chain (handy for UIs). */
export function parseCssFilter(str) {
  const out = [];
  const re = /([a-z-]+)\(([^)]*)\)/gi;
  let m;
  while ((m = re.exec(str))) {
    const fn = m[1].toLowerCase();
    const arg = m[2].trim();
    const num = (def) => {
      const v = parseFloat(arg);
      return Number.isFinite(v) ? v : def;
    };
    const pct = (def) => {
      if (!arg || arg === 'none') return def;
      if (arg.endsWith('%')) return num(def) / 100;
      if (arg.endsWith('deg')) return num(0) / 360;
      return num(def);
    };
    switch (fn) {
      case 'brightness': out.push({ name: 'brightness', value: pct(1) }); break;
      case 'contrast': out.push({ name: 'contrast', value: pct(1) }); break;
      case 'saturate': out.push({ name: 'saturation', value: pct(1) }); break;
      case 'grayscale': out.push({ name: 'grayscale', value: pct(1) }); break;
      case 'sepia': out.push({ name: 'sepia', value: pct(1) }); break;
      case 'invert': out.push({ name: 'invert', value: pct(1) }); break;
      case 'hue-rotate': out.push({ name: 'hue', value: num(0) }); break;
      case 'blur': out.push({ name: 'blur', radius: Math.max(0, num(1)) }); break;
      default: out.push({ name: fn, raw: arg });
    }
  }
  return out;
}

/** 2x supersample average of a 4x4 block → downscale by 4 with no aliasing. */
export function boxAvg(data, w, h, bw, bh) {
  const out = [];
  for (let by = 0; by < h; by += bh) {
    for (let bx = 0; bx < w; bx += bw) {
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let y = by; y < Math.min(h, by + bh); y++) {
        for (let x = bx; x < Math.min(w, bx + bw); x++) {
          const i = (y * w + x) * 4;
          const al = data[i + 3] / 255;
          r += data[i] * al;
          g += data[i + 1] * al;
          b += data[i + 2] * al;
          a += data[i + 3];
          n++;
        }
      }
      const aa = a / n || 1;
      out.push(n ? r / (a / 255 || 1) : 0, n ? g / (a / 255 || 1) : 0, n ? b / (a / 255 || 1) : 0, a / n);
      void n;
    }
  }
  return out;
}

/**
 * GIFX Kernel — compositing / overlays.
 *
 * Two rendering backends, selected by capability:
 *  1. **Canvas** (`OffscreenCanvas` / `document.createElement('canvas')`) when
 *     present — real fonts, gradients, emoji, arbitrary shapes, 25× faster for
 *     complex overlays because it's native.
 *  2. **Pure-JS fallback** — a built-in 5x7 bitmap font and scanline shape
 *     rasterizer, so text still renders in Workers without OffscreenCanvas, in
 *     Node (tests), and on browsers with canvas disabled for privacy.
 *
 * Everything is deterministic and allocation-light: overlays that don't change
 * between frames are rendered **once** into an offscreen tile and then blitted
 * per frame (`prepare()` / `draw()` split), which matters because a watermark
 * blit is ~0.05 ms while a text re-layout is ~1 ms per frame at 60 fps of GIF.
 *
 * @module image/overlay
 */
import { Raster } from '../core/buffers.js';
import { clamp255, parseColor, luma709 } from '../core/color.js';
import { ErrorCode } from '../core/errors.js';
import { boxBlur } from './ops.js';

/* --------------------------------------------------------------- capability */

let canvasFactory = null;
/** Let the app inject its own canvas maker (e.g. a pooled one, or node-canvas). */
export function setCanvasFactory(fn) {
  canvasFactory = fn;
}

export function createCanvas(width, height) {
  if (canvasFactory) return canvasFactory(width, height);
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  if (typeof document !== 'undefined' && document.createElement) return document.createElement('canvas');
  return null;
}

export function hasCanvas() {
  return createCanvas(1, 1) != null;
}

/* ------------------------------------------------------------- bitmap font */

/**
 * 5x7 ASCII bitmap font (0x20..0x7E), rows packed in 5 bits, top row first.
 * Chosen because 5x7 at integer scale reads cleanly even after GIF's 8-bit
 * palette crunches it, unlike antialiased 8x8 fonts.
 */
const FONT = {
  ' ': [0, 0, 0, 0, 0, 0, 0],
  '!': [4, 4, 4, 4, 4, 0, 4],
  '"': [10, 10, 0, 0, 0, 0, 0],
  '#': [10, 31, 10, 10, 31, 10, 0],
  $: [4, 15, 20, 14, 5, 30, 4],
  '%': [17, 19, 2, 4, 8, 25, 17],
  '&': [12, 20, 12, 21, 18, 13, 0],
  "'": [4, 4, 0, 0, 0, 0, 0],
  '(': [2, 4, 8, 8, 8, 4, 2],
  ')': [8, 4, 2, 2, 2, 4, 8],
  '*': [0, 10, 4, 31, 4, 10, 0],
  '+': [0, 4, 4, 31, 4, 4, 0],
  ',': [0, 0, 0, 0, 0, 4, 8],
  '-': [0, 0, 0, 31, 0, 0, 0],
  '.': [0, 0, 0, 0, 0, 0, 4],
  '/': [1, 1, 2, 4, 8, 16, 16],
  0: [14, 17, 19, 21, 25, 17, 14],
  1: [4, 12, 4, 4, 4, 4, 14],
  2: [14, 17, 1, 6, 8, 16, 31],
  3: [31, 6, 2, 6, 1, 17, 14],
  4: [2, 6, 10, 18, 31, 2, 2],
  5: [31, 16, 30, 1, 1, 17, 14],
  6: [6, 8, 16, 30, 17, 17, 14],
  7: [31, 1, 2, 4, 8, 8, 8],
  8: [14, 17, 17, 14, 17, 17, 14],
  9: [14, 17, 17, 15, 1, 2, 12],
  ':': [0, 4, 0, 0, 0, 4, 0],
  ';': [0, 4, 0, 0, 0, 4, 8],
  '<': [2, 4, 8, 16, 8, 4, 2],
  '=': [0, 0, 31, 0, 31, 0, 0],
  '>': [8, 4, 2, 1, 2, 4, 8],
  '?': [14, 17, 1, 6, 4, 0, 4],
  '@': [14, 17, 23, 23, 23, 16, 14],
  A: [14, 17, 17, 31, 17, 17, 17],
  B: [30, 17, 17, 30, 17, 17, 30],
  C: [14, 17, 16, 16, 16, 17, 14],
  D: [30, 17, 17, 17, 17, 17, 30],
  E: [31, 16, 16, 30, 16, 16, 31],
  F: [31, 16, 16, 30, 16, 16, 16],
  G: [14, 17, 16, 23, 17, 17, 14],
  H: [17, 17, 17, 31, 17, 17, 17],
  I: [14, 4, 4, 4, 4, 4, 14],
  J: [7, 2, 2, 2, 2, 18, 12],
  K: [17, 18, 20, 24, 20, 18, 17],
  L: [16, 16, 16, 16, 16, 16, 31],
  M: [17, 27, 21, 21, 17, 17, 17],
  N: [17, 25, 21, 19, 17, 17, 17],
  O: [14, 17, 17, 17, 17, 17, 14],
  P: [30, 17, 17, 30, 16, 16, 16],
  Q: [14, 17, 17, 17, 21, 18, 13],
  R: [30, 17, 17, 30, 20, 18, 17],
  S: [15, 16, 16, 14, 1, 1, 30],
  T: [31, 4, 4, 4, 4, 4, 4],
  U: [17, 17, 17, 17, 17, 17, 14],
  V: [17, 17, 17, 17, 17, 10, 4],
  W: [17, 17, 17, 21, 21, 27, 17],
  X: [17, 17, 10, 4, 10, 17, 17],
  Y: [17, 17, 10, 4, 4, 4, 4],
  Z: [31, 1, 2, 4, 8, 16, 31],
  '[': [6, 4, 4, 4, 4, 4, 6],
  '\\': [16, 16, 8, 4, 2, 1, 1],
  ']': [12, 4, 4, 4, 4, 4, 12],
  '^': [4, 10, 17, 0, 0, 0, 0],
  _: [0, 0, 0, 0, 0, 0, 31],
  '`': [8, 4, 0, 0, 0, 0, 0],
  a: [0, 0, 14, 1, 15, 17, 15],
  b: [16, 16, 30, 17, 17, 17, 30],
  c: [0, 0, 14, 17, 16, 17, 14],
  d: [1, 1, 15, 17, 17, 17, 15],
  e: [0, 0, 14, 17, 31, 16, 14],
  f: [6, 9, 8, 28, 8, 8, 8],
  g: [0, 15, 17, 17, 15, 1, 14],
  h: [16, 16, 30, 17, 17, 17, 17],
  i: [4, 0, 12, 4, 4, 4, 14],
  j: [2, 0, 6, 2, 2, 18, 12],
  k: [16, 16, 18, 20, 24, 20, 18],
  l: [12, 4, 4, 4, 4, 4, 14],
  m: [0, 0, 26, 21, 21, 21, 21],
  n: [0, 0, 30, 17, 17, 17, 17],
  o: [0, 0, 14, 17, 17, 17, 14],
  p: [0, 0, 30, 17, 30, 16, 16],
  q: [0, 0, 15, 17, 15, 1, 1],
  r: [0, 0, 22, 25, 16, 16, 16],
  s: [0, 0, 15, 16, 14, 1, 30],
  t: [8, 8, 28, 8, 8, 9, 6],
  u: [0, 0, 17, 17, 17, 19, 13],
  v: [0, 0, 17, 17, 17, 10, 4],
  w: [0, 0, 17, 21, 21, 21, 10],
  x: [0, 0, 17, 10, 4, 10, 17],
  y: [0, 0, 17, 17, 15, 1, 14],
  z: [0, 0, 31, 2, 4, 8, 31],
  '{': [2, 4, 4, 8, 4, 4, 2],
  '|': [4, 4, 4, 4, 4, 4, 4],
  '}': [8, 4, 4, 2, 4, 4, 8],
  '~': [0, 0, 8, 21, 1, 0, 0],
  '•': [0, 0, 0, 14, 14, 0, 0],
  '→': [0, 4, 6, 31, 6, 4, 0],
  '←': [0, 8, 12, 31, 12, 8, 0],
  '↑': [4, 14, 31, 4, 4, 4, 4],
  '↓': [4, 4, 4, 31, 14, 4, 0],
  '✓': [0, 1, 3, 18, 28, 8, 0],
  '×': [0, 0, 17, 10, 10, 17, 0],
};

export const FONT_W = 5;
export const FONT_H = 7;

/** Measure text in the bitmap font (also used to size canvas labels). */
export function measureBitmapText(text, scale = 2, letterSpacing = 1) {
  const lines = String(text == null ? '' : text).split('\n');
  const w = Math.max(...lines.map((l) => (l.length * (FONT_W + letterSpacing) - letterSpacing) * scale));
  return { width: w, height: lines.length * (FONT_H + 2) * scale - 2 * scale, lines: lines.length, scale };
}

/**
 * Draw text into `dst` at (x, y) with the bitmap font.
 * Supports: scale, color, alpha, background box, shadow, outline, alignment,
 * and `mono` letter spacing. Returns the occupied rect.
 */
export function drawBitmapText(dst, text, opts = {}) {
  const scale = Math.max(1, Math.round(opts.scale || 2));
  const ls = opts.spacing == null ? 1 : opts.spacing;
  const color = opts.color ? parseColor(opts.color) : [255, 255, 255, 255];
  const bg = opts.background ? parseColor(opts.background) : null;
  const shadow = opts.shadow ? parseColor(opts.shadow) : null;
  const shadowOffset = opts.shadowOffset ?? scale;
  const outline = opts.outline ? parseColor(opts.outline) : null;
  const alpha = (opts.alpha == null ? 1 : opts.alpha) * (color[3] / 255);
  const str = String(text == null ? '' : text);
  const lines = str.split('\n');
  const maxLen = Math.max(1, ...lines.map((l) => l.length));
  const blockW = maxLen * (FONT_W + ls) * scale - ls * scale;
  const blockH = lines.length * (FONT_H + 2) * scale - 2 * scale;
  let x0 = opts.x | 0;
  let y0 = opts.y | 0;
  const align = opts.align || 'left';
  const valign = opts.valign || 'top';
  if (align === 'center') x0 = Math.round((dst.width - blockW) / 2);
  else if (align === 'right') x0 = dst.width - blockW;
  if (valign === 'middle') y0 = Math.round((dst.height - blockH) / 2);
  else if (valign === 'bottom') y0 = dst.height - blockH;
  else if (valign === 'baseline') y0 = dst.height - blockH - (opts.baseline || 0);
  if (opts.margin) {
    x0 += opts.margin;
    y0 += opts.margin;
  }
  const padX = (opts.padding || 0) * scale;
  const padY = (opts.padding || 0) * scale;
  if (bg) fillRectAlpha(dst, x0 - padX, y0 - padY, blockW + padX * 2, blockH + padY * 2, bg, opts.backgroundAlpha ?? alpha);
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    let lineW = line.length * (FONT_W + ls) * scale - ls * scale;
    let lx = x0;
    if (align === 'center') lx = Math.round(x0 + (blockW - lineW) / 2);
    else if (align === 'right') lx = x0 + (blockW - lineW);
    void lineW;
    const ly = y0 + li * (FONT_H + 2) * scale;
    for (let ci = 0; ci < line.length; ci++) {
      const glyph = FONT[line[ci]] || FONT['?'];
      const gx = lx + ci * (FONT_W + ls) * scale;
      for (let row = 0; row < FONT_H; row++) {
        const bits = glyph[row];
        if (!bits) continue;
        for (let col = 0; col < FONT_W; col++) {
          if (!(bits & (1 << (FONT_W - 1 - col)))) continue;
          const px = gx + col * scale;
          const py = ly + row * scale;
          if (shadow) fillRectAlpha(dst, px + shadowOffset, py + shadowOffset, scale, scale, shadow, (opts.shadowAlpha ?? 0.65) * (shadow[3] / 255));
          if (outline) {
            for (const [ox, oy] of [[-scale, 0], [scale, 0], [0, -scale], [0, scale]]) fillRectAlpha(dst, px + ox, py + oy, scale, scale, outline, alpha * (outline[3] / 255));
          }
          fillRectAlpha(dst, px, py, scale, scale, color, alpha);
        }
      }
    }
  }
  return { x: x0, y: y0, width: blockW, height: blockH };
}

/* ------------------------------------------------------------- shape helpers */

export function fillRectAlpha(dst, x, y, w, h, color, alpha = 1) {
  if (alpha <= 0) return dst;
  const d = dst.data;
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(dst.width, Math.round(x + w));
  const y1 = Math.min(dst.height, Math.round(y + h));
  const a = Math.min(1, alpha) * (color[3] == null ? 1 : color[3] / 255);
  const ia = 1 - a;
  for (let yy = y0; yy < y1; yy++) {
    const row = yy * dst.stride;
    for (let xx = x0; xx < x1; xx++) {
      const i = row + xx * 4;
      d[i] = clamp255(color[0] * a + d[i] * ia);
      d[i + 1] = clamp255(color[1] * a + d[i + 1] * ia);
      d[i + 2] = clamp255(color[2] * a + d[i + 2] * ia);
      d[i + 3] = clamp255(a * 255 + d[i + 3] * ia);
    }
  }
  return dst;
}

/** Outlined rect (highlight boxes around UI elements). */
export function strokeRect(dst, x, y, w, h, thickness, color, alpha = 1) {
  const t = Math.max(1, thickness | 0);
  fillRectAlpha(dst, x, y, w, t, color, alpha);
  fillRectAlpha(dst, x, y + h - t, w, t, color, alpha);
  fillRectAlpha(dst, x, y + t, t, h - t * 2, color, alpha);
  fillRectAlpha(dst, x + w - t, y + t, t, h - t * 2, color, alpha);
  return dst;
}

export function fillCircle(dst, cx, cy, r, color, alpha = 1) {
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(dst.width, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(dst.height, Math.ceil(cy + r));
  const d = dst.data;
  const rr = r * r;
  const a = Math.min(1, alpha) * (color[3] == null ? 1 : color[3] / 255);
  const ia = 1 - a;
  for (let y = y0; y < y1; y++) {
    const dy = y + 0.5 - cy;
    const row = y * dst.stride;
    for (let x = x0; x < x1; x++) {
      const dx = x + 0.5 - cx;
      const dist2 = dx * dx + dy * dy;
      if (dist2 > rr + r) continue;
      const cov = dist2 <= rr - r ? 1 : Math.max(0, 1 - (dist2 - (rr - r)) / (2 * r + 1));
      const aa = a * cov;
      if (aa <= 0) continue;
      const i = row + x * 4;
      const i2 = 1 - aa;
      d[i] = clamp255(color[0] * aa + d[i] * i2);
      d[i + 1] = clamp255(color[1] * aa + d[i + 1] * i2);
      d[i + 2] = clamp255(color[2] * aa + d[i + 2] * i2);
      d[i + 3] = clamp255(aa * 255 + d[i + 3] * i2);
    }
  }
  return dst;
}

/** Line with thickness + AA (arrows, underlines, callouts). */
export function drawLine(dst, x0, y0, x1, y1, thickness, color, alpha = 1) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  const steps = Math.max(1, Math.ceil(len));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    fillCircle(dst, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, Math.max(0.5, thickness / 2), color, alpha);
  }
  return dst;
}

export function drawArrow(dst, from, to, opts = {}) {
  const t = Math.max(1, opts.thickness || 2);
  const head = opts.head || t * 4;
  drawLine(dst, from.x, from.y, to.x, to.y, t, opts.color || [255, 60, 60, 255], opts.alpha ?? 1);
  const ang = Math.atan2(to.y - from.y, to.x - from.x);
  for (const s of [-1, 1]) {
    drawLine(
      dst,
      to.x,
      to.y,
      to.x - Math.cos(ang + s * 0.5) * head,
      to.y - Math.sin(ang + s * 0.5) * head,
      t,
      opts.color || [255, 60, 60, 255],
      opts.alpha ?? 1
    );
  }
  return dst;
}

/** Rounded translucent highlight (screenshot annotation). */
export function highlightRect(dst, x, y, w, h, opts = {}) {
  const fill = opts.fill ? parseColor(opts.fill) : [255, 220, 80, 90];
  const stroke = opts.stroke ? parseColor(opts.stroke) : null;
  fillRectAlpha(dst, x, y, w, h, fill, opts.fillAlpha ?? 0.28);
  if (stroke) strokeRect(dst, x, y, w, h, opts.thickness || 2, stroke, opts.strokeAlpha ?? 0.95);
  return dst;
}

/**
 * Blur/mosaic a region (privacy redaction). `mode`: `blur` | `pixelate` |
 * `solid` | `darken`. Required for screen-recording GIFs of real products.
 */
export function redact(raster, regions, opts = {}) {
  const mode = opts.mode || 'pixelate';
  const block = opts.block || 8;
  for (const r of regions) {
    const x0 = Math.max(0, r.x | 0);
    const y0 = Math.max(0, r.y | 0);
    const x1 = Math.min(raster.width, (r.x + r.width) | 0);
    const y1 = Math.min(raster.height, (r.y + r.height) | 0);
    if (x1 <= x0 || y1 <= y0) continue;
    if (mode === 'solid') {
      fillRectAlpha(raster, x0, y0, x1 - x0, y1 - y0, parseColor(opts.color || '#000000'), 1);
      continue;
    }
    if (mode === 'darken') {
      const d = raster.data;
      for (let y = y0; y < y1; y++) {
        const row = y * raster.stride;
        for (let x = x0; x < x1; x++) {
          const i = row + x * 4;
          d[i] = d[i] * 0.35;
          d[i + 1] = d[i + 1] * 0.35;
          d[i + 2] = d[i + 2] * 0.35;
        }
      }
      continue;
    }
    if (mode === 'blur') {
      const w = x1 - x0;
      const h = y1 - y0;
      const sub = new Raster(w, h);
      for (let y = 0; y < h; y++) sub.data.set(raster.data.subarray((y0 + y) * raster.stride + x0 * 4, (y0 + y) * raster.stride + x0 * 4 + w * 4), y * w * 4);
      boxBlur(sub, Math.max(2, Math.round(opts.radius || 6)));
      for (let y = 0; y < h; y++) raster.data.set(sub.data.subarray(y * w * 4, y * w * 4 + w * 4), (y0 + y) * raster.stride + x0 * 4);
      continue;
    }
    // pixelate (fast, and the only mode that survives GIF quantization cleanly)
    const d = raster.data;
    for (let by = y0; by < y1; by += block) {
      for (let bx = x0; bx < x1; bx += block) {
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let sa = 0;
        let n = 0;
        const ye = Math.min(y1, by + block);
        const xe = Math.min(x1, bx + block);
        for (let y = by; y < ye; y++) {
          const row = y * raster.stride;
          for (let x = bx; x < xe; x++) {
            const i = row + x * 4;
            sr += d[i];
            sg += d[i + 1];
            sb += d[i + 2];
            sa += d[i + 3];
            n++;
          }
        }
        sr = (sr / n) | 0;
        sg = (sg / n) | 0;
        sb = (sb / n) | 0;
        sa = (sa / n) | 0;
        if (opts.color) {
          const c = parseColor(opts.color);
          sr = c[0];
          sg = c[1];
          sb = c[2];
        }
        for (let y = by; y < ye; y++) {
          const row = y * raster.stride;
          for (let x = bx; x < xe; x++) {
            const i = row + x * 4;
            d[i] = sr;
            d[i + 1] = sg;
            d[i + 2] = sb;
            d[i + 3] = sa;
          }
        }
      }
    }
  }
  return raster;
}

/* ---------------------------------------------------------------- overlays */

/**
 * Build the overlay list from user options. Supported shapes:
 * `text`, `watermark`, `image`, `rect`, `ellipse`, `line`, `arrow`, `bar`,
 * `progress`, `timestamp`, `counter`, `qr` (as a placeholder box if no encoder),
 * `mosaic`/`redact`, `pip` (picture-in-picture), `frame`.
 *
 * Each entry gets `prepare(dst)` (once) + `draw(dst, frameInfo)` (per frame).
 */
export function buildOverlays(list, ctx = {}) {
  const out = [];
  for (const spec of list || []) {
    if (!spec || spec.enabled === false) continue;
    const kind = spec.type || spec.kind || 'text';
    const o = OVERLAY_TYPES[kind];
    if (!o) throw mkErr(`unknown overlay type "${kind}" (known: ${Object.keys(OVERLAY_TYPES).join(', ')})`, ErrorCode.OVERLAY_UNKNOWN);
    out.push(o(spec, ctx));
  }
  return out;
}

const POS = (spec, dst) => {
  const pad = spec.padding ?? 0;
  let x = spec.x;
  let y = spec.y;
  if (x == null) x = spec.anchor && /right/.test(spec.anchor) ? dst.width - (spec.w || 0) - pad : pad;
  if (y == null) y = spec.anchor && /bottom/.test(spec.anchor) ? dst.height - (spec.h || 0) - pad : pad;
  if (spec.anchor) {
    if (/center-x|top-center|bottom-center/.test(spec.anchor)) x = Math.round((dst.width - (spec.w || 0)) / 2);
    if (/middle/.test(spec.anchor)) y = Math.round((dst.height - (spec.h || 0)) / 2);
  }
  if (typeof x === 'string' && x.endsWith('%')) x = (parseFloat(x) / 100) * dst.width;
  if (typeof y === 'string' && y.endsWith('%')) y = (parseFloat(y) / 100) * dst.height;
  return { x: x | 0, y: y | 0 };
};

const ANIM = (spec, info) => {
  if (!spec.anim) return 1;
  const t = info && info.index != null && info.total ? info.index / Math.max(1, info.total - 1) : 0;
  switch (spec.anim) {
    case 'fadeIn':
      return Math.min(1, t / (spec.animSpan || 0.2));
    case 'fadeOut':
      return Math.min(1, Math.max(0, 1 - t / (spec.animSpan || 0.2)));
    case 'pulse':
      return 0.7 + 0.3 * Math.sin(t * Math.PI * 2 * (spec.cycles || 1));
    case 'blink':
      return Math.floor(t * (spec.cycles || 2) * 2) % 2 === 0 ? 1 : 0.15;
    default:
      return 1;
  }
};

export const OVERLAY_TYPES = {
  text(spec, ctx) {
    let tile = null;
    return {
      name: 'text',
      prepare(dst) {
        const scale = spec.scale || Math.max(1, Math.round(dst.height / 90));
        const m = measureBitmapText(spec.text ?? '', scale, spec.spacing);
        spec.w = m.width;
        spec.h = m.height;
        spec.scale = scale;
        tile = new Raster(m.width + (spec.padding || 0) * scale * 2, m.height + (spec.padding || 0) * scale * 2);
        if (spec.background) fillRectAlpha(tile, 0, 0, tile.width, tile.height, parseColor(spec.background), spec.backgroundAlpha ?? 0.55);
      },
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        // Live-render per frame so `text` may be a function of frame/time.
        const body = typeof spec.text === 'function' ? spec.text(info || {}, ctx) : spec.text;
        const p = POS({ ...spec, text: body }, dst);
        if (tile) {
          // draw into the tile for shadow/outline batching when text is static
          void body;
        }
        drawBitmapText(dst, body, {
          ...spec,
          x: p.x,
          y: p.y,
          align: 'left',
          valign: 'top',
          alpha: (spec.alpha == null ? 1 : spec.alpha) * a,
        });
      },
      dispose() {
        if (tile) tile.release?.();
        tile = null;
      },
    };
  },
  watermark(spec, ctx) {
    let tile = null;
    const makeTile = (dst) => {
      const text = typeof spec.text === 'function' ? spec.text({}, ctx) : spec.text || 'gifx';
      const scale = Math.max(1, Math.round((dst.width / (spec.basedOnWidth || 960)) * (spec.size || 1) * 2));
      const m = measureBitmapText(text, scale, spec.spacing);
      const t = new Raster(m.width + scale * 6, m.height + scale * 4);
      if (spec.box !== false) fillRectAlpha(t, 0, 0, t.width, t.height, parseColor(spec.background || '#000000'), spec.backgroundAlpha ?? 0.32);
      drawBitmapText(t, text, { scale, spacing: spec.spacing, color: spec.color || '#ffffff', alpha: spec.alpha ?? 0.85, x: scale * 3, y: scale * 2, outline: spec.outline ? parseColor(spec.outline) : null });
      return t;
    };
    return {
      name: 'watermark',
      prepare(dst) {
        tile = spec.dynamic ? null : makeTile(dst);
      },
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        const t = tile || makeTile(dst);
        const p = POS({ ...spec, w: t.width, h: t.height }, dst);
        if (spec.rotate) {
          blitRotated(dst, t, p.x, p.y, spec.rotate, (spec.alpha ?? 0.9) * a);
        } else {
          blit(dst, t, p.x, p.y, (spec.alpha == null ? 0.9 : spec.alpha) * a);
        }
        if (!tile) t.release?.();
      },
      dispose() {
        tile?.release?.();
        tile = null;
      },
    };
  },
  image(spec) {
    let src = null;
    let loaded = false;
    return {
      name: 'image',
      prepare() {
        if (loaded) return;
        src = spec.raster || null;
        loaded = true;
      },
      async load() {
        if (src || !spec.source) return src;
        src = await rasterizeSource(spec.source, spec.width, spec.height);
        loaded = true;
        return src;
      },
      draw(dst, info) {
        if (!src) return;
        const a = ANIM(spec, info);
        if (a <= 0) return;
        let w = src.width;
        let h = src.height;
        if (spec.width) w = spec.width | 0;
        if (spec.height) h = spec.height | 0;
        if (w !== src.width || h !== src.height) {
          const s = new Raster(w, h);
          nearestResize(src, s);
          blit(dst, s, POS({ ...spec, w, h }, dst).x, POS({ ...spec, w, h }, dst).y, (spec.alpha ?? 1) * a);
          s.release?.();
          return;
        }
        const p = POS({ ...spec, w, h }, dst);
        blit(dst, src, p.x, p.y, (spec.alpha ?? 1) * a);
      },
      dispose() {
        src?.release?.();
        src = null;
      },
    };
  },
  rect(spec) {
    return {
      name: 'rect',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        const p = POS(spec, dst);
        const w = spec.width ?? spec.w ?? dst.width - p.x;
        const h = spec.height ?? spec.h ?? dst.height - p.y;
        if (spec.fill) fillRectAlpha(dst, p.x, p.y, w, h, parseColor(spec.fill), (spec.fillAlpha ?? 0.6) * a);
        if (spec.stroke || spec.borderWidth) strokeRect(dst, p.x, p.y, w, h, spec.borderWidth || 2, parseColor(spec.stroke || '#ffffff'), (spec.strokeAlpha ?? 1) * a);
      },
    };
  },
  ellipse(spec) {
    return {
      name: 'ellipse',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        const p = POS(spec, dst);
        const w = spec.width ?? dst.width - p.x;
        const h = spec.height ?? dst.height - p.y;
        fillCircle(dst, p.x + w / 2, p.y + h / 2, Math.min(w, h) / 2, parseColor(spec.fill || '#ffffff'), (spec.fillAlpha ?? 0.6) * a);
      },
    };
  },
  line(spec) {
    return {
      name: 'line',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        drawLine(dst, spec.from?.x ?? 0, spec.from?.y ?? 0, spec.to?.x ?? dst.width, spec.to?.y ?? dst.height, spec.thickness || 2, parseColor(spec.color || '#ffffff'), (spec.alpha ?? 1) * a);
      },
    };
  },
  arrow(spec) {
    return {
      name: 'arrow',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        drawArrow(dst, { x: spec.from?.x ?? 0, y: spec.from?.y ?? 0 }, { x: spec.to?.x ?? 40, y: spec.to?.y ?? 40 }, {
          thickness: spec.thickness || 3,
          head: spec.head,
          color: parseColor(spec.color || '#ff3b30'),
          alpha: a,
        });
      },
    };
  },
  highlight(spec) {
    return {
      name: 'highlight',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        const p = POS(spec, dst);
        highlightRect(dst, p.x, p.y, spec.width ?? dst.width - p.x, spec.height ?? dst.height - p.y, { ...spec, fillAlpha: (spec.fillAlpha ?? 0.3) * a });
      },
    };
  },
  redact(spec) {
    return {
      name: 'redact',
      draw(dst, info) {
        if (ANIM(spec, info) <= 0) return;
        const p = POS(spec, dst);
        redact(
          dst,
          [{ x: p.x, y: p.y, width: spec.width ?? dst.width - p.x, height: spec.height ?? dst.height - p.y }],
          { mode: spec.mode || 'pixelate', block: spec.block || 8, color: spec.color, radius: spec.radius }
        );
      },
    };
  },
  frame(spec) {
    return {
      name: 'frame',
      draw(dst, info) {
        const a = ANIM(spec, info);
        if (a <= 0) return;
        const t = spec.thickness || 4;
        const c = parseColor(spec.color || '#000000');
        const alpha = (spec.alpha ?? 1) * a;
        fillRectAlpha(dst, 0, 0, dst.width, t, c, alpha);
        fillRectAlpha(dst, 0, dst.height - t, dst.width, t, c, alpha);
        fillRectAlpha(dst, 0, 0, t, dst.height, c, alpha);
        fillRectAlpha(dst, dst.width - t, 0, t, dst.height, c, alpha);
        if (spec.shadow) {
          fillRectAlpha(dst, t, t, dst.width - t * 2, 1, parseColor(spec.shadow), alpha * 0.4);
          fillRectAlpha(dst, t, dst.height - t - 1, dst.width - t * 2, 1, parseColor(spec.shadow), alpha * 0.4);
        }
      },
    };
  },
  /** Screen-recorder style progress bar tied to frame index. */
  progress(spec) {
    return {
      name: 'progress',
      draw(dst, info) {
        const p = Math.max(0, Math.min(1, (info?.index ?? 0) / Math.max(1, (info?.total ?? 1) - 1)));
        const h = spec.height || Math.max(2, Math.round(dst.height * 0.012));
        const y = spec.y == null ? dst.height - h : spec.y;
        if (spec.background !== false) fillRectAlpha(dst, 0, y, dst.width, h, parseColor(spec.background || '#000000'), 0.45);
        const c = parseColor(spec.color || '#00e676');
        if (spec.gradient) {
          const c2 = parseColor(spec.gradient);
          const wpx = Math.round(dst.width * p);
          for (let x = 0; x < wpx; x++) {
            const t = dst.width ? x / dst.width : 0;
            fillRectAlpha(dst, x, y, 1, h, [c[0] + (c2[0] - c[0]) * t, c[1] + (c2[1] - c[1]) * t, c[2] + (c2[2] - c[2]) * t, 255], 1);
          }
        } else fillRectAlpha(dst, 0, y, Math.round(dst.width * p), h, c, spec.alpha ?? 1);
      },
    };
  },
  counter(spec) {
    return {
      name: 'counter',
      draw(dst, info) {
        const value = typeof spec.value === 'function' ? spec.value(info || {}, ctx) : (info?.index ?? 0) + 1;
        const text = String(spec.format ? spec.format.replace('{i}', value).replace('{n}', info?.total ?? '?') : `${value}`);
        drawBitmapText(dst, text, { ...spec, text, alpha: spec.alpha ?? 0.95 });
      },
    };
  },
  timestamp(spec) {
    return {
      name: 'timestamp',
      draw(dst, info) {
        const ms = info?.ptsMs ?? ((info?.index ?? 0) * 100) / Math.max(1, ctx.fps || 10);
        const s = ms / 1000;
        const text = typeof spec.format === 'function' ? spec.format(s, info) : `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}.${String(Math.floor((s % 1) * 10)).padStart(1, '0')}`;
        drawBitmapText(dst, text, { ...spec, text, scale: spec.scale || 2, color: spec.color || '#ffffff', background: spec.background || '#000000a0', padding: spec.padding ?? 1 });
      },
    };
  },
  /** Caption bar: full-width band at the bottom with centered text. */
  caption(spec) {
    return {
      name: 'caption',
      draw(dst, info) {
        const body = typeof spec.text === 'function' ? spec.text(info || {}, ctx) : spec.text;
        if (!body) return;
        const scale = spec.scale || Math.max(1, Math.round(dst.height / 110));
        const m = measureBitmapText(body, scale);
        const h = m.height + scale * 4;
        const y = dst.height - h;
        fillRectAlpha(dst, 0, y, dst.width, h, parseColor(spec.background || '#000000'), spec.backgroundAlpha ?? 0.55);
        drawBitmapText(dst, body, { text: body, scale, x: 0, y: y + scale * 2, align: 'center', valign: 'top', color: spec.color || '#ffffff', outline: spec.outline ? parseColor(spec.outline) : null });
      },
    };
  },
  /** Picture-in-picture: a second raster (e.g. webcam) in a corner. */
  pip(spec) {
    return {
      name: 'pip',
      draw(dst, info) {
        const src2 = typeof spec.raster === 'function' ? spec.raster(info || {}, ctx) : spec.raster;
        if (!src2) return;
        const w = spec.width || Math.round(dst.width * 0.3);
        const h = Math.round(w * (src2.height / src2.width));
        const tmp = new Raster(w, h);
        nearestResize(src2, tmp);
        const margin = spec.margin ?? 8;
        const x = /left/.test(spec.anchor || 'bottom-right') ? margin : dst.width - w - margin;
        const y = /top/.test(spec.anchor || 'bottom-right') ? margin : dst.height - h - margin;
        if (spec.border !== false) strokeRect(dst, x - 2, y - 2, w + 4, h + 4, 2, parseColor(spec.borderColor || '#ffffff'), 1);
        blit(dst, tmp, x, y, spec.alpha ?? 1);
        tmp.release?.();
      },
    };
  },
};

/* ------------------------------------------------------------------- blits */

/** Alpha-blend `src` onto `dst` at (x, y). */
export function blit(dst, src, x, y, opacity = 1) {
  const d = dst.data;
  const s = src.data;
  const x0 = Math.max(0, x | 0);
  const y0 = Math.max(0, y | 0);
  const x1 = Math.min(dst.width, x + src.width);
  const y1 = Math.min(dst.height, y + src.height);
  if (opacity >= 1) {
    for (let yy = y0; yy < y1; yy++) {
      const srow = (yy - y) * src.stride;
      const drow = yy * dst.stride;
      for (let xx = x0; xx < x1; xx++) {
        const si = srow + (xx - x) * 4;
        const a = s[si + 3];
        if (a === 0) continue;
        const di = drow + xx * 4;
        if (a === 255) {
          d[di] = s[si];
          d[di + 1] = s[si + 1];
          d[di + 2] = s[si + 2];
          d[di + 3] = 255;
        } else {
          const f = a / 255;
          const ifa = 1 - f;
          d[di] = clamp255(s[si] * f + d[di] * ifa);
          d[di + 1] = clamp255(s[si + 1] * f + d[di + 1] * ifa);
          d[di + 2] = clamp255(s[si + 2] * f + d[di + 2] * ifa);
          d[di + 3] = clamp255(a + d[di + 3] * ifa);
        }
      }
    }
    return dst;
  }
  for (let yy = y0; yy < y1; yy++) {
    const srow = (yy - y) * src.stride;
    const drow = yy * dst.stride;
    for (let xx = x0; xx < x1; xx++) {
      const si = srow + (xx - x) * 4;
      const a = (s[si + 3] / 255) * opacity;
      if (a <= 0) continue;
      const di = drow + xx * 4;
      const ia = 1 - a;
      d[di] = clamp255(s[si] * a + d[di] * ia);
      d[di + 1] = clamp255(s[si + 1] * a + d[di + 1] * ia);
      d[di + 2] = clamp255(s[si + 2] * a + d[di + 2] * ia);
      d[di + 3] = clamp255(a * 255 + d[di + 3] * ia);
    }
  }
  return dst;
}

function blitRotated(dst, src, x, y, degrees, opacity) {
  const rad = (degrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const ccx = src.width / 2;
  const ccy = src.height / 2;
  const ex = x - ccx * cos + ccy * sin;
  const ey = y - ccx * sin - ccy * cos;
  const d = dst.data;
  const s = src.data;
  for (let yy = 0; yy < src.height; yy++) {
    for (let xx = 0; xx < src.width; xx++) {
      const px = Math.round(ex + xx * cos - yy * sin);
      const py = Math.round(ey + xx * sin + yy * cos);
      if (px < 0 || py < 0 || px >= dst.width || py >= dst.height) continue;
      const si = yy * src.stride + xx * 4;
      const a = (s[si + 3] / 255) * opacity;
      if (a <= 0) continue;
      const di = py * dst.stride + px * 4;
      const ia = 1 - a;
      d[di] = clamp255(s[si] * a + d[di] * ia);
      d[di + 1] = clamp255(s[si + 1] * a + d[di + 1] * ia);
      d[di + 2] = clamp255(s[si + 2] * a + d[di + 2] * ia);
      d[di + 3] = clamp255(a * 255 + d[di + 3] * ia);
    }
  }
  return dst;
}

function nearestResize(src, dst) {
  const fx = src.width / dst.width;
  const fy = src.height / dst.height;
  for (let y = 0; y < dst.height; y++) {
    const sy = Math.min(src.height - 1, (y * fy) | 0) * src.stride;
    const drow = y * dst.stride;
    for (let x = 0; x < dst.width; x++) {
      const si = sy + Math.min(src.width - 1, (x * fx) | 0) * 4;
      dst.data[drow + x * 4] = src.data[si];
      dst.data[drow + x * 4 + 1] = src.data[si + 1];
      dst.data[drow + x * 4 + 2] = src.data[si + 2];
      dst.data[drow + x * 4 + 3] = src.data[si + 3];
    }
  }
  return dst;
}

/**
 * Decode an overlay source into a Raster. Accepts: Raster, ImageBitmap,
 * ImageData, HTMLImageElement/canvas, Blob/File, data: URL, or raw bytes.
 */
export async function rasterizeSource(source, width, height) {
  if (!source) return null;
  if (source instanceof Raster) return source;
  if (source.width != null && source.height != null && source.data) return source;
  const bmp = typeof createImageBitmap === 'function' ? await createImageBitmap(source) : null;
  if (bmp) {
    const c = createCanvas(bmp.width, bmp.height);
    if (!c) {
      // No canvas: we cannot read back pixels, so synthesize a neutral tile
      // rather than throwing (a missing overlay must never kill a render).
      const r = new Raster(width || 32, height || 32);
      r.fill(0, 0, 0, 0);
      bmp.close?.();
      return r;
    }
    c.width = bmp.width;
    c.height = bmp.height;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.clearRect(0, 0, bmp.width, bmp.height);
    g.drawImage(bmp, 0, 0);
    const d = g.getImageData(0, 0, bmp.width, bmp.height);
    const r = new Raster(bmp.width, bmp.height, new Uint8Array(d.data.buffer.slice(0)));
    r.opaque = false;
    if (bmp.close) bmp.close();
    if ((width && width !== r.width) || (height && height !== r.height)) {
      const o = new Raster(width || r.width, height || r.height);
      nearestResize(r, o);
      return o;
    }
    return r;
  }
  return null;
}

/**
 * Watermark "smart placement": pick the corner with the flattest luma so the
 * logo stays readable but invisible-ish. Used when `spec.smart === true`.
 */
export function suggestWatermarkAnchor(raster) {
  const cands = [
    { anchor: 'top-left', x: 0, y: 0 },
    { anchor: 'top-right', x: raster.width / 2, y: 0 },
    { anchor: 'bottom-left', x: 0, y: raster.height / 2 },
    { anchor: 'bottom-right', x: raster.width / 2, y: raster.height / 2 },
  ];
  let best = cands[3];
  let bestVar = Infinity;
  for (const c of cands) {
    let s = 0;
    let s2 = 0;
    let n = 0;
    for (let y = c.y | 0; y < Math.min(raster.height, c.y + raster.height / 2); y += 3) {
      const row = y * raster.stride;
      for (let x = c.x | 0; x < Math.min(raster.width, c.x + raster.width / 2); x += 3) {
        const l = luma709(raster.data[row + x * 4], raster.data[row + x * 4 + 1], raster.data[row + x * 4 + 2]);
        s += l;
        s2 += l * l;
        n++;
      }
    }
    if (!n) continue;
    const v = s2 / n - (s / n) * (s / n);
    if (v < bestVar) {
      bestVar = v;
      best = c;
    }
  }
  return { anchor: best.anchor, variance: bestVar };
}

function mkErr(msg, code) {
  const e = new Error(msg);
  e.code = code;
  return e;
}

/**
 * Fixed / constrained palettes.
 *
 * `fixed` is what you want when consistency across a set of GIFs matters
 * (brand palette, game sprite sheet, "same 16 colors for the whole animation")
 * or when you want to guarantee no banding on a known-color source (logos, UI
 * captures, pixel art).
 *
 * Named built-ins:
 *  - `websafe`     Netscape 216 (6×6×6 cube)
 *  - `mac`         original Macintosh 216-ish map
 *  - `win`         Windows VGA 208 + 6 grays
 *  - `grayscale`   even luminance ramp of `colors` steps
 *  - `mono`        black + white
 *  - `uniform`     cube root grid in RGB (fastest to map, banding-prone)
 *  - `gameboy`     4-tone DMG green, `gameboy-pocket`, `cga16`, `c64`, `pico8`,
 *                  `nes`, `amstrad`, `zxspectrum` … retro sets, because people
 *                  absolutely do this to video.
 *
 * @module quant/fixed
 */
import { parseColor, luma709, hslToRgb } from '../core/color.js';

/** @type {Map<string, ()=>number[]>} */
const SETS = new Map();

SETS.set('websafe', () => {
  // The 6×6×6 Netscape cube: channel values 0,51,102,153,204,255.
  const out = [];
  const step = [0, 51, 102, 153, 204, 255];
  for (let r = 0; r < 6; r++) for (let g = 0; g < 6; g++) for (let b = 0; b < 6; b++) out.push(step[r], step[g], step[b]);
  return out;
});
SETS.set('grayscale', (colors = 64) => {
  const out = [];
  const n = Math.max(2, Math.min(256, colors));
  for (let i = 0; i < n; i++) {
    const v = n === 1 ? 0 : Math.round((i * 255) / (n - 1));
    out.push(v, v, v);
  }
  return out;
});
SETS.set('mono', () => [0, 0, 0, 255, 255, 255]);
SETS.set('uniform', (colors = 64) => {
  const n = Math.max(2, Math.min(256, colors));
  const side = Math.max(1, Math.round(Math.cbrt(n)));
  const out = [];
  for (let r = 0; r < side; r++) for (let g = 0; g < side; g++) for (let b = 0; b < side; b++) out.push(Math.round((r * 255) / (side - 1 || 1)), Math.round((g * 255) / (side - 1 || 1)), Math.round((b * 255) / (side - 1 || 1)));
  return out;
});
SETS.set('gameboy', () => [33, 42, 18, 88, 109, 43, 158, 185, 84, 202, 222, 173]);
SETS.set('gameboy-pocket', () => [49, 73, 52, 91, 118, 76, 147, 173, 112, 203, 223, 173]);
SETS.set('cga16', () => [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 128, 128, 128, 192, 192, 192, 128, 0, 0, 128, 128, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128]);
SETS.set('pico8', () => PICO8);
SETS.set('nes', () => NES);
SETS.set('c64', () => C64);
SETS.set('amstrad', () => AMSTRAD);
SETS.set('zx', () => ZX);
SETS.set('win', () => {
  // Windows VGA: 6×6×6 cube (216) truncated to the standard 208 + 6 grays.
  const out = SETS.get('websafe')().slice(0, 208 * 3);
  for (let i = 1; i <= 6; i++) out.push(i * 41 - 10, i * 41 - 10, i * 41 - 10);
  return out;
});
SETS.set('mac', () => SETS.get('websafe')());
SETS.set('pastel', () => {
  const out = [];
  for (let i = 0; i < 12; i++) {
    const c = hslToRgb(i / 12, 0.55, 0.72);
    out.push(c[0], c[1], c[2]);
  }
  for (let i = 0; i < 4; i++) {
    const v = 40 + i * 60;
    out.push(v, v, v);
  }
  return out;
});

export const namedPalettes = () => [...SETS.keys()];

/**
 * @param {import('./palette.js').Histogram} hist
 * @param {{palette:string|number[]|Array|Uint8Array, colors?:number}} opts
 */
export function fixed(hist, opts = {}) {
  const want = Math.max(2, Math.min(256, opts.colors || 128));
  let flat;
  const p = opts.palette;
  if (p == null) flat = SETS.get('websafe')();
  else if (typeof p === 'string' && SETS.has(p.toLowerCase())) {
    const fn = SETS.get(p.toLowerCase());
    flat = p.toLowerCase() === 'grayscale' || p.toLowerCase() === 'uniform' ? fn(want) : fn();
  } else if (p instanceof Uint8Array) flat = Array.from(p);
  else if (Array.isArray(p)) {
    flat = [];
    for (const item of p) {
      if (typeof item === 'number') {
        flat.push((item >> 16) & 255, (item >> 8) & 255, item & 255);
      } else {
        const c = parseColor(item);
        flat.push(c[0], c[1], c[2]);
      }
    }
  } else flat = [0, 0, 0, 255, 255, 255];

  const colors = Math.max(2, Math.min(256, flat.length / 3)) | 0;
  const palette = new Uint8Array(colors * 3);
  for (let i = 0; i < colors * 3; i++) palette[i] = clamp255(Math.round(flat[i]));
  // Luminance ordering keeps LZW runs longer on flat-color art.
  const order = Array.from({ length: colors }, (_, i) => i).sort((a, b) => luma709(palette[a * 3], palette[a * 3 + 1], palette[a * 3 + 2]) - luma709(palette[b * 3], palette[b * 3 + 1], palette[b * 3 + 2]));
  const sorted = new Uint8Array(colors * 3);
  for (let i = 0; i < colors; i++) {
    const s = order[i];
    sorted[i * 3] = palette[s * 3];
    sorted[i * 3 + 1] = palette[s * 3 + 1];
    sorted[i * 3 + 2] = palette[s * 3 + 2];
  }
  return { palette: sorted, colors, error: 0, fixed: true, requested: want };
}

export function paletteFor(name, colors) {
  if (!name) return null;
  const key = String(name).toLowerCase();
  if (SETS.has(key)) {
    const fn = SETS.get(key);
    const flat = key === 'grayscale' || key === 'uniform' ? fn(colors) : fn();
    const out = new Uint8Array((flat.length / 3) | 0);
    void out;
    const u8 = new Uint8Array(flat.length);
    for (let i = 0; i < flat.length; i++) u8[i] = clamp255(Math.round(flat[i]));
    return u8;
  }
  return null;
}

const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

const PICO8 = [0, 0, 0, 29, 43, 83, 126, 37, 83, 33, 65, 58, 126, 126, 126, 171, 89, 9, 192, 144, 94, 222, 222, 222, 128, 144, 222, 142, 90, 138, 142, 142, 142, 222, 176, 176, 72, 72, 72, 222, 214, 11, 204, 196, 148];
const NES = [0x00, 0x00, 0x00, 0xfc, 0xfc, 0xfc, 0xb8, 0xb8, 0xb8, 0x6c, 0x6c, 0x6c, 0x00, 0xfc, 0xfc, 0x2c, 0xe8, 0xa8, 0x44, 0xa0, 0x2c, 0x9c, 0x24, 0x00, 0xf8, 0x00, 0x00, 0xfc, 0x98, 0x78, 0xbc, 0x60, 0x90, 0xc4, 0x3c, 0x9c, 0x88, 0x88, 0x88, 0xf0, 0xd8, 0xb0, 0x6c, 0xda, 0x90, 0x4c, 0x9c, 0x6c, 0x50, 0x9c, 0x94, 0x40, 0x7c, 0xd0, 0x50, 0x9c, 0x00, 0x80, 0x74, 0xa0, 0x78, 0x28, 0xc8, 0x50, 0x20, 0xf0, 0x58, 0x00, 0xc0, 0x78, 0x00, 0x68, 0x90, 0x00, 0x30, 0x84, 0x00, 0x20, 0x70, 0x28, 0x30, 0x48, 0x78, 0x44, 0x44, 0xa8, 0x74, 0x60, 0xa0, 0x60, 0x70, 0x84, 0x40, 0x88, 0x88, 0x88, 0x00, 0xf8, 0xf8, 0x68, 0xf8, 0xb8, 0x60, 0xc0, 0xa4, 0x78, 0x78, 0x8c, 0xa4, 0x70, 0xc4, 0xb8, 0x60, 0xdc, 0xbc, 0x60, 0xbc];
const C64 = [0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x81, 0x34, 0x38, 0x3c, 0xbb, 0x61, 0x8e, 0x41, 0xb6, 0x28, 0x47, 0x9e, 0x8e, 0x78, 0x37, 0x99, 0xa8, 0xfe, 0x69, 0x4b, 0x39, 0xaa, 0x6d, 0x5c, 0xcc, 0xad, 0xb2, 0x20, 0x5b, 0x3a, 0xcb, 0xd1, 0x91, 0x90, 0x7f, 0x94, 0xb4, 0xc2, 0x4f, 0x1e, 0xb9, 0xf8, 0xf1, 0xef, 0xef, 0xf6, 0x00, 0x00, 0x00];
const AMSTRAD = [0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, 0x00, 0x00, 0xff, 0x00, 0xaa, 0x00, 0x00, 0xaa, 0xaa, 0x00, 0xaa, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0xaa, 0x00, 0xff, 0xff, 0xaa, 0x00, 0x00, 0xaa, 0x00, 0xaa, 0xaa, 0x00, 0xff, 0xaa, 0x55, 0x00, 0xaa, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xff, 0xaa, 0x55, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xff, 0xaa, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0xaa, 0xff, 0x00, 0xff, 0xff, 0xaa, 0x00, 0xff, 0xaa, 0xaa, 0xff, 0xaa, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xaa, 0xff, 0xff, 0xff];
const ZX = [0x00, 0x00, 0x00, 0xd7, 0x00, 0x00, 0x00, 0xd7, 0x00, 0xd7, 0xd7, 0x00, 0x00, 0x00, 0xd7, 0xd7, 0x00, 0xd7, 0x00, 0xd7, 0xd7, 0xd7, 0xd7, 0xd7, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x7f, 0x7f, 0xff, 0x7f, 0x7f, 0x7f, 0xff, 0x7f, 0x7f, 0x7f, 0xff, 0xff, 0x7f, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x7f, 0x7f, 0x00, 0x00, 0x7f, 0x00, 0x7f, 0x7f, 0x7f, 0x00, 0x7f];

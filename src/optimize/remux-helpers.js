/**
 * GIFX Kernel — helpers for the re-encoder.
 *
 * The interesting one is `buildGlobalPalette()`. Sub-framing and
 * "unchanged → transparent" only work if every frame shares one index space:
 * a pixel that is unchanged between frames must keep the *same index*, or the
 * viewer would need the previous frame's local palette to interpret it. gifsicle
 * does the same thing (one global color map when the total fits, local maps
 * otherwise), and the fallback matters: screen recordings with dithered gradients
 * routinely exceed 256 distinct colors, and a re-encoder that assumes otherwise
 * corrupts the image.
 *
 * @module optimize/remux-helpers
 */
import { quantize, buildHistogram } from '../quant/palette.js';
import { createMapper, mapFrame, compactPalette, buildRemap } from '../quant/mapper.js';

export { createMapper, mapFrame, compactPalette, buildRemap };

/**
 * Collect the exact set of colors used by all frames (via their palettes and the
 * transparent indices), deduplicated.
 *
 * @param {object[]} frames parseGif frames
 * @param {Uint8Array} [globalPalette] the file's global color table
 * @param {number} [limit=256]
 * @param {object} [opts] `{background:[r,g,b], includeUnusedGlobal:false}`
 * @returns {{ok:boolean, palette:Uint8Array, colors:number, remaps:Uint8Array[],
 *   used:Uint8Array, overflow:number}} `remaps[i]` maps frame i's local index to
 *   the global index; `overflow` counts colors that did not fit.
 */
export function buildGlobalPalette(frames, globalPalette, opts = {}) {
  const limit = opts.limit || 256;
  const keyOf = (r, g, b) => ((r & 255) << 16) | ((g & 255) << 8) | (b & 255);
  const map = new Map();
  const rgb = [];
  const add = (r, g, b) => {
    const k = keyOf(r, g, b);
    let idx = map.get(k);
    if (idx === undefined) {
      if (rgb.length / 3 >= limit) return -1;
      idx = rgb.length / 3;
      map.set(k, idx);
      rgb.push(r, g, b);
    }
    return idx;
  };
  // Frame palettes are added in full: a color present in a table but unused by
  // that frame's indices may still be needed for *other* frames' pixels.
  let overflow = 0;
  const remaps = new Array(frames.length);
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const pal = f.palette || globalPalette;
    if (!pal) return { ok: false, palette: null, colors: 0, remaps: [], used: null, overflow: 1 << 20 };
    const n = pal.length / 3;
    const table = new Uint8Array(256).fill(0);
    for (let c = 0; c < n; c++) {
      const idx = add(pal[c * 3], pal[c * 3 + 1], pal[c * 3 + 2]);
      if (idx < 0) overflow++;
      table[c] = idx < 0 ? 0 : idx;
    }
    remaps[i] = table;
  }
  if (opts.background && opts.background.length >= 3) add(opts.background[0], opts.background[1], opts.background[2]);
  if (map.size < 2 && rgb.length < 6) {
    // a one-color GIF is legal but many viewers dislike it; give them black+white
    add(0, 0, 0);
    add(255, 255, 255);
  }
  const palette = new Uint8Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) palette[i] = rgb[i];
  const used = new Uint8Array(palette.length / 3);
  return { ok: overflow === 0, palette, colors: palette.length / 3, remaps, used, overflow };
}

/**
 * Which global palette entries are actually referenced? (for shrinking the table)
 * @param {Uint8Array} indicesPerFrame combined index buffers
 */
export function markUsed(indices, used) {
  for (let i = 0; i < indices.length; i++) used[indices[i]] = 1;
  return used;
}

/**
 * Re-quantize one RGBA frame for the lossy path.
 * @returns {{palette:Uint8Array, colors:number, dither:string|null, error:number}}
 */
export function quantizeFrameColors(raster, opts = {}) {
  const colors = Math.max(2, Math.min(256, opts.colors || 128));
  const hist = buildHistogram(raster, { ...(opts.quantizeOptions || {}), colors, exhaustive: opts.exhaustive });
  const res = quantize([raster], {
    colors,
    method: opts.method || 'auto',
    histogram: hist,
    dither: opts.dither || 'none',
    ...(opts.quantizeOptions || {}),
  });
  return {
    palette: res.palette,
    colors: res.colors || res.palette.length / 3,
    dither: opts.dither && opts.dither !== 'none' ? opts.dither : null,
    error: res.error || 0,
    method: res.method,
    alphaThreshold: opts.alphaThreshold,
  };
}

/**
 * Compact a palette to the entries used inside `rect`, remapping `indices`
 * in place. Wraps `quant/mapper.compactPalette` with the rect restriction so
 * pixels outside it (which the encoder will not emit) cannot pin entries open.
 */
export function compactPaletteForRect(indices, palette, colors, rect, W, H) {
  const used = new Uint8Array(256);
  const y1 = Math.min(H, rect.y + rect.height);
  const x1 = Math.min(W, rect.x + rect.width);
  for (let y = rect.y; y < y1; y++) for (let x = rect.x; x < x1; x++) used[indices[y * W + x]] = 1;
  if (rect.width === W && rect.height === H) for (let i = 0; i < indices.length; i++) used[indices[i]] = 1;
  const res = compactPalette(indices, palette, colors, used);
  return res;
}

/** CRC-free helper: how many bytes does a GIF color table cost? */
export const paletteBytes = (colors) => {
  let bits = 1;
  while ((1 << bits) < Math.max(2, colors)) bits++;
  return (1 << Math.max(1, Math.min(8, bits))) * 3;
};

/**
 * GIFX Kernel — GIF re-encoder ("gifsicle -O3" equivalent, plus what gifsicle
 * cannot do because it does not own the decode).
 *
 * Two modes, because the guarantees differ:
 *
 *  - `lossless: true` (default) — output pixels must be bit-identical. Every
 *    technique is therefore *presentation* surgery: drop frames nobody can see and
 *    fold their delay into the neighbour, shrink each frame to its dirty rect,
 *    mark unchanged pixels transparent instead of re-emitting them, compact each
 *    frame's local palette to the entries it uses, retune the LZW clear-code
 *    interval, and quantize delays to centiseconds with drift correction.
 *  - `lossless: false` — additionally re-quantizes (`colors`, `method`, `dither`).
 *    This is where the big wins are: a 256-color screen recording at 64 colors is
 *    usually 30–50% smaller with no visible difference.
 *
 * The identity claim is enforced, not asserted: pass `check: true` (or call
 * `verifyOptimization`) and the result is decoded again and compared per pixel.
 *
 * @module optimize/remux
 */
import { GifWriter, normalizeDelay, Disposal } from '../enc/gif.js';
import { parseGif, composeGifFrames } from '../dec/gif.js';
import { dirtyRect, unionRect } from './diff.js';
import { exactLzwBytes, tuneClearInterval , paletteSlots} from './size.js';
import { createMapper, mapFrame, quantizeFrameColors, buildGlobalPalette, compactPaletteForRect } from './remux-helpers.js';

import { GifxError, ErrorCode } from '../core/errors.js';
import { Raster } from '../core/buffers.js';

export const OPTIMIZE_TECHNIQUES = Object.freeze(['dedupe', 'subFrames', 'transparentUnchanged', 'compactPalettes', 'clearTune', 'delayQuantize', 'requantize', 'stripComments']);

/**
 * @param {Uint8Array|object} input GIF bytes or a `parseGif()` result
 * @param {object} [opts]
 * @param {boolean} [opts.lossless=true]
 * @param {object} [opts.techniques] per-technique switches
 * @param {number} [opts.colors=128] lossy mode palette size
 * @param {string} [opts.method='auto'] lossy quantizer
 * @param {string} [opts.dither='none'] lossy dither
 * @param {number} [opts.maxDelta=0] pixel tolerance for "unchanged" (0 = exact)
 * @param {boolean|string} [opts.stripComments=false] true drops Comment+XMP,
 *   `'xmp'` drops only the XMP packet
 * @param {number|boolean|'auto'} [opts.clearInterval='auto']
 * @param {number} [opts.minDelayCs=2]
 * @param {number} [opts.originalBytes] source size, for the savings report
 * @param {boolean} [opts.check=false] decode the output and compare pixels
 * @param {function} [opts.onProgress]
 * @returns {object} `{bytes, savedBytes, savedRatio, framesIn, framesOut,
 *   techniques, warnings, durationMs, check}`
 */
export function optimizeGif(input, opts = {}) {
  const t0 = Date.now();
  const parsed = input && input.frames ? input : parseGif(input, {});
  if (!parsed || !parsed.frames || !parsed.frames.length) throw new GifxError('GIF has no frames to optimize', { code: ErrorCode.INPUT_NO_FRAMES });
  const src = parsed.frames;
  const W = parsed.width | 0;
  const H = parsed.height | 0;
  if (!(W > 0 && H > 0)) throw new GifxError(`GIF canvas is ${W}x${H}`, { code: ErrorCode.INPUT_CORRUPT });
  const noPixels = src.filter((f) => !f.indices);
  if (noPixels.length) {
    if (opts.strict) throw new GifxError(`${noPixels.length} frame(s) have no decoded pixels`, { code: ErrorCode.INPUT_CORRUPT, data: { frames: noPixels.slice(0, 8).map((f) => f.index) } });
  }
  const enabled = {};
  for (const name of OPTIMIZE_TECHNIQUES) {
    const explicit = opts.techniques && opts.techniques[name];
    if (explicit !== undefined) enabled[name] = !!explicit;
    else if (name === 'requantize') enabled[name] = opts.lossless === false;
    else if (name === 'stripComments') enabled[name] = !!opts.stripComments;
    else enabled[name] = true;
  }
  const maxDelta = Math.max(0, opts.maxDelta | 0);
  const minDelayCs = opts.minDelayCs == null ? 2 : opts.minDelayCs;
  const warnings = [];
  const compactSavings = [];
  const sourceTotalCs = src.reduce((a, f) => a + (f.delayCs || 0), 0);
  const delays = enabled.delayQuantize
    ? quantizeDelays(src.map((f) => f.delayCs || 10), minDelayCs, sourceTotalCs)
    : src.map((f) => Math.max(minDelayCs, Math.round(f.delayCs || 10)));
  const bg = paletteColor(parsed.palette, parsed.backgroundColorIndex || 0);

  // --- one shared index space, if the whole animation fits in 256 colors ----
  const gp = enabled.requantize ? { ok: false, overflow: -1, colors: 0, remaps: [] } : buildGlobalPalette(src, parsed.palette, { limit: 256, background: bg });
  const globalMode = !!gp.ok && gp.colors >= 2;
  if (!globalMode && !enabled.requantize) {
    warnings.push({
      code: ErrorCode.CONFIG_OUT_OF_RANGE,
      message: `the animation uses more than 256 distinct colors (${gp.overflow} did not fit); sub-framing and transparency are off and each frame keeps its local palette`,
    });
  }

  // Shrink the shared table to the colors any frame can actually reference, before
  // the writer commits the (global) color table.
  if (globalMode && enabled.compactPalettes) compactSavings.push(compactGlobalPalette(gp, src, W, H));

  // --- canvas state, exactly as a viewer keeps it --------------------------
  const px = W * H;
  const idxCur = new Uint8Array(px);
  const idxPrev = new Uint8Array(px);
  const rgbCur = new Uint8Array(px * 3);
  const rgbPrev = new Uint8Array(px * 3);
  // Two scratch buffers: the frame handed to the writer must not be overwritten
  // by the next iteration, which is why we alternate instead of reusing one.
  const encBufs = [new Uint8Array(px), new Uint8Array(px)];
  let encSel = 0;
  const startIdx = globalMode ? indexOfColor(gp.palette, bg[0], bg[1], bg[2]) : 0;
  idxPrev.fill(startIdx);
  idxCur.fill(startIdx);
  for (let p = 0; p < px; p++) {
    rgbPrev[p * 3] = bg[0];
    rgbPrev[p * 3 + 1] = bg[1];
    rgbPrev[p * 3 + 2] = bg[2];
  }

  const globalComment = enabled.stripComments ? null : parsed.comments && parsed.comments.length ? (parsed.comments.length === 1 ? parsed.comments[0] : parsed.comments.join('\n')) : null;
  const writer = new GifWriter({
    width: W,
    height: H,
    loop: parsed.loop === undefined ? null : parsed.loop,
    backgroundColorIndex: parsed.backgroundColorIndex || 0,
    pixelAspectRatio: parsed.pixelAspectRatio || 0,
    palette: globalMode ? gp.palette : null,
    colors: globalMode ? gp.colors : 0,
    // GifWriter takes one global comment; a source with several is folded into it
    // (their bytes are counted in `stripComments` savings either way).
    comment: globalComment,
    xmp: enabled.stripComments ? null : parsed.xmp || null,
  });

  const techniques = {
    dedupe: { enabled: enabled.dedupe, frames: 0, savedBytes: 0 },
    subFrames: { enabled: enabled.subFrames && globalMode, savedPixels: 0 },
    transparentUnchanged: { enabled: enabled.transparentUnchanged && globalMode, pixels: 0 },
    compactPalettes: { enabled: enabled.compactPalettes, savedBytes: compactSavings.reduce((a, b) => a + b, 0) },
    clearTune: { enabled: enabled.clearTune, savedBytes: 0, tested: 0 },
    delayQuantize: { enabled: enabled.delayQuantize, sourceTotalMs: sourceTotalCs * 10 },
    requantize: { enabled: enabled.requantize, colors: opts.colors || 128, method: opts.method || 'auto' },
    stripComments: { enabled: enabled.stripComments, savedBytes: enabled.stripComments ? commentBytes(parsed) : 0 },
    globalPalette: { used: globalMode, colors: gp.colors, overflow: gp.overflow },
  };

  let pending = null;
  let pendingDelay = 0;
  let emitted = 0;
  let totalFull = 0;
  let totalEncoded = 0;
  let outDelayCs = 0;
  const lossy = enabled.requantize ? new Raster(W, H) : null;

  // Frames are committed one behind: a frame that turns out to be invisible folds
  // its display time into the *previous* one, which must therefore still be
  // pending. `pendingIdx` is why that is affordable — the encoder reads indices
  // synchronously, so a single reused copy is enough.
  const flush = () => {
    if (!pending) return;
    const delay = normalizeDelay(pendingDelay, minDelayCs);
    outDelayCs += delay;
    writer.addFrame({ ...pending, delay });
    emitted++;
    pending = null;
    pendingDelay = 0;
  };

  // Disposal methods are part of the frame's meaning: a frame that paints nothing
  // but restores the background still changes what the *next* frame sits on. Mirror
  // it in the running composite so sub-framing and dedupe stay exact.
  const applyDisposal = (f) => {
    const d = f.disposal | 0;
    if (d !== 2 && d !== 3) return;
    const rect = frameRect(f, W, H);
    if (!rect.area) return;
    for (let y = rect.y; y < rect.y + rect.height; y++) {
      const row = y * W;
      for (let x = rect.x; x < rect.x + rect.width; x++) {
        const p = row + x;
        if (d === 2) {
          if (globalMode) idxCur[p] = startIdx;
          rgbCur[p * 3] = bg[0];
          rgbCur[p * 3 + 1] = bg[1];
          rgbCur[p * 3 + 2] = bg[2];
        } else {
          if (globalMode) idxCur[p] = idxPrev[p];
          rgbCur[p * 3] = rgbPrev[p * 3];
          rgbCur[p * 3 + 1] = rgbPrev[p * 3 + 1];
          rgbCur[p * 3 + 2] = rgbPrev[p * 3 + 2];
        }
      }
    }
  };

  for (let i = 0; i < src.length; i++) {
    const f = src[i];
    if (globalMode) paintIndexed(idxCur, idxPrev, rgbCur, rgbPrev, f, gp, i, parsed, W, H);
    else paintRgb(rgbCur, rgbPrev, f, parsed, W, H);

    let rect;
    if (!enabled.dedupe && !enabled.subFrames) rect = fullRect(W, H);
    else if (maxDelta > 0 || !globalMode) rect = dirtyRect(rgbPrev, rgbCur, { width: W, height: H, stride: W * 3, bytesPerPixel: 3, threshold: maxDelta });
    else rect = indexDiffRect(idxPrev, idxCur, W, H);
    // A frame whose disposal is 2/3 changes what the *next* frame sits on, so it can
    // never be dropped even when it paints nothing.
    if (!rect.changed && i > 0 && enabled.dedupe && (f.disposal | 0) <= 1) {
      pendingDelay += delays[i];
      techniques.dedupe.frames++;
      techniques.dedupe.savedBytes += Math.max(0, (f.compressedSize || 0) + 18);
      applyDisposal(f);
      idxPrev.set(idxCur);
      rgbPrev.set(rgbCur);
      continue;
    }
    // Sub-framing off (or an empty diff rect) => the frame covers the whole canvas.
    // Frame 0 needs no special case: in lossless mode the viewer and our model both
    // start from the file's background color, and the lossy path below already emits
    // full-canvas frames because the re-quantized palette replaces the shared one.
    if (!techniques.subFrames.enabled || !(rect.width > 0 && rect.height > 0)) rect = fullRect(W, H);

    // Disposal 2 (restore to background) and 3 (restore to previous) describe an
    // action on the frame's drawing area, so the payload has to cover that area in
    // full — otherwise the viewer clears/less than the source did and the base for
    // the following frame drifts. Re-using the source's method keeps the composite
    // exact without inventing extra frames.
    const srcDisposal = f.disposal | 0;
    const passThroughDisposal = globalMode && !enabled.requantize && srcDisposal >= 2;
    if (passThroughDisposal) {
      rect = unionRect(rect, frameRect(f, W, H), W, H);
      rect.area = rect.width * rect.height;
      rect.changed = rect.area > 0;
    }
    let outPalette = globalMode ? null : f.palette || parsed.palette;
    let outColors = globalMode ? gp.colors : outPalette ? Math.max(2, outPalette.length / 3) : 2;
    let encodeRect = globalMode ? rect : frameRect(f, W, H);
    let transparentIndex = -1;

    flush(); // commit the previous frame first: the encode buffer is about to be rewritten
    const pendingIdx = encBufs[encSel];
    if (globalMode) pendingIdx.set(idxCur);
    else copyFrameRect(pendingIdx, f, W, H); // local-palette mode re-encodes the frame verbatim
    if (enabled.requantize) {
      rgbToRgba3(rgbCur, lossy.data, W, H);
      lossy.opaque = true;
      const res = quantizeFrameColors(lossy, { colors: opts.colors || 128, method: opts.method, dither: opts.dither, quantizeOptions: opts.quantizeOptions });
      const mapper = createMapper(res.palette, res.colors, {});
      const mapped = mapFrame(lossy, mapper, { out: pendingIdx, dither: res.dither, temporal: opts.temporalDither ? 'offset' : 'none', frameIndex: i });
      outPalette = res.palette;
      outColors = res.colors;
      encodeRect = fullRect(W, H);
      if (mapped && mapped.transparentIndex >= 0) transparentIndex = mapped.transparentIndex;
    } else {
      // Palette compaction first: it renumbers indices, and the transparent slot
      // has to be chosen in the *final* table's index space.
      if (!globalMode && enabled.compactPalettes) {
        const compacted = compactPaletteForRect(pendingIdx, outPalette, outColors, encodeRect, W, H);
        if (compacted.colors < outColors) {
          techniques.compactPalettes.savedBytes += (outColors - compacted.colors) * 3;
          outPalette = compacted.palette;
          outColors = compacted.colors;
        }
      }
      if (techniques.transparentUnchanged.enabled && encodeRect.area > 0 && encodeRect.area < W * H) {
        let same = 0;
        for (let y = encodeRect.y; y < encodeRect.y + encodeRect.height; y++) {
          const row = y * W;
          for (let x = encodeRect.x; x < encodeRect.x + encodeRect.width; x++) if (idxPrev[row + x] === idxCur[row + x]) same++;
        }
        if (same > 0 && same < encodeRect.area) {
          transparentIndex = encodeTransparent(pendingIdx, idxPrev, encodeRect, outColors, W);
          if (transparentIndex >= 0) techniques.transparentUnchanged.pixels += same;
        }
      }
    }

    totalFull += W * H;
    totalEncoded += encodeRect.width * encodeRect.height;
    techniques.subFrames.savedPixels += W * H - encodeRect.width * encodeRect.height;
    if (!(encodeRect.width > 0 && encodeRect.height > 0)) throw new GifxError(`internal: empty encode rect at frame ${i}`, { code: ErrorCode.BUG_INTERNAL });

    let clearInterval = 0;
    if (typeof opts.clearInterval === 'number') clearInterval = Math.max(0, opts.clearInterval | 0);
    else if (enabled.clearTune && encodeRect.width * encodeRect.height > 12000) {
      const tune = tuneClearInterval(encBufs[encSel], px, { colors: outColors, width: W, quick: encodeRect.width * encodeRect.height > 90000 });
      clearInterval = tune.clearInterval;
      techniques.clearTune.tested += tune.tested;
    }

    encSel ^= 1;
    pending = {
      indices: pendingIdx,
      stride: W, // pendingIdx is the whole canvas; x0/y0 pick the rect to encode
      x0: encodeRect.x,
      y0: encodeRect.y,
      x: encodeRect.x,
      y: encodeRect.y,
      width: encodeRect.width,
      height: encodeRect.height,
      palette: outPalette,
      colors: Math.max(2, Math.min(256, outColors)),
      disposal: passThroughDisposal ? srcDisposal : Disposal.KEEP,
      interlace: globalMode ? false : !!f.interlace,
      transparentIndex,
      lzw: clearInterval > 0 ? { clearInterval } : undefined,
      comment: enabled.stripComments ? null : frameComment(f, globalComment),
    };
    pendingDelay += delays[i];
    applyDisposal(f);
    idxPrev.set(idxCur);
    rgbPrev.set(rgbCur);
    if (opts.onProgress && (i & 15) === 0) opts.onProgress(i + 1, src.length);
  }
  flush();
  const bytes = writer.finish();
  const originalBytes = opts.originalBytes || (input && input.length) || 0;
  const result = {
    bytes,
    originalBytes,
    savedBytes: Math.max(0, originalBytes - bytes.length),
    savedRatio: originalBytes ? 1 - bytes.length / originalBytes : 0,
    framesIn: src.length,
    framesOut: emitted,
    techniques,
    warnings,
    durationMs: outDelayCs * 10,
    sourceDurationMs: sourceTotalCs * 10,
    subFramePixelRatio: totalFull ? totalEncoded / totalFull : 1,
    globalPalette: globalMode,
    paletteColors: globalMode ? gp.colors : 0,
    elapsedMs: Date.now() - t0,
  };
  if (opts.check) result.check = verifyOptimization(parsed, bytes, { maxDelta, ignoreAlpha: opts.checkAlpha === false });
  if (opts.onProgress) opts.onProgress(src.length, src.length);
  return result;
}

/**
 * Mark unchanged pixels transparent inside the rect and return the slot used, or -1
 * when no palette slot is free (all 256 entries are live in this rect).
 */
function encodeTransparent(idxBuf, idxPrev, rect, colors, W) {
  const slot = pickTransparentSlot(idxBuf, rect, W, colors);
  if (slot < 0) return -1;
  markTransparent(idxBuf, idxPrev, rect, slot, W);
  return slot;
}

/**
 * Shrink a shared global palette to the colors any frame can actually produce.
 * Mutates `gp` (palette + every frame's remap table) and returns the savings.
 */
function compactGlobalPalette(gp, frames, W, H) {
  const used = new Uint8Array(256);
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (!f.indices) continue;
    const remap = gp.remaps[i];
    const palN = (f.palette || { length: 0 }).length / 3;
    if (remap) {
      for (let p = 0; p < f.indices.length; p++) {
        const v = f.indices[p];
        used[remap[v < palN ? v : palN - 1]] = 1;
      }
    } else {
      for (let p = 0; p < f.indices.length; p++) used[f.indices[p]] = 1;
    }
  }
  const map = new Uint8Array(256).fill(0);
  let n = 0;
  for (let i = 0; i < gp.colors; i++) if (used[i]) map[i] = n++;
  if (!n || n === gp.colors) return 0;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < gp.colors; i++) {
    if (!used[i]) continue;
    const t = map[i];
    out[t * 3] = gp.palette[i * 3];
    out[t * 3 + 1] = gp.palette[i * 3 + 1];
    out[t * 3 + 2] = gp.palette[i * 3 + 2];
  }
  const saved = (gp.colors - n) * 3;
  gp.palette = out;
  gp.colors = n;
  for (let i = 0; i < gp.remaps.length; i++) {
    const table = gp.remaps[i];
    for (let c = 0; c < table.length; c++) table[c] = map[table[c]] & 255;
  }
  void W;
  void H;
  return saved;
}

const fullRect = (w, h) => ({ x: 0, y: 0, width: w, height: h, area: w * h, changed: true });

/**
 * Paint frame `f` into the shared index space. Returns the number of pixels that
 * changed (0 ⇒ the frame is invisible).
 */
function paintIndexed(idxCur, idxPrev, rgbCur, rgbPrev, f, gp, frameIndex, parsed, W, H, maxDelta) {
  idxCur.set(idxPrev);
  rgbCur.set(rgbPrev);
  const pal = f.palette || parsed.palette;
  if (!f.indices || !pal) return 0;
  const remap = gp.remaps[frameIndex];
  const t = f.transparentIndex;
  // parseGif already normalizes interlaced frames to raster order, so reordering
  // rows again here would scramble them.
  const y0 = Math.max(0, f.y);
  const y1 = Math.min(H, f.y + f.height);
  const x0 = Math.max(0, f.x);
  const x1 = Math.min(W, f.x + f.width);
  let changed = 0;
  const palN = pal.length / 3;
  for (let y = y0; y < y1; y++) {
    const srcRow = y - f.y;
    const sBase = srcRow * f.width;
    for (let x = x0; x < x1; x++) {
      let v = f.indices[sBase + (x - f.x)];
      const p = y * W + x;
      if (v === t) continue;
      if (v >= palN) v = palN - 1; // same clamp the decoder applies
      const g = remap ? remap[v] : v;
      if (idxCur[p] !== g) changed++;
      idxCur[p] = g;
      const c3 = p * 3;
      rgbCur[c3] = pal[v * 3];
      rgbCur[c3 + 1] = pal[v * 3 + 1];
      rgbCur[c3 + 2] = pal[v * 3 + 2];
    }
  }
  void maxDelta;
  return changed;
}

/** Bounding box of differing indices — cheaper and exact where colors are shared. */
function indexDiffRect(a, b, W, H, maxDelta) {
  let minX = W;
  let minY = H;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      if (a[row + x] !== b[row + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return { x: 0, y: 0, width: 0, height: 0, area: 0, changed: false };
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1, area: (maxX - minX + 1) * (maxY - minY + 1), changed: true };
}

/**
 * Pixels identical to the previous composite carry no information, so they are
 * rewritten with the transparent slot: the viewer keeps whatever it already had.
 * Only touches `idxCur` (the encode buffer), never the composite model.
 */
/** A frame's own comments, minus the ones already written at file level. */
function frameComment(f, globalComment) {
  if (!f.comments || !f.comments.length) return null;
  const text = f.comments.length === 1 ? f.comments[0] : f.comments.join('\n');
  return text === globalComment ? null : text;
}

function markTransparent(idxCur, idxPrev, rect, t, W) {
  for (let y = rect.y; y < rect.y + rect.height; y++) {
    const row = y * W;
    for (let x = rect.x; x < rect.x + rect.width; x++) {
      const p = row + x;
      if (idxPrev[p] === idxCur[p]) idxCur[p] = t;
    }
  }
}

/** The frame's own drawing area, clipped into the canvas. */
function frameRect(f, W, H) {
  const x = Math.max(0, f.x | 0);
  const y = Math.max(0, f.y | 0);
  const x1 = Math.min(W, (f.x | 0) + (f.width | 0));
  const y1 = Math.min(H, (f.y | 0) + (f.height | 0));
  const width = Math.max(0, x1 - x);
  const height = Math.max(0, y1 - y);
  return { x, y, width, height, area: width * height, changed: width > 0 && height > 0 };
}

/** Copy a parsed frame's index rows into a canvas-strided buffer (verbatim re-encode). */
function copyFrameRect(dst, f, W, H) {
  if (!f.indices) return;
  const rect = frameRect(f, W, H);
  const rowSkip = f.x < 0 ? -f.x : 0;
  const colSkip = f.y < 0 ? -f.y : 0;
  for (let y = 0; y < rect.height; y++) {
    const sRow = (y + colSkip) * f.width;
    const dRow = (rect.y + y) * W;
    for (let x = 0; x < rect.width; x++) dst[dRow + rect.x + x] = f.indices[sRow + x + rowSkip] | 0;
  }
}

/** Exact color lookup in a palette (lossless global-palette mode). */
function indexOfColor(palette, r, g, b) {
  for (let i = 0; i < palette.length; i += 3) if (palette[i] === r && palette[i + 1] === g && palette[i + 2] === b) return i / 3;
  return 0;
}

/** Paint into RGB space only (local-palette fallback, and the lossy path). */
function paintRgb(cur, prev, f, parsed, W, H) {
  cur.set(prev);
  const pal = f.palette || parsed.palette;
  if (!f.indices || !pal) return 0;
  const t = f.transparentIndex;
  const palN = pal.length / 3;
  // parseGif already normalizes interlaced frames to raster order, so reordering
  // rows again here would scramble them.
  const y0 = Math.max(0, f.y);
  const y1 = Math.min(H, f.y + f.height);
  const x0 = Math.max(0, f.x);
  const x1 = Math.min(W, f.x + f.width);
  let changed = 0;
  for (let y = y0; y < y1; y++) {
    const srcRow = y - f.y;
    const sBase = srcRow * f.width;
    for (let x = x0; x < x1; x++) {
      const v = f.indices[sBase + (x - f.x)];
      if (v === t || v >= palN) continue;
      const p = (y * W + x) * 3;
      const vi = v * 3;
      if (cur[p] !== pal[vi] || cur[p + 1] !== pal[vi + 1] || cur[p + 2] !== pal[vi + 2]) changed++;
      cur[p] = pal[vi];
      cur[p + 1] = pal[vi + 1];
      cur[p + 2] = pal[vi + 2];
    }
  }
  return changed;
}

function withinDelta(cur, prev, c3, maxDelta) {
  if (!maxDelta) return cur[c3] === prev[c3] && cur[c3 + 1] === prev[c3 + 1] && cur[c3 + 2] === prev[c3 + 2];
  return Math.abs(cur[c3] - prev[c3]) <= maxDelta && Math.abs(cur[c3 + 1] - prev[c3 + 1]) <= maxDelta && Math.abs(cur[c3 + 2] - prev[c3 + 2]) <= maxDelta;
}

function nearestIndex(cur, palette, n, c3) {
  const r = cur[c3];
  const g = cur[c3 + 1];
  const b = cur[c3 + 2];
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const dr = r - palette[p];
    const dg = g - palette[p + 1];
    const db = b - palette[p + 2];
    const d = dr * dr * 2 + dg * dg * 3 + db * db * 2;
    if (d <= bestD) {
      if (!d) return i;
      bestD = d;
      best = i;
    }
  }
  return best;
}

/**
 * A palette index we can use for "keep the previous pixel", or -1 when there is none.
 *
 * The GIF LZW alphabet is fixed by the color-table size, so an index is only legal
 * below `1 << paletteBits(colors)`: a 16-entry table cannot encode index 16, and
 * doing that anyway silently corrupts the frame. Free padding slots come first; when
 * the table is exactly full we reuse an entry this rect does not reference, which is
 * safe because unchanged pixels are the only ones we rewrite.
 */
function pickTransparentSlot(idxBuf, rect, W, colors) {
  const capacity = paletteSlots(colors);
  if (colors < capacity) return colors;
  const used = new Uint8Array(capacity);
  for (let y = rect.y; y < rect.y + rect.height; y++) for (let x = rect.x; x < rect.x + rect.width; x++) used[idxBuf[y * W + x]] = 1;
  for (let i = 0; i < capacity; i++) if (!used[i]) return i;
  return -1;
}

function rgbToRgba3(srcRgb, dstRgba, W, H) {
  for (let p = 0; p < W * H; p++) {
    dstRgba[p * 4] = srcRgb[p * 3];
    dstRgba[p * 4 + 1] = srcRgb[p * 3 + 1];
    dstRgba[p * 4 + 2] = srcRgb[p * 3 + 2];
    dstRgba[p * 4 + 3] = 255;
  }
}

function paletteColor(palette, index) {
  if (!palette || index < 0 || index * 3 + 2 >= palette.length) return [0, 0, 0];
  return [palette[index * 3], palette[index * 3 + 1], palette[index * 3 + 2]];
}

/**
 * Paint frame `f` onto a copy of `prev`, honouring transparency, sub-frame
 * offsets, interlacing and out-of-bounds clamping (real files overflow often).
 */
export function paintInto(cur, prev, f, parsed, W, H) {
  cur.set(prev);
  const pal = f.palette || parsed.palette;
  if (!f.indices || !pal) return;
  const t = f.transparentIndex;
  const palN = pal.length / 3;
  const y0 = Math.max(0, f.y);
  const y1 = Math.min(H, f.y + f.height);
  const x0 = Math.max(0, f.x);
  const x1 = Math.min(W, f.x + f.width);
  for (let y = y0; y < y1; y++) {
    const srcRow = y - f.y;
    const sBase = srcRow * f.width;
    for (let x = x0; x < x1; x++) {
      const v = f.indices[sBase + (x - f.x)];
      if (v === t || v >= palN) continue;
      const pi = (y * W + x) * 3;
      const vi = v * 3;
      cur[pi] = pal[vi];
      cur[pi + 1] = pal[vi + 1];
      cur[pi + 2] = pal[vi + 2];
    }
  }
}

function interlaceOrder(h) {
  const out = new Uint16Array(h);
  let p = 0;
  const starts = [0, 4, 2, 1];
  const steps = [8, 8, 4, 2];
  for (let pass = 0; pass < 4; pass++) for (let y = starts[pass]; y < h; y += steps[pass]) out[p++] = y;
  return out;
}

/**
 * Centisecond quantization that preserves total run time.
 *
 * Quantizing each delay independently drifts by up to 0.5 cs per frame — over a
 * 900-frame GIF that is 4.5 s of error per loop, enough to desync a GIF from a
 * sound or a CSS animation. So the *cumulative* time is quantized and the residue
 * is spread over the longest frames, where ±1 cs is invisible.
 */
export function quantizeDelays(delaysCs, minCs = 2, totalCs) {
  const out = new Array(delaysCs.length);
  let acc = 0;
  let sum = 0;
  for (let i = 0; i < delaysCs.length; i++) {
    acc += Math.max(0, Number(delaysCs[i]) || 0);
    const target = Math.round(acc);
    const v = Math.max(minCs, target - sum);
    out[i] = v;
    sum += v;
  }
  if (totalCs) {
    let drift = Math.round(totalCs) - sum;
    if (drift !== 0) {
      const sign = drift > 0 ? 1 : -1;
      const order = out.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      let left = Math.abs(drift);
      let k = 0;
      while (left > 0 && k < order.length * 6) {
        const idx = order[k % order.length][1];
        if (sign > 0 || out[idx] > minCs) {
          out[idx] += sign;
          left--;
        }
        k++;
      }
      drift = Math.round(totalCs) - out.reduce((a, b) => a + b, 0);
      if (drift !== 0) out[Math.max(0, out.length - 1)] = Math.max(minCs, out[out.length - 1] + drift);
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = normalizeDelay(out[i], minCs);
  return out;
}

function commentBytes(parsed) {
  let n = 0;
  const subBlocks = (len) => 4 + len + 1; // introducer + label + count byte(s) approximated by one block per 255
  for (const text of parsed.allComments || []) n += subBlocks(text.length + Math.ceil(text.length / 255));
  if (parsed.xmp) n += 4 + new TextEncoder().encode(parsed.xmp).length + Math.ceil(parsed.xmp.length / 255) + 20;
  return n;
}

/**
 * Decode both GIFs and compare every pixel. `worstDelta` 0 means the
 * optimization was bit-exact in what the viewer sees.
 */
export function verifyOptimization(parsed, bytes, opts = {}) {
  const a = composeGifFrames(parsed, { background: opts.background });
  const b = composeGifFrames(parseGif(bytes), { background: opts.background });
  if (!a.length || !b.length) return { ok: false, reason: 'nothing to compare', framesA: a.length, framesB: b.length, worstDelta: 255, firstBadFrame: 0 };
  const maxDelta = opts.maxDelta | 0;
  const ignoreAlpha = !!opts.ignoreAlpha;
  // Mid-frame samples: comparing frame *numbers* would flag a correct file whose
  // static frames were merged, so each source frame is matched to whichever output
  // frame is on screen at that instant.
  const ends = b.map((r) => r.pts * 1000 + r.durationMs);
  let worst = 0;
  let firstBad = -1;
  let badPixels = 0;
  let wi = 0;
  let bIndex = 0;
  for (let i = 0; i < a.length; i++) {
    const t = a[i].pts * 1000 + a[i].durationMs / 2;
    while (bIndex + 1 < b.length && t >= ends[bIndex]) bIndex++;
    const pa = a[i].data;
    const pb = b[bIndex].data;
    if (pa.length !== pb.length) return { ok: false, reason: 'frame size mismatch', framesA: a.length, framesB: b.length, worstDelta: 255, firstBadFrame: i };
    for (let p = 0; p < pa.length; p += 4) {
      const d = Math.max(Math.abs(pa[p] - pb[p]), Math.abs(pa[p + 1] - pb[p + 1]), Math.abs(pa[p + 2] - pb[p + 2]), ignoreAlpha ? 0 : Math.abs(pa[p + 3] - pb[p + 3]));
      if (d > worst) {
        worst = d;
        wi = i;
      }
      if (d > maxDelta) badPixels++;
    }
    if (worst > maxDelta && firstBad < 0) firstBad = wi;
  }
  if (firstBad < 0 && worst > maxDelta) firstBad = wi;
  const durationMsA = a[a.length - 1].pts * 1000 + a[a.length - 1].durationMs;
  const durationMsB = b[b.length - 1].pts * 1000 + b[b.length - 1].durationMs;
  const durationTol = opts.durationToleranceMs == null ? 20 : opts.durationToleranceMs;
  return {
    ok: firstBad < 0 && Math.abs(durationMsA - durationMsB) <= durationTol,
    reason: firstBad >= 0 ? `pixels differ from frame ${firstBad}` : Math.abs(durationMsA - durationMsB) > durationTol ? 'total duration moved' : 'identical',
    samples: a.length,
    framesA: a.length,
    framesB: b.length,
    badPixels,
    worstDelta: worst,
    firstBadFrame: firstBad,
    durationMsA,
    durationMsB,
    durationDeltaMs: durationMsB - durationMsA,
  };
}

/**
 * Convenience wrapper used by the public API: bytes/Blob/File in, optimized GIF
 * out, with the source size known for the savings report.
 */
export async function optimizeGifFile(input, opts = {}) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(await (input.arrayBuffer ? input.arrayBuffer() : input));
  const parsed = parseGif(bytes, {});
  return optimizeGif(parsed, { ...opts, originalBytes: bytes.length });
}

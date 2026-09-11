/**
 * GIFX kernel — public surface.
 *
 * A single import for a browser app that does video → GIF (and GIF/APNG/WebP
 * work) entirely on the client, with no server, no ffmpeg and no runtime
 * dependencies:
 *
 *   import { probe, optimizeGif, GifWriter, quantize } from '@gifx/kernel';
 *
 * Layers, in pipeline order:
 *
 *   core/     errors with hints, arenas + rasters, color math, events/progress
 *   media/    container demuxers (MP4/WebM/AVI/WebP), frame sources, WebCodecs
 *   image/    resize/crop/orient, filtergraph, dither, temporal dither, overlay
 *   quant/    six quantizers + the palette mapper
 *   enc/      LZW and the GIF writer        dec/ GIF reader, compositor, analysis
 *   optimize/ dirty-rect/dedupe/remux optimizer + the size model
 *
 * The higher-level orchestration the full library advertises — `convert()`,
 * `createGifEngine()`, the worker pool, the APNG/WebP/MediaRecorder encoders,
 * the DOM editor UI and the bundled `dist/` build — is not part of this surface
 * yet; everything re-exported below is implemented and covered by `test/`.
 */

export const VERSION = '1.0.0';

// --- core -----------------------------------------------------------------
export * from './core/errors.js';
export * from './core/buffers.js';
export * from './core/color.js';
export * from './core/events.js';
export * from './core/env.js';

// --- demux + probe --------------------------------------------------------
export * from './media/probe.js';
export * from './media/mp4.js';
export * from './media/webm.js';
export * from './media/riff.js';
export * from './media/webcodecs.js';
export * from './media/frames.js';

// --- pixel work -----------------------------------------------------------
export * from './image/ops.js';
export * from './image/filters.js';
export * from './image/dither.js';
export * from './image/temporal.js';
export * from './image/overlay.js';
export * from './image/layout.js';

// --- quantization ---------------------------------------------------------
export * from './quant/palette.js';
export * from './quant/mapper.js';

// --- encoders / decoders --------------------------------------------------
export * from './enc/lzw.js';
export * from './enc/gif.js';
export * from './dec/gif.js';

// --- optimization ---------------------------------------------------------
export * from './optimize/diff.js';
export * from './optimize/size.js';
export * from './optimize/remux.js';
export * from './optimize/remux-helpers.js';

// Names that two modules legitimately expose. Stated explicitly so `export *`
// ambiguity can never drop them from the barrel: the choice here is the public
// meaning, the other module keeps its own copy for internal use.
export { Raster, Arena, Scratch, getArena, setArena, nextPow2 } from './core/buffers.js';
export { clamp01, clamp255 } from './core/color.js';
export { createMapper, mapFrame, compactPalette, buildRemap } from './quant/mapper.js';
/** Container sniffing: `probe(bytes | File | Blob | Response)`. */
export { probe } from './media/probe.js';
/** Runtime capability probe (WebCodecs, workers, `createImageBitmap` options…). */
export { probe as probeCapabilities, capabilities, requireFeature, suggestWorkers, testBitmapOptions } from './core/env.js';

/**
 * The names a consumer can rely on. `tools/check.mjs` asserts each of these is
 * actually present on the barrel, which is what keeps an ambiguous `export *`
 * from silently deleting a public export.
 */
export const PUBLIC_SURFACE = [
  'VERSION',
  'GifxError',
  'ErrorCode',
  'wrapError',
  'Raster',
  'getArena',
  'Emitter',
  'ProgressReporter',
  'identify',
  'probe',
  'parseMp4',
  'parseWebm',
  'parseRiff',
  'decodeTrackFrames',
  'createDecodeDriver',
  'createFrameSource',
  'planFrames',
  'scale',
  'crop',
  'detectBlackBars',
  'applyFilters',
  'parseFiltergraph',
  'buildHistogram',
  'quantize',
  'autoMethod',
  'installBuiltins',
  'createMapper',
  'mapFrame',
  'lzwEncode',
  'lzwDecode',
  'GifWriter',
  'measureGif',
  'parseGif',
  'composeGifFrames',
  'analyzeGif',
  'dirtyRect',
  'collapseDuplicates',
  'planDisposal',
  'exactLzwBytes',
  'predictCandidate',
  'payloadBudget',
  'maxFramesForBudget',
  'bytesToHuman',
  'optimizeGif',
  'optimizeGifFile',
  'verifyOptimization',
  'quantizeDelays',
  'PUBLIC_SURFACE',
];

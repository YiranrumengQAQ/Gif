/**
 * GIFX Kernel — frame sources.
 *
 * One interface, five backends. Everything else in the library (filters,
 * quantizer, encoder, optimizer, target-size search) only ever sees:
 *
 *   const src = await createFrameSource(input, opts);
 *   src.meta                       // { width, height, fps, durationMs, … }
 *   for await (const f of src.frames(plan)) use(f.raster)   // Raster
 *   await src.close();
 *
 * Backends, in preference order:
 *  1. `rasters`      — caller already has decoded frames (worker re-encode,
 *                      second pass of a size search, canvas/ImageData input).
 *  2. `gif`          — our own decoder: keeps per-frame delays *and* indices, so
 *                      GIF→GIF re-encodes never re-time or re-dither the source.
 *  3. `webcodecs`     — demux + VideoDecoder (accurate, worker-friendly).
 *  4. `videoElement` — `<video>` + canvas; the universal fallback, main thread
 *                      only. Two capture strategies (see `policy` below).
 *  5. `stills`       — one or many images, optionally timed into a slideshow.
 *
 * The `<video>` fallback needs a policy because its two capture modes have very
 * different costs:
 *  - *play-through* (`requestVideoFrameCallback`): ~2–4 ms/frame, perfect for
 *    "whole clip at 12 fps", but it can only visit frames in order and its
 *    timestamps come from the media clock, so a mid-clip trim still needs a
 *    seek to get in.
 *  - *seek-per-frame*: accurate and random-access, but 30–90 ms/frame, so using
 *    it for every frame of a 30 s clip is 10× slower than needed.
 * `chooseCapturePolicy()` decides from the plan; it is pure and unit-tested.
 *
 * @module media/frames
 */
import { GifxError, ErrorCode, wrapError } from '../core/errors.js';
import { Raster, getArena } from '../core/buffers.js';
import { identify, probe, readBytes } from './probe.js';
import { parseMp4, mp4FrameTimes, parseMp4 as _unusedMp4 } from './mp4.js';
import { parseWebm, webmDecoderConfig } from './webm.js';
import { parseRiff } from './riff.js';
import { decodeTrackFrames, webcodecsVideoSupport, annexBTrack, frameToRaster } from './webcodecs.js';
import { parseGif, composeGifFrames, gifDuration } from '../dec/gif.js';
import { planFrameRate } from '../image/temporal.js';

void _unusedMp4;

/**
 * @typedef {Object} FrameSource
 * @property {string} kind
 * @property {object} meta
 * @property {function(object=):AsyncGenerator<{raster:Raster,ptsMs:number,index:number,durationMs?:number}>} frames
 * @property {function():Promise<void>} close
 * @property {function(number,number=):Promise<{raster:Raster}|null>} [seek]
 */

/**
 * Build the right source for `input`.
 *
 * @param {File|Blob|ArrayBuffer|Uint8Array|string|HTMLVideoElement|HTMLImageElement|ImageBitmap|ImageData|Raster|Raster[]|Array<{raster:Raster,ptsMs:number}>} input
 * @param {object} [opts] engine options (subset used here: `fps`, `start`, `end`,
 *   `duration`, `maxFrames`, `decode`, `transparent`, `exhaustive`, `probe`)
 */
export async function createFrameSource(input, opts = {}) {
  const ctx = { opts, warnings: [] };
  // --- already-decoded rasters -------------------------------------------------
  if (Array.isArray(input)) return rasterSourceFromList(input, opts);
  if (input instanceof Raster) return singleRasterSource(input, opts);
  if (typeof ImageData !== 'undefined' && input instanceof ImageData) return singleRasterSource(rasterFromImageData(input), opts);
  if (typeof HTMLCanvasElement !== 'undefined' && input instanceof HTMLCanvasElement) return singleRasterSource(rasterFromCanvas(input), opts);
  if (typeof OffscreenCanvas !== 'undefined' && input instanceof OffscreenCanvas) return singleRasterSource(await rasterFromOffscreen(input), opts);
  if (typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap) return singleBitmapSource(input, opts);
  if (typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement) return videoElementSource(input, opts, ctx);
  if (typeof HTMLImageElement !== 'undefined' && input instanceof HTMLImageElement) return imageElementSource(input, opts, ctx);

  // --- URL -------------------------------------------------------------------
  let bytes = null;
  let name = '';
  let mime = '';
  let blobForElement = null;
  if (typeof input === 'string') {
    if (/^data:/i.test(input) || input.startsWith('blob:') || /^https?:/i.test(input)) {
      const res = await fetchUrl(input, opts);
      blobForElement = res.blob;
      bytes = res.bytes;
      name = input;
      mime = res.type || '';
    } else {
      // a bare filename in a non-browser context is a common mistake — say so
      throw new GifxError(`cannot open "${input}" — pass a File/Blob/ArrayBuffer, or a blob:/http(s): URL`, { code: ErrorCode.INPUT_INVALID, data: { input: String(input).slice(0, 200) } });
    }
  } else if (typeof Blob !== 'undefined' && input instanceof Blob) {
    blobForElement = input;
    name = input.name || '';
    mime = input.type || '';
    const max = opts.headBytes || 0;
    bytes = await readBytes(input, max ? { maxBytes: max, range: 'head' } : {});
  } else {
    bytes = await readBytes(input, {});
  }
  if (!bytes || !bytes.length) throw new GifxError('input is empty (0 bytes)', { code: ErrorCode.INPUT_EMPTY });

  let info;
  try {
    info = identify(bytes, name, mime);
  } catch (e) {
    // last resort: some servers hand back `.mp4`-named blobs with a wrong header
    // (e.g. a 404 page). Report the real problem, including the head bytes.
    throw e;
  }
  switch (info.format) {
    case 'gif':
      return gifSource(bytes, opts, ctx);
    case 'mp4':
    case 'webm':
    case 'mkv':
    case 'avi':
    case 'h264-annexb':
      return videoSource(bytes, info, { ...opts, name, mime, blobForElement, ctx });
    default:
      return stillSource(bytes, info, { ...opts, name, blobForElement, ctx });
  }
}

async function fetchUrl(url, opts) {
  let res;
  try {
    res = await fetch(url, opts.fetchInit);
  } catch (e) {
    throw wrapError(e, { code: ErrorCode.NET_FETCH_FAILED, message: `fetch("${url}") failed`, data: { url } });
  }
  if (!res.ok) throw new GifxError(`${res.status} ${res.statusText} fetching ${url}`, { code: ErrorCode.NET_HTTP_STATUS, retryable: res.status >= 500 || res.status === 429, data: { status: res.status, url } });
  const blob = await res.blob().catch(() => null);
  const buf = await res.arrayBuffer().catch(async () => (blob ? (await blob.arrayBuffer()) : null));
  if (!buf) throw new GifxError(`could not read the response body for ${url}`, { code: ErrorCode.NET_FETCH_FAILED });
  return { bytes: new Uint8Array(buf), type: (blob && blob.type) || res.headers.get('content-type') || '', blob };
}

/* ---------------------------------------------------------------- rasters */

function rasterSourceFromList(list, opts) {
  const frames = [];
  for (let i = 0; i < list.length; i++) {
    const it = list[i];
    if (it instanceof Raster) frames.push({ raster: it, ptsMs: (opts.startMs ?? 0) + i * (opts.frameMs || 100), index: i, owned: false });
    else if (it && it.raster instanceof Raster) frames.push({ raster: it.raster, ptsMs: it.ptsMs ?? it.time ?? i * (opts.frameMs || 100), index: it.index ?? i, durationMs: it.durationMs, owned: false });
    else if (it && it.data && it.width) frames.push({ raster: rasterFromPlainObject(it), ptsMs: (it.ptsMs ?? i * (opts.frameMs || 100)), index: i, owned: false });
    else throw new GifxError(`frames[${i}] is not a Raster-like object`, { code: ErrorCode.INPUT_INVALID, data: { type: typeof it } });
  }
  const meta = sourceMetaFromFrames(frames, opts);
  return {
    kind: 'rasters',
    meta,
    async *frames(plan) {
      let emitted = 0;
      for (let i = 0; i < frames.length; i++) {
        if (plan && plan.skip && plan.skip(i)) continue;
        if (plan && plan.maxFrames && emitted >= plan.maxFrames) return;
        emitted++;
        yield { ...frames[i], index: i };
      }
    },
    async close() {
      for (const f of frames) if (f.owned) f.raster.release?.();
    },
  };
}

function singleRasterSource(raster, opts) {
  const durationMs = opts.duration != null ? opts.duration : 1000;
  return {
    kind: 'rasters',
    meta: { width: raster.width, height: raster.height, durationMs, fps: 1000 / durationMs, frameCount: 1, hasAlpha: raster.hasAlpha || false, source: 'raster' },
    async *frames() {
      yield { raster, ptsMs: 0, index: 0, durationMs };
    },
    async close() {
      /* caller owns it */
    },
  };
}

function singleBitmapSource(bitmap, opts) {
  const durationMs = opts.duration != null ? opts.duration : 1000;
  return {
    kind: 'rasters',
    meta: { width: bitmap.width, height: bitmap.height, durationMs, frameCount: 1, source: 'imagebitmap' },
    async *frames() {
      const r = new Raster(bitmap.width, bitmap.height);
      await bitmapToRaster(bitmap, r);
      yield { raster: r, ptsMs: 0, index: 0, durationMs, owned: true };
      bitmap.close?.();
    },
    async close() {},
  };
}

async function bitmapToRaster(bitmap, out) {
  const c = createCanvasLike(bitmap.width, bitmap.height);
  if (!c) {
    // No canvas: we can still take the ImageBitmap's pixels via createImageBitmap
    // → copyTo when available (VideoFrame-like), else fail clearly.
    if (typeof bitmap.copyTo === 'function') {
      await bitmap.copyTo(out.data, { layout: [{ offset: 0, stride: out.width * 4 }] });
      return out;
    }
    throw new GifxError('cannot read pixels from an ImageBitmap without a canvas', { code: ErrorCode.NO_OFFSCREEN_CANVAS });
  }
  c.width = bitmap.width;
  c.height = bitmap.height;
  const g = c.getContext('2d', { alpha: true, willReadFrequently: true });
  g.drawImage(bitmap, 0, 0);
  const d = g.getImageData(0, 0, bitmap.width, bitmap.height);
  out.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, out.data.length));
  return out;
}

function createCanvasLike(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  if (typeof document !== 'undefined' && document.createElement) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }
  return null;
}

function rasterFromImageData(img) {
  const r = new Raster(img.width, img.height);
  r.data.set(new Uint8Array(img.data.buffer, img.data.byteOffset, r.data.length));
  r.hasAlpha = true;
  return r;
}
function rasterFromPlainObject(o) {
  const r = new Raster(o.width, o.height);
  r.data.set(o.data.subarray(0, r.data.length));
  return r;
}
function rasterFromCanvas(c) {
  const r = new Raster(c.width, c.height);
  const g = c.getContext('2d', { willReadFrequently: true });
  const d = g.getImageData(0, 0, c.width, c.height);
  r.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, r.data.length));
  return r;
}
async function rasterFromOffscreen(c) {
  try {
    const bmp = c.transferToImageBitmap ? c.transferToImageBitmap() : await createImageBitmap(c);
    const r = new Raster(bmp.width, bmp.height);
    await bitmapToRaster(bmp, r);
    bmp.close?.();
    return r;
  } catch (e) {
    throw wrapError(e, { code: ErrorCode.NO_OFFSCREEN_CANVAS, message: 'could not read pixels from the OffscreenCanvas' });
  }
}

function sourceMetaFromFrames(frames, opts) {
  const first = frames[0];
  const last = frames[frames.length - 1] || first;
  const forced = opts.duration != null ? opts.duration : null;
  const durationMs = frames.length > 1 ? Math.max(1, forced || (last.ptsMs ?? 0) - (first.ptsMs ?? 0)) : forced || first.durationMs || 1000;
  return {
    width: first.raster.width,
    height: first.raster.height,
    frameCount: frames.length,
    durationMs,
    fps: (frames.length * 1000) / Math.max(1, durationMs),
    hasAlpha: frames.some((f) => f.raster.hasAlpha),
    source: 'rasters',
    opaque: opts.opaque !== false,
  };
}

/* -------------------------------------------------------------------- GIF */

function gifSource(bytes, opts, ctx) {
  let parsed = null;
  try {
    parsed = parseGif(bytes);
  } catch (e) {
    throw wrapError(e, { code: ErrorCode.INPUT_CORRUPT, message: 'GIF could not be parsed' });
  }
  if (!parsed.frames.length) throw new GifxError('GIF has no frames', { code: ErrorCode.INPUT_NO_FRAMES });
  if (opts.maxFrames && parsed.frames.length > opts.maxFrames && opts.enforceMaxFrames) {
    ctx.warnings.push({ code: ErrorCode.GIF_TOO_MANY_FRAMES, message: `${parsed.frames.length} frames exceeds limits.maxFrames; truncating`, count: parsed.frames.length });
  }
  const durationMs = (gifDuration(parsed) || {}).ms || parsed.frames.reduce((a, f) => a + (f.delayMs || 0), 0);
  const meta = {
    width: parsed.width,
    height: parsed.height,
    frameCount: parsed.frames.length,
    durationMs,
    fps: durationMs ? (parsed.frames.length * 1000) / durationMs : 0,
    hasAlpha: parsed.frames.some((f) => f.transparentIndex >= 0),
    loops: parsed.loops,
    gif: {
      version: parsed.version,
      hasGlobalPalette: parsed.hasGlobalPalette,
      colorResolution: parsed.colorResolution,
      backgroundColorIndex: parsed.backgroundColorIndex,
      pixelAspectRatio: parsed.pixelAspectRatio,
      comment: parsed.comment,
      xmp: parsed.xmp,
      palette: parsed.palette,
      frames: parsed.frames.map((f) => ({ delayMs: f.delayMs, delayCs: f.delayCs, disposal: f.disposal, transparentIndex: f.transparentIndex, x: f.x, y: f.y, width: f.width, height: f.height, interlace: f.interlace, localPalette: !!f.palette })),
    },
    source: 'gif',
    // The GIF's own delays are authoritative: a GIF→GIF re-encode must not
    // silently retime them, so the engine only overrides when fps is explicit.
    preserveTiming: opts.fps == null,
  };
  return {
    kind: 'gif',
    meta,
    parsed,
    bytes,
    async *frames(plan = {}) {
      const comps = composeGifFrames(parsed, {
        maxFrames: plan.maxFrames || opts.maxFrames || parsed.frames.length,
        only: plan.indices ? [...plan.indices] : undefined,
      });
      for (const c of comps) {
        const r = new Raster(parsed.width, parsed.height);
        r.data.set(c.data.subarray(0, r.data.length));
        r.pts = c.pts * 1000;
        r.duration = c.durationMs;
        r.index = c.index;
        r.hasAlpha = c.frame.transparentIndex >= 0 || parsed.frames[c.index]?.alphaInPalette === true;
        yield { raster: r, ptsMs: c.pts * 1000, index: c.index, durationMs: c.durationMs, owned: true, gifFrame: c.frame, delayCs: c.delayCs };
      }
    },
    /** Raw palette indices + palettes: lets a re-encode skip quantization entirely. */
    async *indices() {
      for (let i = 0; i < parsed.frames.length; i++) {
        const f = parsed.frames[i];
        yield { indices: f.indices, palette: f.palette || parsed.palette, delayMs: f.delayMs, delayCs: f.delayCs, disposal: f.disposal, transparentIndex: f.transparentIndex, x: f.x, y: f.y, width: f.width, height: f.height, interlace: f.interlace, index: i };
      }
    },
    async close() {
      parsed = null;
    },
  };
}

/* ------------------------------------------------------------------ stills */

async function stillSource(bytes, info, opts) {
  const decodable = await decodeStill(bytes, info, opts);
  const rasters = decodable.rasters;
  const perFrame = opts.duration != null ? opts.duration / rasters.length : 1000;
  const meta = {
    width: rasters[0].width,
    height: rasters[0].height,
    frameCount: rasters.length,
    durationMs: perFrame * rasters.length,
    fps: 1000 / perFrame,
    hasAlpha: decodable.hasAlpha,
    animated: rasters.length > 1,
    source: info.format,
  };
  return {
    kind: 'stills',
    meta,
    async *frames() {
      for (let i = 0; i < rasters.length; i++) yield { raster: rasters[i], ptsMs: i * perFrame, index: i, durationMs: perFrame, owned: i > 0 || !decodable.sharedFirst };
    },
    async close() {
      for (let i = decodable.sharedFirst ? 1 : 0; i < rasters.length; i++) rasters[i].release?.();
    },
  };
}

/**
 * Decode a still image (or animated image) to rasters.
 * Prefers `ImageDecoder` (handles animated PNG/WebP/AVIF/GIF natively, runs in
 * workers), falls back to `createImageBitmap` + canvas.
 */
export async function decodeStill(bytes, info, opts = {}) {
  const blob = new Blob([bytes.slice()], { type: info.mime || `image/${info.format}` });
  if (typeof ImageDecoder === 'function') {
    try {
      const dec = new ImageDecoder({ data: blob, type: info.mime || 'image/png' });
      await dec.completed;
      const count = dec.track ? Number(dec.track.frameCount) || 1 : 1;
      const rasters = [];
      let hasAlpha = false;
      for (let i = 0; i < count; i++) {
        const { image } = await dec.decode({ frameIndex: i });
        const r = new Raster(image.displayWidth, image.displayHeight);
        await bitmapToRaster(image, r);
        image.close?.();
        hasAlpha = hasAlpha || r.detectAlpha(7) === true;
        rasters.push(r);
        if (opts.maxFrames && rasters.length >= opts.maxFrames) break;
      }
      dec.close?.();
      if (rasters.length) return { rasters, hasAlpha, sharedFirst: false };
    } catch (e) {
      if (opts.strict) throw wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: `ImageDecoder failed on ${info.format}` });
      opts.ctx?.warnings?.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, message: `ImageDecoder: ${e.message}; using createImageBitmap` });
    }
  }
  if (typeof createImageBitmap === 'function') {
    try {
      const bmp = await createImageBitmap(blob);
      const r = new Raster(bmp.width, bmp.height);
      await bitmapToRaster(bmp, r);
      bmp.close?.();
      return { rasters: [r], hasAlpha: false, sharedFirst: false };
    } catch (e) {
      throw wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: `could not decode ${info.format} image`, data: { format: info.format } });
    }
  }
  throw new GifxError(`no still-image decoder available for "${info.format}"`, { code: ErrorCode.ENV_UNSUPPORTED, data: { needs: 'ImageDecoder or createImageBitmap' } });
}

/* ------------------------------------------------------------------ video */

async function videoSource(bytes, info, opts) {
  const ctx = opts.ctx || { warnings: [] };
  const plan = await buildVideoPlan(bytes, info, opts);
  const useWebCodecs = opts.prefer !== 'element' && plan.track && plan.samples?.length && (await webcodecsVideoSupport()).ok !== false;
  if (useWebCodecs) {
    try {
      return await webcodecsSource(bytes, plan, opts, ctx);
    } catch (e) {
      if (opts.strict) throw e;
      ctx.warnings.push({ code: e.code || ErrorCode.DECODE_CORRUPT_FRAME, message: `WebCodecs path failed (${e.message}); falling back to <video>` });
    }
  }
  if (opts.blob || opts.blobForElement || typeof document !== 'undefined') {
    return await elementSourceForBytes(bytes, info, opts, ctx);
  }
  throw new GifxError(`this environment can decode neither ${info.format} (no WebCodecs) nor fall back to <video> (no DOM)`, {
    code: ErrorCode.NO_VIDEO_DECODER,
    data: { format: info.format, hint: 'pass pre-decoded frames (Raster[]) or run in a browser' },
    hint: 'In Node/worker-without-DOM, supply `frames` (Raster[]) or use a browser build. The demuxer already extracted ' + (plan.samples?.length || 0) + ' samples.',
  });
}

/**
 * Demux + pick the video track + build a sample table.
 * Exposed for `probe()`-style reuse and for tests.
 */
export async function buildVideoPlan(bytes, info, opts = {}) {
  const plan = { format: info.format, tracks: [], warnings: [], samples: null, track: null, timescale: 1000, frameTimes: null, durationMs: 0, width: 0, height: 0, fps: 0, rotation: 0, hasAlpha: false, codec: '', seekable: false, fragmented: false };
  if (info.format === 'h264-annexb') {
    const t = annexBTrack(bytes, opts);
    plan.track = t.track;
    plan.timescale = t.track.timescale;
    plan.frameTimes = mp4FrameTimes({ samples: t.track.samples, timescale: t.track.timescale });
    plan.width = t.track.width;
    plan.height = t.track.height;
    plan.samples = t.track.samples;
    plan.codec = t.track.codecString;
    plan.durationMs = plan.frameTimes.length ? plan.frameTimes[plan.frameTimes.length - 1].pts : 0;
    plan.fps = plan.durationMs ? (plan.frameTimes.length * 1000) / plan.durationMs : 0;
    return plan;
  }
  if (info.format === 'webm' || info.format === 'mkv') {
    const w = parseWebm(bytes);
    if (!w.video) throw new GifxError('WebM has no video track', { code: ErrorCode.INPUT_NO_VIDEO_TRACK, data: { tracks: w.tracks.map((t) => ({ type: t.trackType, codec: t.codecId })) } });
    const v = w.video;
    plan.tracks = w.tracks;
    plan.width = v.width;
    plan.height = v.height;
    plan.rotation = 0;
    plan.durationMs = w.durationMs;
    plan.hasAlpha = !!w.hasAlpha;
    plan.codec = v.codecId;
    plan.seekable = !!w.seekable;
    plan.decoderConfig = webmDecoderConfig(v);
    plan.lacing = !!w.lacing;
    const blocks = w.blocks || [];
    if (blocks.length) {
      plan.frameTimes = blocks.map((b, i) => ({ index: i, pts: b.pts * (w.timecodeScale / 1e6), key: !!b.key, size: b.size, offset: b.dataStart, data: null, track: b }));
      plan.samples = blocks.map((b, i) => ({ offset: b.dataStart, size: b.size, pts: b.pts, dts: b.pts, duration: 0, key: !!b.key, index: i, _webm: b }));
      plan.timescale = 1e6 / (w.timecodeScale || 1e6) / 1000;
      plan.fps = plan.durationMs ? (blocks.length * 1000) / plan.durationMs : 0;
    }
    return plan;
  }
  const m = parseMp4(bytes);
  const track = m.video;
  if (!track) {
    throw new GifxError(m.tracks.length ? 'MP4 contains no decodable video track' : 'MP4 has no tracks', {
      code: ErrorCode.INPUT_NO_VIDEO_TRACK,
      data: { handlers: m.tracks.map((t) => t.handler), brands: m.brands },
    });
  }
  if (m.encrypted) throw new GifxError('this MP4 is DRM-protected (pssh/sinf present)', { code: ErrorCode.INPUT_ENCRYPTED });
  plan.container = m;
  plan.track = track;
  plan.width = track.width || m.width;
  plan.height = track.height || m.height;
  plan.rotation = m.rotation || 0;
  plan.durationMs = m.durationMs || (track.durationMs || 0);
  plan.fps = m.fps || (track.timescale && track.avgDuration ? track.timescale / track.avgDuration : 0);
  plan.codec = m.codecString || track.codecString || track.codec;
  plan.timescale = track.timescale || 1000;
  plan.frameTimes = mp4FrameTimes(track);
  plan.samples = track.samples;
  plan.seekable = !!(track.samples && track.samples.every((s) => s.offset));
  plan.fragmented = !!m.fragmented;
  plan.fastStart = !!m.fastStart;
  if (m.partial) plan.warnings.push({ code: ErrorCode.INPUT_CORRUPT, message: 'container parsed with truncation; metadata may be incomplete' });
  if (track.offsetsInvalid) plan.seekable = false;
  if (track.codecString) plan.decoderConfig = { codec: track.codecString, codedWidth: plan.width, codedHeight: plan.height, description: track.description };
  if (track.cleanAperture) plan.crop = track.cleanAperture;
  plan.hasAlpha = false;
  return plan;
}

async function webcodecsSource(bytes, plan, opts, ctx) {
  const src = { bytes, track: plan.track, frameTimes: plan.frameTimes, container: plan.container, format: plan.format, webm: plan.samples && plan.samples[0]?._webm ? plan : null };
  if (!src.track) {
    // WebM path: synthesise a track-like object from the block index.
    src.track = {
      samples: plan.samples.map((s, i) => ({ offset: s.offset, size: s.size, pts: s.pts, dts: s.pts, duration: 0, key: s.key })),
      timescale: plan.timescale,
      width: plan.width,
      height: plan.height,
      codecString: (plan.decoderConfig && plan.decoderConfig.codec) || 'vp8',
      description: plan.decoderConfig && plan.decoderConfig.description,
    };
  }
  const inPointMs = opts.start || 0;
  const outPointMs = opts.end != null ? opts.end : opts.duration != null ? inPointMs + opts.duration : undefined;
  const driver = {
    kind: 'webcodecs',
    meta: {
      width: plan.width,
      height: plan.height,
      durationMs: plan.durationMs,
      fps: plan.fps,
      frameCount: plan.frameTimes ? plan.frameTimes.length : plan.samples.length,
      codec: plan.codec,
      rotation: plan.rotation,
      hasAlpha: plan.hasAlpha,
      seekable: plan.seekable,
      decoder: 'webcodecs',
      warnings: plan.warnings,
    },
    plan,
    async *frames(framePlan = {}) {
      const indices = framePlan.indices || null;
      const gen = decodeTrackFrames(src, {
        frameIndices: indices,
        inPointMs: framePlan.startMs ?? inPointMs,
        outPointMs: framePlan.endMs ?? outPointMs,
        maxInFlight: opts.maxInFlight || 8,
        strict: opts.strict,
        signal: opts.signal,
        opaque: opts.transparent === false,
        onWarning: (e, at) => ctx.warnings.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, atSample: at, message: String(e && e.message) }),
        decoderFactory: opts.decoderFactory,
        codec: opts.codec,
        frameToRasterOpts: { opaque: opts.transparent === false },
      });
      for await (const f of gen) yield f;
    },
    async close() {},
  };
  return driver;
}

/* -------------------------------------------------------- <video> fallback */

/**
 * Capture policy for the media-element backend.
 * Pure + exported so the tradeoff is unit-testable (and documented).
 *
 * @param {{wantFrames:number, sourceFrames:number, durationMs:number, coveredMs?:number,
 *   hasTrim:boolean, seekCostMs:number, playCostMs:number, playbackRate?:number, rVFC:boolean}} p
 * @returns {{mode:'play'|'seek', reason:string, estMs:number}}
 */
export function chooseCapturePolicy(p) {
  const coverage = p.sourceFrames > 0 ? p.wantFrames / p.sourceFrames : 1;
  // Play-through is bounded by *wall clock*: it has to run the media in real
  // time from the first wanted frame to the last, no matter how few frames we
  // sample. Seeking is bounded by per-frame cost. That asymmetry is the whole
  // decision: sparse frames late in a long clip are cheaper to seek.
  const rate = p.playbackRate || 1;
  const spanMs = p.coveredMs != null ? p.coveredMs : p.durationMs || 0;
  const playCost = spanMs / rate + p.wantFrames * p.playCostMs + (p.hasTrim ? p.seekCostMs : 0);
  const seekCost = p.wantFrames * p.seekCostMs;
  if (!p.rVFC) {
    // Without requestVideoFrameCallback, play-through means sampling on
    // timeupdate (~4 Hz in some browsers) — never worth it for accuracy.
    return { mode: 'seek', reason: 'no-requestVideoFrameCallback', estMs: seekCost };
  }
  if (p.wantFrames <= 3) return { mode: 'seek', reason: 'few-frames', estMs: seekCost };
  if (coverage > 0.35 && !p.hasTrim && playCost < seekCost) return { mode: 'play', reason: 'high-coverage-sequential', estMs: playCost };
  if (playCost * 1.25 < seekCost) return { mode: 'play', reason: 'cheaper', estMs: playCost };
  return { mode: 'seek', reason: 'accurate-or-sparse', estMs: seekCost };
}

/**
 * Media-element source. Works everywhere a browser does, including old Safari,
 * and needs no demuxing. `elFactory`/`urlFactory` are injectable for tests.
 */
export async function videoElementSource(element, opts = {}, ctx = { warnings: [] }, meta = null) {
  const el = element;
  const doc = typeof document !== 'undefined' ? document : null;
  if (!meta) {
    await once(el, ['loadedmetadata'], opts.signal, opts.metadataTimeoutMs || 20000).catch((e) => {
      throw wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: `media metadata never loaded (${mediaError(el)})`, data: { error: mediaError(el) } });
    });
  }
  const w = el.videoWidth || el.width || 0;
  const h = el.videoHeight || el.height || 0;
  if (!w || !h) throw new GifxError(`media element reported ${w}x${h} — the codec is not supported by this browser`, { code: ErrorCode.INPUT_UNSUPPORTED_CODEC, data: { error: mediaError(el) } });
  const durationMs = (Number.isFinite(el.duration) ? el.duration : 0) * 1000;
  const estimatedFps = opts.fps || estimateElementFps(el) || 0;
  const info = meta || {
    width: w,
    height: h,
    durationMs,
    fps: estimatedFps,
    frameCount: estimatedFps && durationMs ? Math.round((durationMs / 1000) * estimatedFps) : Math.round(durationMs / 100),
    source: 'video-element',
    decoder: 'media-element',
    hasAlpha: false,
  };
  const wantCount = Math.max(1, Math.min(info.frameCount || 1, opts.maxFrames || 1e9));
  const policy = chooseCapturePolicy({
    wantFrames: wantCount,
    sourceFrames: info.frameCount || wantCount,
    durationMs: info.durationMs,
    hasTrim: !!(opts.start || opts.end),
    seekCostMs: opts.seekCostMs || 45,
    playCostMs: opts.playCostMs || 6,
    rVFC: typeof el.requestVideoFrameCallback === 'function',
  });
  const canvas = (opts.canvasFactory && opts.canvasFactory(info.width, info.height)) || createCanvasLike(info.width, info.height);
  if (!canvas) throw new GifxError('no canvas available for the <video> fallback', { code: ErrorCode.NO_OFFSCREEN_CANVAS });
  const g = canvas.getContext('2d', { alpha: false, willReadFrequently: true, desynchronized: true });
  const schedule = planFrameRate(
    frameTimestamps(info, opts),
    opts.fps || info.fps || 10,
    { startMs: opts.start || 0, endMs: opts.end != null ? opts.end : opts.duration != null ? (opts.start || 0) + opts.duration : info.durationMs, dupPolicy: opts.dupPolicy || 'keep' }
  );
  return {
    kind: 'video-element',
    meta: { ...info, policy },
    async *frames() {
      if (policy.mode === 'play') yield* playCapture(el, g, canvas, schedule, opts, ctx, info);
      else yield* seekCapture(el, g, canvas, schedule, opts, ctx, info);
    },
    async close() {
      try {
        el.pause?.();
      } catch {
        /* detached */
      }
    },
  };
  void doc;
}

function frameTimestamps(info, opts) {
  const fps = info.fps || 30;
  const n = info.frameCount || Math.max(1, Math.round(((info.durationMs || 0) / 1000) * fps));
  const step = 1000 / fps;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = i * step;
  if (opts.frameTimes && opts.frameTimes.length === n) return opts.frameTimes;
  return out;
}

function estimateElementFps(el) {
  // Chromium exposes `mozFrameRate`/`requestVideoFrameCallback` metadata; where
  // neither exists we assume the source is 30 but *never* plan more frames than
  // the element can actually deliver, so the fps planner clamps it later.
  try {
    if (typeof el.mozFrameRate === 'number' && el.mozFrameRate > 0) return el.mozFrameRate;
    const cbs = el.getVideoPlaybackQuality?.();
    if (cbs && cbs.totalVideoFrames > 4 && el.currentTime > 0.25) return cbs.totalVideoFrames / el.currentTime;
  } catch {
    /* older engines throw on this */
  }
  return 0;
}

async function* seekCapture(el, g, canvas, schedule, opts, ctx, info) {
  const seen = new Set();
  for (let i = 0; i < schedule.schedule.length; i++) {
    const slot = schedule.schedule[i];
    const t = slot.ptsMs / 1000;
    if (seen.has(t)) continue;
    seen.add(t);
    el.pause?.();
    const ok = await seekTo(el, t, opts.seekTimeoutMs || 5000, opts.fastSeek !== false);
    if (!ok) {
      ctx.warnings.push({ code: ErrorCode.INPUT_SEEK_FAILED, message: `seek to ${t.toFixed(3)}s never completed`, frame: i });
      if (opts.strict) throw new GifxError(`seek to ${t.toFixed(3)}s failed (${mediaError(el) || 'timeout'})`, { code: ErrorCode.INPUT_SEEK_FAILED });
      continue;
    }
    if (opts.shouldStop && opts.shouldStop()) return;
    g.drawImage(el, 0, 0, canvas.width, canvas.height);
    const r = new Raster(canvas.width, canvas.height);
    const d = g.getImageData(0, 0, canvas.width, canvas.height);
    r.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, r.data.length));
    r.opaque = true;
    r.pts = slot.ptsMs;
    yield { raster: r, ptsMs: slot.ptsMs, index: i, durationMs: slot.durationMs, owned: true };
    if (opts.onFrame) opts.onFrame(i, schedule.schedule.length);
    await yieldToBrowser();
  }
}

async function* playCapture(el, g, canvas, schedule, opts, ctx, info) {
  const out = [];
  let done = false;
  let error = null;
  let lastT = -1;
  let index = 0;
  const push = () => {
    const t = el.currentTime * 1000;
    // pick the schedule slots this frame covers (media clock is not exact)
    while (index < schedule.schedule.length && schedule.schedule[index].ptsMs <= t + 8) {
      const slot = schedule.schedule[index];
      g.drawImage(el, 0, 0, canvas.width, canvas.height);
      const r = new Raster(canvas.width, canvas.height);
      const d = g.getImageData(0, 0, canvas.width, canvas.height);
      r.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, r.data.length));
      r.opaque = true;
      r.pts = slot.ptsMs;
      out.push({ raster: r, ptsMs: slot.ptsMs, index: index, durationMs: slot.durationMs, owned: true, actualPtsMs: t });
      index++;
    }
    lastT = t;
  };
  const onFrame = () => {
    push();
    if (!done) el.requestVideoFrameCallback(onFrame);
  };
  const onEnded = () => {
    done = true;
  };
  const onError = () => {
    error = new GifxError(`media element error during playback (${mediaError(el)})`, { code: ErrorCode.INPUT_DECODE_FAILED, data: { error: mediaError(el) } });
    done = true;
  };
  el.addEventListener?.('ended', onEnded);
  el.addEventListener?.('error', onError);
  if (schedule.schedule.length) {
    await seekTo(el, (opts.start || schedule.schedule[0].ptsMs) / 1000, opts.seekTimeoutMs || 5000, false);
  }
  el.playbackRate = Math.max(0.25, Math.min(4, opts.playbackRate || 1));
  try {
    await el.play();
  } catch (e) {
    el.removeEventListener?.('ended', onEnded);
    el.removeEventListener?.('error', onError);
    throw wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: `play() was refused (${e.name}: ${e.message}). The element must be muted+autoplay-allowed in a user gesture.` });
  }
  el.requestVideoFrameCallback(onFrame);
  try {
    while (!done || out.length) {
      if (out.length) {
        const f = out.shift();
        if (opts.shouldStop && opts.shouldStop()) return;
        yield f;
        continue;
      }
      if (error) throw error;
      await sleep(4);
      if (typeof el.requestVideoFrameCallback === 'function' && !el._gifxArmed) {
        el._gifxArmed = true;
        el.requestVideoFrameCallback(() => {
          el._gifxArmed = false;
          onFrame();
        });
      }
      if (done && !out.length && el.ended === false && el.paused === false && el.currentTime * 1000 <= lastT + 1) {
        // stalled (no new frames and not ended) — finish what we can
        ctx.warnings.push({ code: ErrorCode.DECODE_NO_OUTPUT, message: 'playback stalled; emitting frames captured so far' });
        break;
      }
    }
  } finally {
    el.pause?.();
    el.removeEventListener?.('ended', onEnded);
    el.removeEventListener?.('error', onError);
  }
}

function seekTo(el, seconds, timeoutMs, allowFastSeek) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (v) => {
      if (settled) return;
      settled = true;
      el.removeEventListener?.('seeked', onSeeked);
      el.removeEventListener?.('error', onError);
      clearTimeout(timer);
      resolve(v);
    };
    const onSeeked = () => finish(true);
    const onError = () => finish(false);
    el.addEventListener?.('seeked', onSeeked);
    el.addEventListener?.('error', onError);
    const target = Math.max(0, Math.min((Number.isFinite(el.duration) ? el.duration : seconds + 0.001) - 0.0005, seconds));
    try {
      if (allowFastSeek && typeof el.fastSeek === 'function') el.fastSeek(target);
      else el.currentTime = target;
    } catch (e) {
      finish(false);
      return;
    }
    const timer = setTimeout(() => {
      // Firefox sometimes never fires `seeked` for a sub-frame delta; accept the
      // current position instead of losing the frame.
      finish(Math.abs(el.currentTime - target) < 0.25);
    }, timeoutMs);
  });
}

function mediaError(el) {
  const e = el && el.error;
  if (!e) return null;
  const map = { 1: 'ABORTED', 2: 'NETWORK', 3: 'DECODE', 4: 'SRC_NOT_SUPPORTED' };
  return { code: e.code, name: map[e.code] || 'UNKNOWN', message: e.message || '' };
}

function once(target, events, signal, timeoutMs) {
  return new Promise((resolve, reject) => {
    let done = false;
    const ok = (e) => {
      if (done) return;
      done = true;
      cleanup();
      resolve(e);
    };
    const bad = (e) => {
      if (done) return;
      done = true;
      cleanup();
      reject(e instanceof Error ? e : new GifxError(`event ${e && e.type} fired`, { code: ErrorCode.INPUT_DECODE_FAILED }));
    };
    const timer = timeoutMs ? setTimeout(() => bad(new GifxError(`timed out after ${timeoutMs}ms waiting for ${events.join('/')}`, { code: ErrorCode.DECODE_TIMEOUT, retryable: true })), timeoutMs) : null;
    const onAbort = () => bad(new GifxError('aborted', { code: ErrorCode.ABORTED }));
    const cleanup = () => {
      for (const ev of events) target.removeEventListener?.(ev, ev === 'error' ? bad : ok);
      signal?.removeEventListener?.('abort', onAbort);
      if (timer) clearTimeout(timer);
    };
    for (const ev of events) target.addEventListener?.(ev, ev === 'error' ? bad : ok, { once: true });
    signal?.addEventListener?.('abort', onAbort, { once: true });
  });
}

async function elementSourceForBytes(bytes, info, opts, ctx) {
  if (typeof document === 'undefined' || typeof URL === 'undefined' || !URL.createObjectURL) {
    throw new GifxError('the <video> fallback needs a DOM and URL.createObjectURL', { code: ErrorCode.ENV_UNSUPPORTED });
  }
  const blob = new Blob([bytes.slice()], { type: info.mime || 'video/mp4' });
  const url = URL.createObjectURL(blob);
  const el = document.createElement('video');
  el.muted = true;
  el.autoplay = false;
  el.playsInline = true;
  el.preload = 'auto';
  el.crossOrigin = opts.crossOrigin || null;
  el.src = url;
  try {
    const src = await videoElementSource(el, opts, ctx);
    const origClose = src.close;
    src.close = async () => {
      await origClose();
      try {
        el.removeAttribute('src');
        el.load?.();
      } catch {
        /* ignore */
      }
      URL.revokeObjectURL(url);
    };
    return src;
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
}

async function imageElementSource(img, opts, ctx) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const r = new Raster(w, h);
  const c = createCanvasLike(w, h);
  if (!c) throw new GifxError('cannot read an <img> without a canvas', { code: ErrorCode.NO_OFFSCREEN_CANVAS });
  c.width = w;
  c.height = h;
  c.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0);
  const d = c.getContext('2d').getImageData(0, 0, w, h);
  r.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, r.data.length));
  return singleRasterSource(r, opts);
}

/* ----------------------------------------------------------------- helpers */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let yieldSupported = typeof scheduler !== 'undefined' && scheduler.yield;
async function yieldToBrowser() {
  if (yieldSupported) {
    try {
      await scheduler.yield();
      return;
    } catch {
      yieldSupported = false;
    }
  }
  await sleep(0);
}

/**
 * Normalize user input into a frame plan the source walks.
 * Pure so it can be tested and reused by the size search.
 */
export function planFrames(meta, opts = {}) {
  const fps = opts.fps == null ? null : Math.max(1, Math.min(60, opts.fps));
  const startMs = Math.max(0, opts.start || 0);
  const endMs = opts.end != null ? Math.max(startMs + 33, opts.end) : opts.duration != null ? startMs + opts.duration : meta.durationMs || 0;
  const total = meta.frameCount || Math.max(1, Math.round(((meta.durationMs || 0) / 1000) * (meta.fps || fps || 30)));
  const maxFrames = Math.max(1, Math.min(opts.maxFrames || 1e9, opts.frames || 1e9));
  const sourceTimes = frameTimestamps(meta, opts);
  const plan = planFrameRate(sourceTimes, fps || meta.fps || 10, { startMs, endMs: endMs || sourceTimes[sourceTimes.length - 1] + 33, dupPolicy: opts.dupPolicy || (opts.dropDuplicates === false ? 'keep' : 'skip') });
  let indices = plan.schedule.map((s) => s.index);
  if (indices.length > maxFrames) {
    // thin out evenly instead of truncating, so the GIF still covers the clip
    const step = indices.length / maxFrames;
    const thinned = [];
    for (let i = 0; i < maxFrames; i++) thinned.push(indices[Math.floor(i * step)]);
    indices = thinned;
  }
  if (opts.frames && opts.frames < indices.length) indices = indices.slice(0, opts.frames);
  const unique = [...new Set(indices)].sort((a, b) => a - b);
  return {
    indices: unique,
    count: unique.length,
    startMs,
    endMs: endMs || (unique.length + 1) * (1000 / (fps || meta.fps || 10)),
    fps: fps || plan.outFps || meta.fps || 10,
    frameMs: 1000 / (fps || plan.outFps || meta.fps || 10),
    durationMs: endMs ? endMs - startMs : unique.length * (1000 / (fps || plan.outFps || 10)),
    total,
    dropped: indices.length - unique.length,
    schedule: plan.schedule,
  };
}

export { Raster };

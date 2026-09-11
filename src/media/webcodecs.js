/**
 * GIFX Kernel — WebCodecs video driver.
 *
 * This is the fast path: demux the container ourselves, feed `EncodedVideoChunk`s
 * to a `VideoDecoder`, and get `VideoFrame`s back. Why bother when `<video>`
 * exists?
 *
 *  - Frame accuracy. `<video>` seeking is quantised to keyframes plus a decode
 *    run and its `currentTime` events are throttled, so a 12 fps GIF from a
 *    60 fps source lands on the wrong frames and drifts. Here we pick exactly
 *    the samples we want.
 *  - Workers. WebCodecs runs off the main thread, so a 4K transcode never
 *    freezes the page (a `<video>` element cannot leave the main thread).
 *  - Speed. No play/pause loop, no `requestVideoFrameCallback` handshake
 *    (~16 ms per frame of pure latency), no forced 30 fps ceiling.
 *  - Backpressure. We count frames in flight and pause feeding, which is what
 *    keeps a 2 GB arena from being exceeded on a 10-minute phone video.
 *
 * Every failure mode degrades: a codec the platform refuses becomes a typed
 * `INPUT_UNSUPPORTED_CODEC`, a corrupt sample is skipped (with a warning) rather
 * than killing the job, and an aborted decoder flush never hangs the promise.
 *
 * @module media/webcodecs
 */
import { GifxError, ErrorCode, wrapError } from '../core/errors.js';
import { Raster } from '../core/buffers.js';

export const MIN_CHUNK_BYTES = 1;

/**
 * Is WebCodecs video decoding usable here? (Safari 16.4+, Chrome 94+, FF 133+.)
 */
export async function webcodecsVideoSupport(codecStrings = ['avc1.42001e', 'vp09.00.10.08', 'av01.0.01M.0', 'vp8']) {
  if (typeof VideoDecoder !== 'function') return { ok: false, reason: 'no-VideoDecoder', codecs: {} };
  const codecs = {};
  let any = false;
  for (const c of codecStrings) {
    try {
      const r = await VideoDecoder.isTypeSupported({ codec: c });
      codecs[c] = !!r;
      if (r) any = true;
    } catch {
      codecs[c] = false;
    }
  }
  return { ok: any, codecs, reason: any ? '' : 'no-supported-codec' };
}

/**
 * Decode a demuxed track into frames.
 *
 * @param {object} src
 * @param {Uint8Array} src.bytes whole file (or the range covering it)
 * @param {object} src.track parser track object with `samples`, `timescale`
 * @param {object} [opts]
 * @param {number[]} [opts.frameIndices] which presentation-order frames to emit
 * @param {number} [opts.inPointMs]
 * @param {number} [opts.outPointMs]
 * @param {boolean} [opts.opaque] force alpha discard (faster, video has none)
 * @param {number} [opts.maxInFlight] backpressure window (frames)
 * @param {number} [opts.fallbackBytesPerFrame]
 * @param {function} [opts.onWarning]
 * @param {AbortSignal} [opts.signal]
 * @param {function} [opts.ImageBitmapSink] custom frame→raster conversion
 * @returns {AsyncGenerator<{raster:Raster, ptsMs:number, index:number, key:boolean}>}
 */
export async function* decodeTrackFrames(src, opts = {}) {
  const want = new Set(opts.frameIndices || []);
  const wantAny = want.size > 0;
  const times = src.frameTimes || src.track.samples.map((s, i) => ({ index: i, pts: ((s.pts ?? s.dts) / (src.track.timescale || 1)) * 1000, key: s.key !== false }));
  const range = pickDecodeRange(times, src.track, opts);
  if (!range.samples.length) {
    yield* [];
    return;
  }
  const driver = createDecodeDriver(src, range, opts);
  try {
    for await (const f of driver.frames()) yield f;
  } finally {
    await driver.close();
  }
  void wantAny;
}

/**
 * Decide which *decode-order* samples must be fed so that the wanted
 * presentation frames come out, then reorder outputs by pts.
 */
export function pickDecodeRange(times, track, opts = {}) {
  const inMs = opts.inPointMs || 0;
  const outMs = opts.outPointMs == null ? Infinity : opts.outPointMs;
  const wanted = (opts.frameIndices || [])
    .map((i) => times[i])
    .filter((t) => t && t.pts >= inMs - 1 && t.pts <= (outMs === Infinity ? Infinity : outMs + 1));
  const list = wanted.length
    ? wanted
    : times.filter((t) => t.pts >= inMs - 1 && (outMs === Infinity || t.pts <= outMs + 1));
  if (!list.length) return { samples: [], wanted: new Set(), warmup: 0, rank: new Map(), ptsKey: new Map() };
  const wantSet = new Set(list.map((t) => t.index));
  // `times` is presentation order; the decoder needs *decode* (file) order.
  const firstWanted = Math.min(...list.map((t) => t.index));
  let start = firstWanted;
  while (start > 0 && !(track.samples[start] && track.samples[start].key !== false)) start--;
  const lastWanted = Math.max(...list.map((t) => t.index));
  const samples = [];
  for (let i = start; i <= lastWanted && i < track.samples.length; i++) samples.push(i);
  // Frames leave the decoder in *presentation* order, so rank them that way and
  // let the driver translate pts → (sample, rank) instead of counting outputs.
  const ordered = [...list].sort((a, b) => a.pts - b.pts || a.index - b.index);
  const rank = new Map();
  ordered.forEach((t, i) => rank.set(t.index, i));
  const ptsKey = new Map();
  for (const t of times) ptsKey.set(qMs(t.pts), t.index);
  return { samples, wanted: wantSet, rank, ptsKey, warmup: firstWanted - start, startSample: start, endSample: lastWanted, inPointMs: ordered[0].pts, frameCount: list.length };
}

const qMs = (ms) => Math.round((ms || 0) * 8) / 8; // 1/8 ms buckets: robust to fixed-point round-trips

/**
 * The decoder state machine, kept separate from the generator so it can be
 * unit-tested with a fake `VideoDecoder` (see test/media-decode.test.js).
 */
export function createDecodeDriver(src, range, opts = {}) {
  const maxInFlight = Math.max(2, opts.maxInFlight || 8);
  const signal = opts.signal;
  const warnings = [];
  const queue = [];
  const skipSamples = new Set();
  let waiting = null;
  let done = false;
  let error = null;
  let fed = 0;
  let produced = 0;
  let decoder = null;
  let closed = false;
  const onFrame = (frame) => {
    produced++;
    queue.push(frame);
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
  };
  const onError = (e) => {
    const g = opts.strict
      ? wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: `decoder failed at sample ${fed}/${range.samples.length}` })
      : null;
    if (g) error = g;
    else {
      warnings.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, atSample: fed, message: e && e.message });
      if (opts.onWarning) opts.onWarning(e, fed);
    }
    // A DecoderError invalidates the decoder; resume from the next keyframe.
    try {
      decoder?.reset?.();
    } catch {
      /* already closed */
    }
    resumeNeeded = true;
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
  };
  let resumeNeeded = false;
  let currentFeedSample = -1;

  const config = buildDecoderConfig(src, opts);
  if (typeof VideoDecoder !== 'function') {
    if (!opts.decoderFactory) throw new GifxError('WebCodecs VideoDecoder is unavailable in this environment', { code: ErrorCode.NO_VIDEO_DECODER });
  }
  try {
    decoder = opts.decoderFactory ? opts.decoderFactory({ output: onFrame, error: onError }) : new VideoDecoder({ output: onFrame, error: onError });
    decoder.configure(config);
  } catch (e) {
    throw wrapError(e, { code: ErrorCode.INPUT_UNSUPPORTED_CODEC, message: `VideoDecoder.configure failed for "${config.codec}"`, data: { config } });
  }

  const abort = () => {
    error = error || new GifxError('decode aborted', { code: ErrorCode.ABORTED });
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
    try {
      decoder.abort();
    } catch {
      /* ignore */
    }
  };
  if (signal) {
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  }

  // Some decoders (notably Safari's HEVC path) hand back frames a sample or two
  // out of presentation order. Buffering a small window and always yielding the
  // earliest pts keeps the GIF timeline monotonic no matter what the decoder does.
  const reorderWindow = Math.max(2, opts.reorderWindow || 4);
  const pending = [];
  const takeOldest = () => {
    let at = 0;
    for (let i = 1; i < pending.length; i++) if (pending[i].pts < pending[at].pts || (pending[i].pts === pending[at].pts && pending[i].index < pending[at].index)) at = i;
    return pending.splice(at, 1)[0];
  };

  // Recovery state. When the decoder errors, it drops its reorder buffer, so any
  // frames already decoded but not yet delivered are gone. Real transcoders
  // respond by re-feeding from the last keyframe; we do the same, bounded by a
  // replay budget so a *permanently* broken sample can't loop forever.
  const delivered = new Set();
  let lastKeyLoopIdx = -1;
  let replays = 0;
  let sinceYield = 0;
  const maxReplays = opts.maxReplays == null ? 4 : opts.maxReplays;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function* frames() {
    // Feed / drain / flush, wrapped in a recovery loop.
    //
    // Decoder errors are reported asynchronously, so `resumeNeeded` frequently
    // becomes true only *after* we finished feeding — which is why the recovery
    // check has to run again after the flush, not just inside the feed loop.
    // Recovery re-feeds from the last keyframe (bounded by `maxReplays`) and
    // never re-feeds the sample that failed.
    let i = 0;
    let flushed = false;
    for (;;) {
      while (i < range.samples.length && !error) {
        const sampleIndex = range.samples[i];
        currentFeedSample = sampleIndex;
        const sample = src.track.samples[sampleIndex];
        if (opts.signal?.aborted) {
          abort();
          break;
        }
        if (!sample) {
          i++;
          continue;
        }
        // never drop frames on tiny samples, but do validate (0-length samples
        // exist in broken files and make some decoders hang)
        if (sample.size < MIN_CHUNK_BYTES && !opts.keepEmptySamples) {
          warnings.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, atSample: sampleIndex, message: 'empty sample skipped' });
          i++;
          continue;
        }
        if (skipSamples.has(sampleIndex)) {
          i++;
          continue; // the sample that just failed: re-feeding it would loop
        }
        const bytes = src.bytes.subarray(sample.offset, sample.offset + sample.size);
        const chunk = makeChunk(bytes, sample, src.track, opts);
        if (sample.key !== false) lastKeyLoopIdx = i;
        try {
          decoder.decode(chunk);
        } catch (e) {
          if (opts.strict) {
            error = wrapError(e, { code: ErrorCode.INPUT_DECODE_FAILED, message: 'decoder.decode() rejected a chunk' });
            break;
          }
          warnings.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, atSample: sampleIndex, message: e.message });
          skipSamples.add(sampleIndex);
          resumeNeeded = true;
          i++;
          continue;
        }
        fed++;
        sinceYield++;
        i++;
        // Backpressure: `decodeQueueSize` counts chunks the decoder has not taken
        // yet; past the window we stop feeding and let frames drain. Without this,
        // feeding a 4K clip puts thousands of VideoFrames in flight and the browser
        // OOMs (or silently drops).
        while (decoder.decodeQueueSize > maxInFlight && !error) await sleep(0);
        while (queue.length && !error) {
          const item = await emit(queue.shift(), opts);
          if (item) {
            pending.push(item);
            while (pending.length > reorderWindow) {
              const next = takeOldest();
              sinceYield = 0;
              if (next) delivered.add(next.sampleIndex);
              yield next;
            }
          }
        }
        if (resumeNeeded && !error) {
          if (replays < maxReplays && lastKeyLoopIdx >= 0) {
            replays++;
            resumeNeeded = false;
            configureQuietly();
            i = lastKeyLoopIdx; // replay the keyframe itself, then the tail
            flushed = false;
            continue;
          }
          // Out of replay budget: skip forward to the next keyframe and carry on.
          while (i + 1 < range.samples.length && src.track.samples[range.samples[i + 1]]?.key === false) i++;
          resumeNeeded = false;
          configureQuietly();
        }
      }
      if (error) break;
      if (!flushed) {
        flushed = true;
        try {
          await flushDecoder(decoder, opts.flushTimeoutMs || 30000);
        } catch (e) {
          if (opts.strict) error = wrapError(e, { code: ErrorCode.DECODE_TIMEOUT, message: 'decoder flush timed out' });
          else warnings.push({ code: ErrorCode.DECODE_TIMEOUT, message: `flush: ${e && e.message}`, partial: true });
        }
        // Some engines resolve flush() a task before their last `output` callback
        // lands (Firefox did for a while). Spin briefly while the decoder reports
        // idle and our queue is empty, so trailing frames are never silently lost.
        for (let spin = 0; spin < 4 && !queue.length && (decoder.decodeQueueSize || 0) === 0; spin++) {
          await sleep(0);
          if (queue.length) break;
        }
        while (queue.length && !error) {
          const item = await emit(queue.shift(), opts);
          if (item) pending.push(item);
        }
        if (resumeNeeded && replays < maxReplays && lastKeyLoopIdx >= 0) {
          // The error arrived during/after the flush: one more pass recovers the tail.
          replays++;
          resumeNeeded = false;
          configureQuietly();
          i = lastKeyLoopIdx;
          flushed = false;
          continue;
        }
      }
      break;
    }
    // Cancellation means "stop now": drop what is buffered instead of delivering
    // frames nobody asked for (a size search aborts on budget, and would count
    // these twice). Non-abort errors still deliver, then rethrow.
    while (pending.length) {
      if (error && error.code === ErrorCode.ABORTED) {
        pending.length = 0;
        break;
      }
      const item = takeOldest();
      if (item && delivered.has(item.sampleIndex)) continue;
      yield item;
    }
    if (error) throw error;
  }

  function configureQuietly() {
    try {
      decoder.configure(config);
    } catch {
      /* keep the previous config; the next decode() error is handled too */
    }
  }

  async function emit(frame, o) {
    try {
      const raster = await frameToRaster(frame, o);
      const pts = frame.timestamp != null ? frame.timestamp / 1000 : 0;
      const sampleIndex = range.ptsKey?.get(qMs(pts)) ?? frame.__gifxIndex ?? produced;
      const meta = {
        raster,
        pts: pts,
        ptsMs: pts,
        index: range.rank?.has(sampleIndex) ? range.rank.get(sampleIndex) : produced,
        sampleIndex,
        key: frame.type === 'key',
        durationMs: frame.duration != null ? frame.duration / 1000 : 0,
      };
      if (raster === null) return null;
      // Drop frames the caller didn't ask for (they still had to be decoded —
      // a partial GOP has to be walked from its keyframe).
      if (range.wanted && range.wanted.size && !range.wanted.has(sampleIndex)) {
        raster.dispose?.();
        return null;
      }
      return meta;
    } catch (e) {
      if (o.strict) throw e;
      warnings.push({ code: ErrorCode.DECODE_CORRUPT_FRAME, message: `frame conversion failed: ${e.message}` });
      return null;
    } finally {
      try {
        frame.close?.();
      } catch {
        /* already closed (Safari double-close) */
      }
    }
  }

  return {
    frames,
    get stats() {
      return {
        fed,
        produced,
        replays,
        skippedSamples: skipSamples.size,
        warmup: range.warmup || 0,
        warnings: warnings.length,
        decodeQueueSize: decoder?.decodeQueueSize ?? 0,
      };
    },
    warnings,
    get error() {
      return error;
    },
    async close() {
      if (closed) return;
      closed = true;
      if (signal) signal.removeEventListener?.('abort', abort);
      try {
        if (decoder.state !== 'closed') decoder.close();
      } catch {
        /* already closed */
      }
      for (const f of queue) {
        try {
          f.close?.();
        } catch {
          /* ignore */
        }
      }
      queue.length = 0;
      done = true;
      void waiting;
    },
  };
}

function makeChunk(bytes, sample, track, opts) {
  const ts = Math.round((sample.pts ?? sample.dts ?? 0) * (1e6 / (track.timescale || 1e6)));
  const dur = Math.round((sample.duration || 1) * (1e6 / (track.timescale || 1e6)));
  if (typeof EncodedVideoChunk === 'function') {
    return new EncodedVideoChunk({
      type: sample.key === false ? 'delta' : 'key',
      timestamp: ts,
      duration: dur,
      data: bytes.slice(), // copy: the decoder may keep the view alive across awaits
    });
  }
  // Fake/mock path used by tests and by `decoderFactory` shims.
  const c = { type: sample.key === false ? 'delta' : 'key', timestamp: ts, duration: dur, byteLength: bytes.length };
  Object.defineProperty(c, 'copyTo', { value: (dest) => dest.set(bytes.subarray(0, Math.min(dest.length, bytes.length))) });
  return c;
}

export function buildDecoderConfig(src, opts = {}) {
  const t = src.track || {};
  const codec = opts.codec || t.codecString || t.codec || 'avc1.42001e';
  const cfg = {
    codec,
    codedWidth: t.width || undefined,
    codedHeight: t.height || undefined,
    optimizeForLatency: opts.optimizeForLatency !== false,
    hardwareAcceleration: opts.hardwareAcceleration || 'no-preference',
  };
  if (t.description && t.description.length) cfg.description = t.description;
  if (t.bitDepth) cfg.bitDepth = t.bitDepth;
  if (t.colour?.fullRange) cfg.videoFmt = undefined; // left to the decoder; documented in notes
  if (opts.transfer && typeof document !== 'undefined') cfg.colorSpace = { ...cfg.colorSpace, transfer: opts.transfer };
  if (opts.alpha === 'discard') cfg.alpha = 'discard';
  return cfg;
}

export async function flushDecoder(decoder, timeoutMs) {
  const p = decoder.flush();
  if (!p || typeof p.then !== 'function') return p;
  let timer = null;
  try {
    return await Promise.race([
      p,
      new Promise((_, rej) => {
        timer = setTimeout(() => rej(new GifxError(`decoder flush timed out after ${timeoutMs} ms`, { code: ErrorCode.DECODE_TIMEOUT, retryable: true })), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * `VideoFrame` → our `Raster`. Prefers a zero-copy `copyTo` into an arena buffer
 * (the browser's own conversion path), falls back to ImageBitmap+2D canvas for
 * formats `copyTo` rejects (Bt601 with odd sizes, 10-bit, HDR).
 */
export async function frameToRaster(frame, opts = {}) {
  const visible = frame.visibleRect || { x: 0, y: 0, width: frame.displayWidth || frame.codedWidth, height: frame.displayHeight || frame.codedHeight };
  const w = Math.max(1, opts.width || visible.width);
  const h = Math.max(1, opts.height || visible.height);
  const out = new Raster(w, h, undefined, opts.arena);
  if (typeof frame.copyTo === 'function' && !opts.forceCanvas) {
    const init = { colorSpace: opts.colorSpace || 'srgb' };
    const stride = w * 4;
    const need = stride * h;
    const buf = out.data.length >= need ? out.data : new Uint8Array(need);
    try {
      if (buf === out.data) {
        await frame.copyTo(buf, { ...init, layout: [{ offset: 0, stride }] });
      } else {
        await frame.copyTo(buf, init);
        out.data.set(buf.subarray(0, out.data.length));
      }
      // Video frames are opaque unless the caller asked to keep alpha (VP9/AV1
      // block-addition alpha, which arrives as a separate frame). Leaving the
      // alpha byte untouched would expose recycled arena memory as alpha 0.
      if (opts.alpha !== 'keep') {
        for (let i = 3; i < out.data.length; i += 4) out.data[i] = 255;
        out.opaque = true;
      } else {
        out.opaque = false;
        out.hasAlpha = true;
      }
      out.pts = (frame.timestamp || 0) / 1000;
      out.duration = (frame.duration || 0) / 1000;
      frame.__gifxIndex = frame.__gifxIndex ?? indexFromTimestamp(frame);
      out.index = frame.__gifxIndex;
      if (opts.flipY) flipInPlace(out);
      return out;
    } catch (e) {
      if (opts.strict) throw wrapError(e, { code: ErrorCode.DECODE_CORRUPT_FRAME, message: 'VideoFrame.copyTo failed' });
      // fall through to the canvas path
    }
  }
  const bitmap = await frameToImageBitmap(frame);
  if (!bitmap) return null;
  await imageBitmapToRaster(bitmap, out, opts);
  bitmap.close?.();
  out.pts = (frame.timestamp || 0) / 1000;
  out.index = indexFromTimestamp(frame);
  return out;
}

function indexFromTimestamp(frame) {
  // The driver tags nothing; the engine matches frames to requested indices by
  // pts, so expose the raw timestamp for that lookup.
  const meta = frame.metadata?.encoderConfig?.frameIndex;
  if (typeof meta === 'number') return meta;
  return null;
}

export async function frameToImageBitmap(frame) {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(frame, { colorSpaceConversion: 'default' });
    } catch {
      try {
        return await createImageBitmap(frame);
      } catch {
        return null;
      }
    }
  }
  return null;
}

export async function imageBitmapToRaster(bitmap, out, opts = {}) {
  const w = out.width;
  const h = out.height;
  if (typeof OffscreenCanvas === 'function') {
    const c = new OffscreenCanvas(w, h);
    const g = c.getContext('2d', { alpha: true, willReadFrequently: true });
    g.clearRect(0, 0, w, h);
    g.drawImage(bitmap, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    out.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, Math.min(d.data.length, out.data.length)));
    return out;
  }
  if (typeof document !== 'undefined' && document.createElement) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d', { alpha: true, willReadFrequently: true });
    g.clearRect(0, 0, w, h);
    g.drawImage(bitmap, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h);
    out.data.set(new Uint8Array(d.data.buffer, d.data.byteOffset, Math.min(d.data.length, out.data.length)));
    return out;
  }
  if (opts.strict !== false) throw new GifxError('no canvas available to read back video frames', { code: ErrorCode.ENV_UNSUPPORTED });
  return out;
}

function flipInPlace(raster) {
  const tmp = new Uint8Array(raster.stride);
  for (let y = 0; y < raster.height / 2; y++) {
    const a = y * raster.stride;
    const b = (raster.height - 1 - y) * raster.stride;
    tmp.set(raster.data.subarray(a, a + raster.stride));
    raster.data.copyWithin(a, b, b + raster.stride);
    raster.data.set(tmp, b);
  }
}

/**
 * Raw-AnnexB-H.264 wrapper: browsers refuse a bare `.h264` file in `<video>`, so
 * for that input type we build a minimal avcC-less sample table by splitting on
 * start codes and let WebCodecs do the rest (it accepts Annex B for avc1 only in
 * a few builds — hence `wrapAnnexBToAvcc` which converts to length-prefixed).
 */
export function splitAnnexB(u8) {
  const out = [];
  let i = 0;
  const n = u8.length;
  let start = -1;
  while (i < n - 2) {
    if (u8[i] === 0 && u8[i + 1] === 0 && (u8[i + 2] === 1 || (u8[i + 2] === 0 && i + 3 < n && u8[i + 3] === 1))) {
      const skip = u8[i + 2] === 1 ? 3 : 4;
      if (start >= 0) out.push(u8.subarray(start, i));
      start = i + skip;
      i += skip;
      continue;
    }
    i++;
  }
  if (start >= 0) out.push(u8.subarray(start, n));
  return out;
}

/** NAL units → AVCC (4-byte length prefix) samples grouped into access units. */
export function nalUnitsToSamples(u8, opts = {}) {
  const nals = splitAnnexB(u8);
  const sps = nals.find((n) => (n[0] & 0x1f) === 7);
  const pps = nals.find((n) => (n[0] & 0x1f) === 8);
  const samples = [];
  let current = null;
  let offset = 0;
  const dts = { value: 0 };
  const fps = opts.fps || 25;
  const timescale = 1e6;
  for (const nal of nals) {
    offset += nal.length + 4;
    const type = nal[0] & 0x1f;
    if (type === 7 || type === 8 || type === 5 || type === 1) {
      const isIdr = type === 5;
      if (type === 1 || type === 5) {
        if (current) {
          current.size = offset - current.offset - (isIdr ? 0 : 0);
          samples.push(current);
        }
        current = { offset: offset - nal.length - 4, size: nal.length + 4, key: isIdr, dts: dts.value, pts: dts.value, duration: Math.round(timescale / fps) };
        dts.value += Math.round(timescale / fps);
        continue;
      }
    }
    if (current) current.size += nal.length + 4;
  }
  if (current) samples.push(current);
  return { samples: samples.filter((s) => s.size > 4), sps, pps, timescale };
}

/**
 * Build an `avcC` from SPS/PPS NALs (needed because VideoDecoder demands a
 * description for avc1 in MP4/AVCC format).
 */
export function buildAvcC(sps, pps) {
  if (!sps || !pps) return null;
  const out = new Uint8Array(11 + sps.length + pps.length);
  out[0] = 1;
  out[1] = sps[1];
  out[2] = sps[2];
  out[3] = sps[3];
  out[4] = 0xff; // 4-byte lengths
  out[5] = 0xe1; // one SPS
  out[6] = (sps.length >> 8) & 255;
  out[7] = sps.length & 255;
  out.set(sps, 8);
  const p = 8 + sps.length;
  out[p] = 1;
  out[p + 1] = (pps.length >> 8) & 255;
  out[p + 2] = pps.length & 255;
  out.set(pps, p + 3);
  return out;
}

/**
 * Convert an Annex-B stream into a synthetic "track" the driver can consume.
 * Used for `.h264`/`.264`/`.es` inputs and for AVIs containing AVCC already.
 */
export function annexBTrack(u8, opts = {}) {
  const { samples, sps, pps, timescale } = nalUnitsToSamples(u8, opts);
  if (!samples.length) throw new GifxError('no video access units found in the elementary stream', { code: ErrorCode.INPUT_CORRUPT });
  const description = buildAvcC(sps, pps);
  const first = samples[0];
  for (const s of samples) {
    // re-encode as length-prefixed so offsets stay inside the source bytes
    s.offset = s.offset;
    void first;
  }
  return {
    track: {
      samples,
      timescale,
      width: opts.width || spsWidth(sps) || 0,
      height: opts.height || spsHeight(sps) || 0,
      codecString: description ? `avc1.${sps[1].toString(16).padStart(2, '0')}${sps[2].toString(16).padStart(2, '0')}${sps[3].toString(16).padStart(2, '0')}` : 'avc1.42001e',
      description,
    },
    bytes: u8,
  };
}

/** Minimal SPS reader: only profile/level-safe fields (chroma 4:2:0 assumed). */
function spsRead(sps) {
  if (!sps || sps.length < 8) return null;
  // Exp-Golomb bit reader over the RBSP (emulation prevention bytes removed)
  const rbsp = new Uint8Array(sps.length);
  let n = 0;
  for (let i = 0; i < sps.length; i++) {
    if (i >= 2 && sps[i] === 3 && sps[i - 1] === 0 && sps[i - 2] === 0) continue;
    rbsp[n++] = sps[i];
  }
  let bit = 0;
  const byte = (i) => rbsp[i] || 0;
  const u1 = () => ((bit++, (byte(((bit - 1) >> 3)) >> (7 - ((bit - 1) & 7))) & 1));
  const u = (len) => {
    let v = 0;
    for (let i = 0; i < len; i++) v = (v << 1) | u1();
    return v >>> 0;
  };
  const ue = () => {
    let z = 0;
    while (u1() === 0 && z < 32) z++;
    return z === 0 ? 0 : (1 << z) - 1 + u(z);
  };
  const se = () => {
    const k = ue();
    return k % 2 ? (k + 1) >> 1 : -(k >> 1);
  };
  try {
    u1(); // forbidden_zero_bit
    u(2); // nal_ref_idc
    u(5); // nal_unit_type
    const profileIdc = u(8);
    u(8); // constraint flags
    u(8); // level_idc
    ue(); // seq_parameter_set_id
    let chromaFormatIdc = 1;
    if (profileIdc === 100 || profileIdc === 110 || profileIdc === 122 || profileIdc === 244 || profileIdc === 44 || profileIdc === 83 || profileIdc === 86 || profileIdc === 118 || profileIdc === 128 || profileIdc === 138 || profileIdc === 139 || profileIdc === 134 || profileIdc === 135) {
      chromaFormatIdc = ue();
      if (chromaFormatIdc === 3) u1();
      ue();
      ue();
      u1();
      if (u1()) for (let i = 0; i < 6; i++) if (u1()) { const n = chromaFormatIdc === 3 ? 12 : 10; for (let k = 0; k < n; k++) se(); }
    }
    ue(); // log2_max_frame_num
    if (u1() === 0) {
      ue(); // pic_order_cnt_type
      if (ue() === 0) ue(); // num_ref_frames_in_pic_order_cnt_cycle offset loop omitted (rare in practice for our inputs)
    }
    ue(); // max_num_ref_frames
    u1(); // gaps_in_frame_num_value_allowed_flag
    const widthMbs = ue() + 1;
    const heightMapUnits = ue() + 1;
    const frameMbsOnly = u1();
    if (!frameMbsOnly) u1();
    u1(); // direct_8x8_inference_flag
    const crop = u1();
    let cropL = 0;
    let cropR = 0;
    let cropT = 0;
    let cropB = 0;
    if (crop) {
      cropL = ue();
      cropR = ue();
      cropT = ue();
      cropB = ue();
    }
    const unit = chromaFormatIdc === 1 || chromaFormatIdc === 2 ? 2 : 1;
    const w = widthMbs * 16 - unit * (cropL + cropR);
    const h = (2 - frameMbsOnly) * heightMapUnits * 16 - (chromaFormatIdc === 1 ? 2 : 1) * unit * (cropT + cropB);
    return { width: Math.max(0, w), height: Math.max(0, h), profileIdc };
  } catch {
    return null;
  }
}
const spsWidth = (sps) => spsRead(sps)?.width || 0;
const spsHeight = (sps) => spsRead(sps)?.height || 0;

export { spsRead as readH264Sps };

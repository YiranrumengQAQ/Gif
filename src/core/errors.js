/**
 * GIFX Kernel — error taxonomy.
 *
 * Every error thrown by the library is a {@link GifxError} carrying a stable
 * machine-readable `code` (from {@link ErrorCode}) so hosts can branch on
 * failure modes without string matching. Errors also carry `retryable` and
 * `hint` fields produced by the diagnostic engine (see `core/diagnose.js`).
 *
 * @module core/errors
 */

/** Stable error codes. Never reuse, never renumber. */
export const ErrorCode = Object.freeze({
  OK: 'OK',

  /* ---- environment / capability ---- */
  ENV_UNSUPPORTED: 'ENV_UNSUPPORTED',
  NO_WORKERS: 'NO_WORKERS',
  NO_OFFSCREEN_CANVAS: 'NO_OFFSCREEN_CANVAS',
  NO_VIDEO_DECODER: 'NO_VIDEO_DECODER',
  NO_COMPRESSION_STREAM: 'NO_COMPRESSION_STREAM',
  NO_DECOMPRESSION_STREAM: 'NO_DECOMPRESSION_STREAM',
  NO_CREATE_IMAGE_BITMAP: 'NO_CREATE_IMAGE_BITMAP',
  NO_MEDIA_RECORDER: 'NO_MEDIA_RECORDER',
  NO_OPFS: 'NO_OPFS',
  NO_SHARED_ARRAY_BUFFER: 'NO_SHARED_ARRAY_BUFFER',
  INSECURE_CONTEXT: 'INSECURE_CONTEXT',

  /* ---- input ---- */
  INPUT_EMPTY: 'INPUT_EMPTY',
  INPUT_TOO_LARGE: 'INPUT_TOO_LARGE',
  INPUT_UNRECOGNIZED: 'INPUT_UNRECOGNIZED',
  INPUT_CORRUPT: 'INPUT_CORRUPT',
  INPUT_NO_VIDEO_TRACK: 'INPUT_NO_VIDEO_TRACK',
  INPUT_NO_FRAMES: 'INPUT_NO_FRAMES',
  INPUT_UNSUPPORTED_CODEC: 'INPUT_UNSUPPORTED_CODEC',
  INPUT_ENCRYPTED: 'INPUT_ENCRYPTED',
  INPUT_DECODE_FAILED: 'INPUT_DECODE_FAILED',
  INPUT_SEEK_FAILED: 'INPUT_SEEK_FAILED',
  INPUT_TIMEOUT: 'INPUT_TIMEOUT',
  GIF_TOO_MANY_FRAMES: 'GIF_TOO_MANY_FRAMES',

  /* ---- configuration ---- */
  CONFIG_INVALID: 'CONFIG_INVALID',
  CONFIG_UNKNOWN_KEY: 'CONFIG_UNKNOWN_KEY',
  CONFIG_OUT_OF_RANGE: 'CONFIG_OUT_OF_RANGE',
  CONFIG_CONFLICT: 'CONFIG_CONFLICT',

  /* ---- processing ---- */
  ABORTED: 'ABORTED',
  CANCELLED: 'CANCELLED',
  OUT_OF_MEMORY: 'OUT_OF_MEMORY',
  BACKPRESSURE_TIMEOUT: 'BACKPRESSURE_TIMEOUT',
  WORKER_DIED: 'WORKER_DIED',
  WORKER_TIMEOUT: 'WORKER_TIMEOUT',
  TASK_FAILED: 'TASK_FAILED',
  INTERNAL: 'INTERNAL',
  NOT_INITIALIZED: 'NOT_INITIALIZED',
  ALREADY_RUNNING: 'ALREADY_RUNNING',
  STATE_ERROR: 'STATE_ERROR',

  /* ---- output ---- */
  ENCODE_FAILED: 'ENCODE_FAILED',
  OUTPUT_EMPTY: 'OUTPUT_EMPTY',
  OUTPUT_TOO_LARGE: 'OUTPUT_TOO_LARGE',
  TARGET_SIZE_UNREACHABLE: 'TARGET_SIZE_UNREACHABLE',
  NO_ENCODER: 'NO_ENCODER',
  SAVE_FAILED: 'SAVE_FAILED',
  CLIPBOARD_FAILED: 'CLIPBOARD_FAILED',

  /* ---- demux / decode ---- */
  FORMAT_UNSUPPORTED: 'FORMAT_UNSUPPORTED',
  INPUT_MISSING: 'INPUT_MISSING',
  INPUT_INVALID: 'INPUT_INVALID',
  INPUT_BASE64: 'INPUT_BASE64',
  DEMUX_TOO_SMALL: 'DEMUX_TOO_SMALL',
  DEMUX_PARSE: 'DEMUX_PARSE',
  DEMUX_SAMPLE_TABLE: 'DEMUX_SAMPLE_TABLE',
  DECODE_NO_OUTPUT: 'DECODE_NO_OUTPUT',
  DECODE_TIMEOUT: 'DECODE_TIMEOUT',
  DECODE_CORRUPT_FRAME: 'DECODE_CORRUPT_FRAME',

  /* ---- filters / overlays ---- */
  FILTER_UNKNOWN: 'FILTER_UNKNOWN',
  FILTER_INVALID: 'FILTER_INVALID',
  FILTER_ASYNC: 'FILTER_ASYNC',
  OVERLAY_UNKNOWN: 'OVERLAY_UNKNOWN',
  LAYOUT_EMPTY: 'LAYOUT_EMPTY',
  QUANT_FAILED: 'QUANT_FAILED',
  QUANT_PALETTE_TOO_SMALL: 'QUANT_PALETTE_TOO_SMALL',

  /* ---- networking / files ---- */
  NET_FETCH_FAILED: 'NET_FETCH_FAILED',
  NET_HTTP_STATUS: 'NET_HTTP_STATUS',
  NET_RANGE_UNSUPPORTED: 'NET_RANGE_UNSUPPORTED',
  FS_UNSUPPORTED: 'FS_UNSUPPORTED',
  ENV_NO_FETCH: 'ENV_NO_FETCH',

  /* ---- unknown ---- */
  UNKNOWN: 'UNKNOWN',
});

/** HTTP-ish severity so UIs can pick colour/toast style without a table. */
export const ErrorSeverity = Object.freeze({
  INFO: 'info',
  WARN: 'warn',
  ERROR: 'error',
  FATAL: 'fatal',
});

/** Maps codes → human hints. Keys are `code` or `code:contextSubtag`. */
const HINTS = {
  [ErrorCode.NO_WORKERS]:
    'Web Workers unavailable (shared worker blocked, file:// origin, or CSP worker-src). GIFX falls back to the main thread — expect 4-8x slower conversions. Serve the page over http(s) and allow `worker-src blob:`.',
  [ErrorCode.NO_OFFSCREEN_CANVAS]:
    'OffscreenCanvas missing: use the `dom` rasterizer backend (document.createElement("canvas")).',
  [ErrorCode.NO_VIDEO_DECODER]:
    'WebCodecs VideoDecoder missing (Safari <16.4, Firefox <133). GIFX automatically falls back to the <video>+requestVideoFrameCallback decoder; it is slower and cannot decode some HEVC/10-bit streams.',
  [ErrorCode.NO_COMPRESSION_STREAM]:
    'CompressionStream missing, APNG will store uncompressed (large). Upgrade the browser or set output:{format:"gif"}.',
  [ErrorCode.INPUT_UNSUPPORTED_CODEC]:
    'The container parsed but the codec is not decodable in this browser. Transcode to H.264/AVC or VP8/VP9 (mp4/webm) first — browsers refuse HEVC in MP4 outside Safari/Firefox-with-ffmpeg.',
  [ErrorCode.INPUT_ENCRYPTED]:
    'The MP4 has a `encv`/`enca` sample entry (FastStart DRM / FairPlay). DRM cannot be stripped in the browser; supply a clear-source file.',
  [ErrorCode.INPUT_CORRUPT]:
    'Truncated or malformed container. Try `probe()` on the file for the exact offset, or repair with `ffmpeg -i in.mp4 -c copy out.mp4`.',
  [ErrorCode.INPUT_NO_VIDEO_TRACK]:
    'Audio-only or subtitle-only file detected. GIFX needs at least one video track.',
  [ErrorCode.INPUT_NO_FRAMES]:
    'Zero frames produced: trim range is empty, start >= end, or the decoder refused every frame. Check `trim`/`fps`.',
  [ErrorCode.CONFIG_OUT_OF_RANGE]:
    'A value was clamped out of its allowed range. See the `path` field for the offending key and `docs/config.md` for limits.',
  [ErrorCode.OUT_OF_MEMORY]:
    'Typed-array arena exhausted. Lower `maxFramesInFlight`, `scale`, or set `output.streaming: true` to write to OPFS instead of RAM.',
  [ErrorCode.WORKER_DIED]:
    'A kernel worker was killed (OOM or crash). The job was retried on the main thread; set `workers:{crashBudget:0}` to fail fast instead.',
  [ErrorCode.ABORTED]:
    'The conversion was cancelled through AbortSignal. This is not an error unless it was unexpected.',
  [ErrorCode.TARGET_SIZE_UNREACHABLE]:
    'Target size could not be met even with the most aggressive allowed settings. Widen `search.range` (lower fps / fewer colors / smaller width) or raise the byte budget.',
  [ErrorCode.GIF_TOO_MANY_FRAMES]:
    'Source GIF exceeds limits.maxFrames; raise it or pass `downsample`.',
  [ErrorCode.INSECURE_CONTEXT]:
    'OPFS, SharedArrayBuffer and clipboard write all require a secure context (https or localhost).',
  [ErrorCode.NO_OPFS]:
    'Origin Private File System unavailable; falling back to in-memory output buffers.',
};

/**
 * Base class for every error the kernel throws.
 *
 * @example
 * try { await gifx.convert(...) }
 * catch (e) {
 *   if (e instanceof GifxError && e.retryable) retry(e);
 *   console.error(e.code, e.path, e.hint);
 * }
 */
export class GifxError extends Error {
  /**
   * @param {ErrorCode|string} code
   * @param {string} message
   * @param {object} [details]
   * @param {string} [details.path] config key / frame index / byte offset the error refers to
   * @param {object} [details.data] arbitrary machine payload
   * @param {boolean} [details.retryable]
   * @param {Error} [details.cause]
   * @param {string} [details.severity]
   */
  constructor(code, message, details = {}) {
    // Two accepted spellings, because both read naturally at a call site:
    //   new GifxError(ErrorCode.X, 'msg', { data })          // 3-arg legacy
    //   new GifxError('msg', { code: ErrorCode.X, data })     // options form
    if (arguments.length === 2 && typeof code === 'string' && message && typeof message === 'object') {
      const opts = /** @type {object} */ (message);
      details = opts;
      message = code; // the string we were handed first IS the message
      code = /** @type {any} */ (opts).code;
    } else if (code && typeof code === 'object' && arguments.length <= 2) {
      details = /** @type {object} */ (code);
      message = /** @type {any} */ (details).message;
      code = /** @type {any} */ (details).code;
    } else if (message === undefined && typeof code === 'string') {
      // `new GifxError(ErrorCode.X)` / `new GifxError('just a message')`
      message = /^[A-Z0-9_]+$/.test(code) ? undefined : code;
    }
    super(message != null ? String(message) : code != null ? String(code) : 'Error');
    this.name = 'GifxError';
    this.code = code || ErrorCode.UNKNOWN;
    /** Config path, frame index, byte offset… whatever localises the failure. */
    this.path = details.path;
    this.data = details.data || details.details || {};
    this.retryable = details.retryable === undefined ? defaultRetryable(this.code) : !!details.retryable;
    this.severity = details.severity || defaultSeverity(this.code);
    this.hint = details.hint || details.suggestion || HINTS[this.code] || null;
    this.timestamp = Date.now();
    if (details.cause) this.cause = details.cause;
    if (Error.captureStackTrace) Error.captureStackTrace(this, GifxError);
  }

  /** Serializable form, handy for `postMessage` / logs / test snapshots. */
  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      path: this.path,
      severity: this.severity,
      retryable: this.retryable,
      hint: this.hint,
      data: this.data,
      stack: this.stack,
    };
  }

  static isGifxError(e) {
    return !!e && (e instanceof GifxError || e.name === 'GifxError');
  }

  /** Wrap an unknown throwable into a GifxError without losing the cause. */
  static from(err, code = ErrorCode.INTERNAL, message, details = {}) {
    if (err instanceof GifxError && !code) return err;
    const g = new GifxError(code, message || (err && err.message) || 'Unknown error', {
      ...details,
      cause: err,
    });
    return g;
  }
}

function defaultRetryable(code) {
  switch (code) {
    case ErrorCode.WORKER_DIED:
    case ErrorCode.WORKER_TIMEOUT:
    case ErrorCode.INPUT_TIMEOUT:
    case ErrorCode.INPUT_DECODE_FAILED:
    case ErrorCode.INPUT_SEEK_FAILED:
    case ErrorCode.BACKPRESSURE_TIMEOUT:
    case ErrorCode.TASK_FAILED:
      return true;
    default:
      return false;
  }
}

function defaultSeverity(code) {
  switch (code) {
    case ErrorCode.ABORTED:
    case ErrorCode.CANCELLED:
      return ErrorSeverity.INFO;
    case ErrorCode.CONFIG_UNKNOWN_KEY:
    case ErrorCode.CONFIG_OUT_OF_RANGE:
    case ErrorCode.TARGET_SIZE_UNREACHABLE:
    case ErrorCode.NO_SHARED_ARRAY_BUFFER:
      return ErrorSeverity.WARN;
    case ErrorCode.OUT_OF_MEMORY:
    case ErrorCode.WORKER_DIED:
    case ErrorCode.INTERNAL:
      return ErrorSeverity.FATAL;
    default:
      return ErrorSeverity.ERROR;
  }
}

/**
 * Aggregate error for jobs that partially failed (e.g. 3 of 400 frames were
 * undecodable but the GIF was still produced).
 */
export class GifxAggregateError extends GifxError {
  constructor(message, errors = [], details = {}) {
    super(details.code || ErrorCode.TASK_FAILED, message, details);
    this.name = 'GifxAggregateError';
    /** @type {GifxError[]} */
    this.errors = errors;
  }

  get partial() {
    return this.errors.length > 0 && this.errors.length < (this.data.expected || Infinity);
  }

  toJSON() {
    return { ...super.toJSON(), errors: this.errors.map((e) => (e && e.toJSON ? e.toJSON() : String(e))) };
  }
}

/** Hints for the demux/filter family (kept next to the codes they describe). */
Object.assign(HINTS, {
  [ErrorCode.FORMAT_UNSUPPORTED]:
    'Sniffing the first bytes failed, so the extension was wrong too. `probe()` reports the exact header it saw; convert the file to MP4(H.264) or WebM(VP8/VP9) if it is something exotic.',
  [ErrorCode.DEMUX_SAMPLE_TABLE]:
    'The container header is present but its sample table does not resolve (moov at the end of a partially downloaded file, or a muxer bug). Re-download the file completely, or let GIFX fall back to the <video> decoder with `decode:{prefer:\"element\"}`.',
  [ErrorCode.DEMUX_TOO_SMALL]:
    'The file is smaller than a valid container header. It was probably truncated by an aborted download or a bad Blob slice.',
  [ErrorCode.DECODE_CORRUPT_FRAME]:
    'The decoder emitted a damaged frame. GIFX keeps going and reports it under `result.warnings` — only raise `decode.strict` if you need to fail instead.',
  [ErrorCode.FILTER_UNKNOWN]:
    'Filter names are lowercase and match FFmpeg where possible (`scale`, `crop`, `fps`, `eq`, `unsharp`, `hue`). See `GIFX.listFilters()` for the full list.',
  [ErrorCode.OVERLAY_UNKNOWN]:
    'Overlay types: text, caption, watermark, image, rect, ellipse, line, arrow, frame, highlight, redact, progress, counter, timestamp, pip.',
  [ErrorCode.NO_SHARED_ARRAY_BUFFER]:
    'SharedArrayBuffer needs cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`). GIFX transfers buffers between workers instead, which is a little slower but needs no headers.',
});

/**
 * Normalize anything thrown (including a DOMException or a string) into a
 * GifxError with a useful code, preserving `cause` for stack traces.
 */
export function wrapError(err, opts = {}) {
  if (err instanceof GifxError && (!opts.code || opts.code === err.code)) return err;
  const name = err && err.name;
  let code = opts.code;
  if (!code) {
    if (name === 'AbortError') code = ErrorCode.ABORTED;
    else if (name === 'NotSupportedError' || name === 'NotSupported') code = ErrorCode.ENV_UNSUPPORTED;
    else if (name === 'DataError' || name === 'EncodingError') code = ErrorCode.INPUT_CORRUPT;
    else if (name === 'QuotaExceededError') code = ErrorCode.OUT_OF_MEMORY;
    else if (name === 'RangeError' && /memory|ArrayBuffer/i.test(String(err.message))) code = ErrorCode.OUT_OF_MEMORY;
    else if (name === 'TypeError' && /fetch|network|Failed to fetch/i.test(String(err.message))) code = ErrorCode.NET_FETCH_FAILED;
    else if (name === 'SecurityError') code = ErrorCode.INSECURE_CONTEXT;
    else code = err && err.code ? err.code : ErrorCode.INTERNAL;
  }
  const message = opts.message || (err && err.message) || String(err);
  return new GifxError(code, message, {
    cause: err,
    path: opts.path,
    data: opts.data || (err && err.details) || {},
    retryable: opts.retryable,
    hint: opts.hint,
    severity: opts.severity,
  });
}

/** Convenience factory used all over the kernel. */
export function fail(code, message, details) {
  throw new GifxError(code, message, details);
}

export function assert(cond, code, message, details) {
  if (!cond) throw new GifxError(code, message, details);
}

/**
 * Turn a plain object error (as received from `worker.onerror`'s structured
 * clone) back into a GifxError-ish thing.
 */
export function reviveError(payload) {
  if (!payload) return null;
  if (payload instanceof Error) return payload;
  const e = new GifxError(payload.code || ErrorCode.UNKNOWN, payload.message || 'Worker error', payload);
  e.stack = payload.stack || e.stack;
  return e;
}

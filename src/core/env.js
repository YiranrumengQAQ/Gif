/**
 * GIFX Kernel — environment / capability probing.
 *
 * One place that answers "what can this browser do?" so every stage can
 * gracefully degrade instead of throwing. All checks are cached and side-effect
 * free; `probe()` is also exposed publicly so UIs can render a support matrix.
 *
 * @module core/env
 */
import { GifxError, ErrorCode } from './errors.js';

const CACHE = { value: null };

/** @returns {boolean} */
function supportsStructuredCloneTransfer() {
  try {
    const c = new MessageChannel();
    c.port1.postMessage(new Uint8Array(1), []);
    c.port1.close();
    c.port2.close();
    return true;
  } catch {
    return false;
  }
}

/**
 * Detect everything. Safe in Node (returns a mostly-false environment) so the
 * pure kernels can be unit tested there.
 */
export function probe(force = false) {
  if (CACHE.value && !force) return CACHE.value;
  const g = globalThis;
  const hasDoc = typeof g.document !== 'undefined' && !!g.document.createElement;
  const isWorker = typeof g.importScripts === 'function' || (typeof g.self !== 'undefined' && g.self instanceof Object && !!g.onmessage !== undefined && typeof g.postMessage === 'function' && !hasDoc);

  const env = {
    /* --- surfaces --- */
    isWorker,
    isBrowser: hasDoc,
    blob: typeof g.Blob === 'function',
    url: typeof g.URL === 'object' && typeof g.URL.createObjectURL === 'function',
    isSecureContext: g.isSecureContext === true || (!hasDoc && g.location?.protocol === 'file:'),
    protocol: g.location?.protocol || (hasDoc ? 'about:' : 'node:'),
    origin: g.location?.origin || 'null',

    /* --- concurrency --- */
    workers: typeof g.Worker === 'function',
    workerBlobUrls: supportsBlobWorker(),
    sharedArrayBuffer: typeof g.SharedArrayBuffer === 'function',
    atomics: typeof g.Atomics === 'object' && !!g.Atomics.waitAsync,
    hardwareConcurrency: Math.max(1, g.navigator?.hardwareConcurrency || 1),
    deviceMemory: g.navigator?.deviceMemory || null,

    /* --- rasterization --- */
    offscreenCanvas: typeof g.OffscreenCanvas === 'function' && supportsOffscreen2D(),
    createImageBitmap: typeof g.createImageBitmap === 'function',
    imageBitmapOptions: supportsBitmapOptions(),
    imageData: typeof g.ImageData === 'function',
    webgl2: supportsWebGL2(),

    /* --- codecs --- */
    videoDecoder: typeof g.VideoDecoder === 'function',
    videoEncoder: typeof g.VideoEncoder === 'function',
    webCodecsConfig: typeof g.VideoDecoder === 'function' && 'configure' in g.VideoDecoder.prototype,
    requestVideoFrameCallback: hasDoc && 'requestVideoFrameCallback' in (g.HTMLVideoElement?.prototype || {}),
    mediaRecorder: typeof g.MediaRecorder === 'function',
    captureStream: hasDoc && typeof g.HTMLCanvasElement === 'function' && 'captureStream' in g.HTMLCanvasElement.prototype,

    /* --- codecs by mime --- */
    codecs: probeCodecs(),

    /* --- compression / storage --- */
    compressionStream: typeof g.CompressionStream === 'function',
    decompressionStream: typeof g.DecompressionStream === 'function',
    opfs: !!g.navigator?.storageManager?.getDirectory,
    indexedDB: typeof g.indexedDB === 'object' && !!g.indexedDB,
    showSaveFilePicker: typeof g.showSaveFilePicker === 'function',
    clipboardWrite: typeof g.navigator?.clipboard?.write === 'function',

    /* --- misc --- */
    structuredClone: typeof g.structuredClone === 'function',
    transferables: hasDoc || isWorker ? supportsStructuredCloneTransfer() : false,
    offscreenTransfer: hasDoc || isWorker,
    wasm: typeof g.WebAssembly === 'object',
    bigInteger: typeof g.BigInt === 'function',
    queueMicrotask: typeof g.queueMicrotask === 'function',
    schedulerYield: typeof g.scheduler?.yield === 'function',
    prefersReducedMotion: !!g.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches,
    dark: !!g.matchMedia?.('(prefers-color-scheme: dark)')?.matches,

    /* --- limits --- */
    maxCanvasArea: probeMaxCanvas(),
    userAgent: g.navigator?.userAgent || 'node',
    platform: detectPlatform(g.navigator?.userAgent || ''),
    engine: detectEngine(g.navigator?.userAgent || ''),
  };
  env.suggestedWorkers = suggestWorkers(env);
  env.summary = summarize(env);
  return (CACHE.value = env);
}

function supportsBlobWorker() {
  try {
    return typeof Blob === 'function' && typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function';
  } catch {
    return false;
  }
}

function supportsOffscreen2D() {
  try {
    const c = new OffscreenCanvas(2, 2);
    const ctx = c.getContext('2d', { willReadFrequently: false, alpha: true });
    if (!ctx) return false;
    const ok = typeof ctx.drawImage === 'function' && typeof c.convertToBlob === 'function';
    return ok;
  } catch {
    return false;
  }
}

function supportsWebGL2() {
  try {
    if (typeof OffscreenCanvas === 'function') {
      const gl = new OffscreenCanvas(2, 2).getContext('webgl2');
      if (gl) return true;
    }
    if (typeof document !== 'undefined' && document.createElement) {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (gl) {
        gl.getExtension('WEBGL_lose_context')?.loseContext();
        return true;
      }
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * `createImageBitmap(src, {resizeWidth, resizeHeight, resizeQuality, cropX…})`
 * does downscale+crop on the compositor in one shot — orders of magnitude
 * faster than a canvas drawImage for video frames. Safari shipped the crop
 * options later than Chrome, so feature-detect instead of version-sniffing.
 */
function supportsBitmapOptions() {
  if (typeof createImageBitmap !== 'function') return false;
  try {
    // Synchronous capability sniff is impossible; infer from Chrome/Safari
    // support windows. `probe()` consumers that need certainty call
    // `testBitmapOptions()` with a real source.
    const ua = navigator?.userAgent || '';
    if (/Version\/1[0-5]\./.test(ua)) return false; // Safari <=15
    if (/Firefox\/(1[0-2][0-9]|[0-9][0-9])\./.test(ua)) return /Firefox\/1[3-9][0-9]/.test(ua) ? true : false;
    return true;
  } catch {
    return true;
  }
}

/** Runtime verification of createImageBitmap options using a real bitmap. */
export async function testBitmapOptions() {
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') return { resize: false, crop: false };
  try {
    const src = new OffscreenCanvas(8, 8);
    const c = src.getContext('2d');
    c.fillStyle = 'red';
    c.fillRect(0, 0, 8, 8);
    let resize = false;
    let crop = false;
    try {
      const bmp = await createImageBitmap(src, { resizeWidth: 4, resizeHeight: 4, resizeQuality: 'high' });
      resize = bmp.width === 4 && bmp.height === 4;
      bmp.close?.();
    } catch {
      /* fall through */
    }
    try {
      const bmp = await createImageBitmap(src, { cropX: 2, cropY: 2, cropWidth: 4, cropHeight: 4 });
      crop = bmp.width === 4 && bmp.height === 4;
      bmp.close?.();
    } catch {
      /* fall through */
    }
    return { resize, crop };
  } catch {
    return { resize: false, crop: false };
  }
}

const MIME_CANDIDATES = [
  ['video/mp4; codecs="avc1.42E01E"', 'h264'],
  ['video/mp4; codecs="avc1.640028"', 'h264-high'],
  ['video/mp4; codecs="hev1.1.6.L93.B0"', 'hevc'],
  ['video/webm; codecs="vp8"', 'vp8'],
  ['video/webm; codecs="vp9"', 'vp9'],
  ['video/webm; codecs="vp09.00.10.08"', 'vp9-profile0'],
  ['video/webm; codecs="av01.0.05M.08"', 'av1'],
  ['video/x-matroska; codecs="theora, vorbis"', 'theora'],
  ['video/mp4; codecs="avc1.4D001E, mp4a.40.2"', 'h264-aac'],
];

function probeCodecs() {
  const out = {};
  try {
    const v = typeof document !== 'undefined' && document.createElement ? document.createElement('video') : null;
    if (!v) return out;
    for (const [mime, name] of MIME_CANDIDATES) {
      try {
        out[name] = v.canPlayType(mime) === 'probably' ? 'probably' : v.canPlayType(mime) === 'maybe' ? 'maybe' : 'no';
      } catch {
        out[name] = 'no';
      }
    }
    // animated webp / apng support: probe by decoding a 1x1 file lazily elsewhere
  } catch {
    /* ignore */
  }
  return out;
}

/**
 * Browsers silently fail canvases above ~16384px per side / ~268MP total.
 * We clamp requested sizes to this so "scale to 12000px" produces a clear
 * error instead of a blank canvas.
 */
function probeMaxCanvas() {
  if (typeof document === 'undefined' || !document.createElement) return { side: 32767, area: 1 << 30, bytes: 1 << 30 };
  try {
    const sides = [32767, 16384, 8192];
    for (const side of sides) {
      const c = document.createElement('canvas');
      c.width = side;
      c.height = 1;
      if (c.getContext('2d')) {
        return { side, area: side * side, bytes: side * side * 4 };
      }
    }
    return { side: 4096, area: 4096 * 4096, bytes: 4096 * 4096 * 4 };
  } catch {
    return { side: 4096, area: 4096 * 4096, bytes: 0 };
  }
}

export function suggestWorkers(env = CACHE.value || probe()) {
  const n = env.hardwareConcurrency || 2;
  if (!env.workers) return 0;
  // Leave one core for the UI thread + compositor; cap to avoid thread
  // thrash on big-Macs where more workers just contend on memory bandwidth.
  return Math.max(1, Math.min(n - 1, 12));
}

function detectPlatform(ua) {
  if (/iPhone|iPad|iPod/.test(ua)) return 'ios';
  if (/Android/.test(ua)) return 'android';
  if (/Mac OS X|Macintosh/.test(ua)) return 'mac';
  if (/Windows/.test(ua)) return 'windows';
  if (/Linux/.test(ua)) return 'linux';
  if (!ua) return 'node';
  return 'other';
}
function detectEngine(ua) {
  if (/Firefox\//.test(ua)) return 'gecko';
  if (/Edg\//.test(ua)) return 'edgehtml?chromium';
  if (/Version\/[\d.]+.*Safari/.test(ua) && /Apple/.test(navigator?.vendor || 'Apple')) return 'webkit';
  if (/Chrome\//.test(ua)) return 'chromium';
  if (!ua) return 'node';
  return 'unknown';
}

function summarize(env) {
  const fast = env.videoDecoder && env.offscreenCanvas && env.workers;
  const mid = env.offscreenCanvas && env.workers;
  return {
    tier: fast ? 'webcodecs' : mid ? 'html-video' : 'main-thread',
    label: fast ? 'Fast path: WebCodecs + workers + OffscreenCanvas' : mid ? 'Standard path: <video> decode + workers' : 'Compatibility path: main-thread decode',
    parallel: env.suggestedWorkers,
  };
}

/** Public one-liner for UIs / bug reports. */
export function capabilities() {
  const env = probe();
  const rows = [];
  const push = (k, v, note = '') => rows.push({ key: k, value: v, note });
  push('WebCodecs VideoDecoder', env.videoDecoder, 'hardware-assisted decode, exact frame access');
  push('WebCodecs VideoEncoder', env.videoEncoder, 'used for optional mp4/webm re-encode');
  push('OffscreenCanvas', env.offscreenCanvas, 'rasterize inside workers');
  push('createImageBitmap options', env.imageBitmapOptions, 'GPU-side crop+resize');
  push('Workers', env.workers, `${env.suggestedWorkers} lanes recommended`);
  push('SharedArrayBuffer', env.sharedArrayBuffer, 'needs COOP/COEP; enables zero-copy stats');
  push('CompressionStream', env.compressionStream, 'zlib for APNG/PNG paths');
  push('DecompressionStream', env.decompressionStream, 'inflate for GIF/PNG import & ZIP sequences');
  push('requestVideoFrameCallback', env.requestVideoFrameCallback, 'frame-accurate <video> sampling');
  push('OPFS', env.opfs, 'stream huge outputs without holding them in RAM');
  push('showSaveFilePicker', env.showSaveFilePicker, 'save dialog with real file handle');
  push('ClipboardItem', env.clipboardWrite, 'copy GIF straight to clipboard');
  push('MediaRecorder', env.mediaRecorder, 'fallback webm writer');
  push('WebGL2', env.webgl2, 'optional GPU filters');
  push('canvas max side', env.maxCanvasArea.side, 'hard browser ceiling');
  push('deviceMemory', env.deviceMemory ? `${env.deviceMemory} GB` : 'unknown', 'drives arena budget');
  return { env, rows };
}

/**
 * Throws a descriptive GifxError when a required capability is missing.
 * @param {string} feature key of the object returned by {@link probe}
 * @param {string} [why] shown in the message (e.g. "GPU filters")
 */
export function requireFeature(feature, why = '') {
  const env = probe();
  if (env[feature]) return true;
  throw new GifxError(ErrorCode[`NO_${String(feature).toUpperCase()}`] || ErrorCode.ENV_UNSUPPORTED, `${feature} is unavailable${why ? ` (${why})` : ''} in this environment`, {
    path: `env.${feature}`,
    data: { tier: env.summary.tier },
    hint: HINTS[feature],
  });
}
const HINTS = {
  workers: 'Serve over http(s) with `worker-src blob:` allowed, or import the UMD build which inlines workers.',
  videoDecoder: 'Upgrade to Chrome 94+/Edge 94+/Safari 16.4+/Firefox 133+ for the fast path, or set decode:{backend:"html-video"}.',
  offscreenCanvas: 'Workers fall back to sending ImageData back to the main thread for rasterizing.',
  compressionStream: 'Use Chrome 80+/Safari 16.4+; otherwise APNG is stored uncompressed.',
  opfs: 'Requires https or localhost. Output stays in memory; expect high RAM use on long jobs.',
};

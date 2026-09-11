/**
 * GIFX Kernel — container probing.
 *
 * Sniffs the *bytes*, never just the extension: browsers happily hand you a
 * `.mp4` that is really WebM (screen recorders) and a `.mov` that is a QuickTime
 * stream we can't decode. Everything downstream (decoder selection, whether we
 * can be frame-accurate, whether a re-mux is legal) keys off this result.
 *
 * `parseMp4` / `parseWebm` / `parseRiff` are pure functions over a `Uint8Array`
 * and are defensive to a fault: every read is bounds-checked, truncated files
 * degrade to `partial: true` instead of throwing.
 *
 * @module media/probe
 */
import { GifxError, wrapError, ErrorCode } from '../core/errors.js';
import { parseMp4 } from './mp4.js';
import { parseWebm } from './webm.js';
import { parseRiff } from './riff.js';

/** Byte-length we read for sniffing. 64 KiB covers moov-at-end files' ftyp plus a stsd. */
export const SNIFF_BYTES = 64 * 1024;

export const FORMATS = {
  mp4: { ext: ['mp4', 'm4v', 'mov', 'm4a', 'f4v'], mime: ['video/mp4', 'video/quicktime', 'audio/mp4'], decodable: true },
  webm: { ext: ['webm'], mime: ['video/webm'], decodable: true },
  mkv: { ext: ['mkv', 'mka'], mime: ['video/x-matroska'], decodable: true },
  avi: { ext: ['avi'], mime: ['video/x-msvideo'], decodable: 'partial' },
  gif: { ext: ['gif'], mime: ['image/gif'], decodable: true },
  png: { ext: ['png', 'apng'], mime: ['image/png'], decodable: true },
  jpeg: { ext: ['jpg', 'jpeg', 'jpe'], mime: ['image/jpeg'], decodable: true },
  webp: { ext: ['webp'], mime: ['image/webp'], decodable: true },
  avif: { ext: ['avif'], mime: ['image/avif'], decodable: true },
  bmp: { ext: ['bmp', 'dib'], mime: ['image/bmp'], decodable: true },
  ico: { ext: ['ico', 'cur'], mime: ['image/vnd.microsoft.icon'], decodable: true },
  tiff: { ext: ['tif', 'tiff'], mime: ['image/tiff'], decodable: true },
  svg: { ext: ['svg', 'svgz'], mime: ['image/svg+xml'], decodable: true },
  pdf: { ext: ['pdf'], mime: ['application/pdf'], decodable: false },
  zip: { ext: ['zip'], mime: ['application/zip'], decodable: 'partial' },
  flac: { ext: ['flac'], mime: ['audio/flac'], decodable: false },
  mp3: { ext: ['mp3'], mime: ['audio/mpeg'], decodable: false },
  m3u8: { ext: ['m3u8'], mime: ['application/vnd.apple.mpegurl'], decodable: false },
  tg: { ext: ['tg'], mime: ['video/telegram'], decodable: false },
};

/**
 * Identify a file from its first bytes.
 * @param {Uint8Array|ArrayBuffer} bytes
 * @param {string} [hintName] filename/URL for extension+MIME tie-breaking
 * @param {string} [hintMime]
 * @returns {{format:string, mime:string, container:string, confidence:number, note?:string, ext:string}}
 */
export function identify(bytes, hintName = '', hintMime = '') {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || []);
  const head = u8.subarray(0, Math.min(u8.length, 64));
  const ascii = (o, n) => String.fromCharCode(...head.subarray(o, o + n));
  const at = (o, s) => {
    for (let i = 0; i < s.length; i++) if (head[o + i] !== s.charCodeAt(i)) return false;
    return head.length >= o + s.length;
  };
  const ext = /\.([a-z0-9]{2,5})(?:$|[?#])/i.exec(String(hintName || ''))?.[1]?.toLowerCase() || '';
  const guess = () => {
    for (const [fmt, info] of Object.entries(FORMATS)) if (info.ext.includes(ext)) return { format: fmt, ...info };
    return null;
  };
  // --- video containers ---
  if (at(4, 'ftyp')) {
    const brand = ascii(8, 4);
    const fmt = brand === 'qt  ' ? 'mp4' : 'mp4';
    const mime = brand === 'qt  ' ? 'video/quicktime' : 'video/mp4';
    return { format: fmt, container: fmt, mime, brand, ext, confidence: 0.99, compatible: readMp4Brands(u8) };
  }
  if (at(0, '\x1aE\xdf\xa3')) return { format: 'webm', container: 'matroska', mime: 'video/webm', ext, confidence: 0.99 };
  if (at(0, 'RIFF')) {
    const w = ascii(8, 4);
    if (w === 'AVI ') return { format: 'avi', container: 'riff', mime: 'video/x-msvideo', ext, confidence: 0.95 };
    if (w === 'WEBP') return { format: 'webp', container: 'riff', mime: 'image/webp', ext, confidence: 0.98 };
    if (w === 'ANIM') return { format: 'aiff', container: 'aiff', mime: 'audio/aiff', ext, confidence: 0.8 };
    if (w === 'CDX ') return { format: 'riff-cdxl', container: 'riff', mime: 'application/octet-stream', ext, confidence: 0.6 };
    return { format: 'riff', container: 'riff', mime: 'application/octet-stream', ext, confidence: 0.6, note: `unsupported RIFF fourcc "${w}"` };
  }
  if (at(0, 'FTYP') || at(0, 'FLV') || (head[0] === 0x46 && head[1] === 0x4c && head[2] === 0x06)) {
    return { format: 'flv', container: 'flv', mime: 'video/x-flv', ext, confidence: 0.9, note: 'FLV must be remuxed; browsers cannot decode it' };
  }
  if (at(0, 'OggS')) return { format: 'ogg', container: 'ogg', mime: 'video/ogg', ext, confidence: 0.95 };
  // --- images ---
  if (at(0, 'GIF87a') || at(0, 'GIF89a')) return { format: 'gif', container: 'gif', mime: 'image/gif', ext, version: ascii(0, 6), confidence: 1 };
  if (at(0, '\x89PNG\r\n\x1a\n')) return { format: 'png', container: 'png', mime: 'image/png', ext, confidence: 1, animated: hasApngChunks(u8) };
  if (head[0] === 0xff && head[1] === 0xd8) return { format: 'jpeg', container: 'jpeg', mime: 'image/jpeg', ext, confidence: 1 };
  if (at(0, 'BM')) return { format: 'bmp', container: 'bmp', mime: 'image/bmp', ext, confidence: 0.95 };
  if (at(0, 'II*\x00') || at(0, 'MM\x00*')) return { format: 'tiff', container: 'tiff', mime: 'image/tiff', ext, confidence: 0.95 };
  if (head[0] === 0x00 && head[1] === 0x00 && head[2] === 0x01 && head[3] === 0x00) return { format: 'ico', container: 'ico', mime: 'image/x-icon', ext, confidence: 0.9 };
  if (at(0, 'fLaC')) return { format: 'flac', container: 'flac', mime: 'audio/flac', ext, confidence: 0.95 };
  if (head[0] === 0x49 && (head[1] === 0x44 || head[1] === 0x33)) return { format: 'mp3', container: 'mp3', mime: 'audio/mpeg', ext, confidence: 0.9 };
  if (/^\s*(<\?xml[\s\S]{0,400}?)?<svg/i.test(String.fromCharCode.apply(null, Array.from(head.slice(0, 512))))) {
    return { format: 'svg', container: 'svg', mime: 'image/svg+xml', ext, confidence: 0.95 };
  }
  if (at(0, '%PDF')) return { format: 'pdf', container: 'pdf', mime: 'application/pdf', ext, confidence: 1, note: 'PDFs cannot be decoded in-browser without a renderer' };
  if (at(0, 'PK\x03\x04')) return { format: 'zip', container: 'zip', mime: 'application/zip', ext, confidence: 0.98 };
  if (/^#EXTM3U/.test(String.fromCharCode(...head.slice(0, 16)))) return { format: 'm3u8', container: 'hls', mime: 'application/vnd.apple.mpegurl', ext, confidence: 1, note: 'HLS needs a segment demuxer' };
  // --- codecs in raw streams (people do paste these) ---
  if (head[0] === 0x00 && head[1] === 0x00 && head[2] === 0x00 && (head[3] === 0x01 || head[3] === 0x09)) {
    return { format: 'h264-annexb', container: 'es', mime: 'video/mp4', ext, confidence: 0.7, note: 'raw H.264 stream — will be wrapped in a minimal MP4' };
  }
  // fall back to name/MIME
  const g = guess();
  if (g) return { format: g.format, container: g.format, mime: g.mime[0], ext, confidence: 0.4, note: 'identified by extension only' };
  const mg = Object.entries(FORMATS).find(([, info]) => info.mime.some((m) => hintMime && hintMime.startsWith(m.split('/')[0] + '/')));
  if (mg) return { format: mg[0], container: mg[0], mime: hintMime, ext, confidence: 0.3, note: 'identified by MIME type only' };
  const e = new GifxError(`Unrecognized file format (${u8.length} bytes, first bytes: ${hex(head.subarray(0, 8))})`, {
    code: ErrorCode.FORMAT_UNSUPPORTED,
    severity: 'fatal',
    details: { size: u8.length, head: hex(head.subarray(0, 16)) },
    suggestion: 'Supported: MP4/M4V/MOV (H.264/HEVC/AV1), WebM/MKV (VP8/VP9/AV1), AVI, and PNG/JPEG/GIF/WebP/AVIF/BMP/TIFF/SVG stills.',
  });
  throw e;
}

const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join(' ');

function readMp4Brands(u8) {
  const out = [];
  const n = Math.min(u8.length - 16, 64);
  for (let o = 16; o + 4 <= n; o += 4) {
    const b = String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);
    if (/^[\x20-\x7e]{4}$/.test(b)) out.push(b.trim());
  }
  return out;
}

function hasApngChunks(u8) {
  // AC19 chunk appears right after IHDR in an APNG
  const limit = Math.min(u8.length - 8, 64);
  for (let o = 8; o < limit; o += 1) if (u8[o + 4] === 0x61 && u8[o + 5] === 0x63 && u8[o + 6] === 0x54 && u8[o + 7] === 0x6c) return true;
  return false;
}

/**
 * Read the whole file (or a Blob slice) as a Uint8Array, with a size guard.
 * @param {Blob|ArrayBuffer|Uint8Array|string|Response} input
 * @param {object} [opts] `{maxBytes, range:'bytes'|'head'|'tail'}`
 */
export async function readBytes(input, opts = {}) {
  if (!input) throw new GifxError('readBytes: no input', { code: ErrorCode.INPUT_MISSING });
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (typeof input === 'string') {
    if (/^data:/i.test(input)) {
      const b64 = input.slice(input.indexOf(',') + 1);
      const bin = atobSafe(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    if (typeof fetch !== 'function') throw new GifxError('fetch() is unavailable in this environment', { code: ErrorCode.ENV_NO_FETCH });
    const res = await fetch(input, opts.headers ? { headers: opts.headers } : undefined);
    if (!res.ok) throw new GifxError(`fetch failed: ${res.status} ${res.statusText}`, { code: ErrorCode.NET_HTTP_STATUS, retryable: res.status >= 500, details: { url: input, status: res.status } });
    return new Uint8Array(await res.arrayBuffer());
  }
  if (typeof Response !== 'undefined' && input instanceof Response) return new Uint8Array(await input.arrayBuffer());
  if (typeof Blob !== 'undefined' && input instanceof Blob) {
    if (opts.maxBytes && input.size > opts.maxBytes && opts.range === 'head') return new Uint8Array(await input.slice(0, opts.maxBytes).arrayBuffer());
    return new Uint8Array(await input.arrayBuffer());
  }
  if (typeof input.arrayBuffer === 'function') return new Uint8Array(await input.arrayBuffer());
  throw new GifxError(`cannot read bytes from ${Object.prototype.toString.call(input)}`, { code: ErrorCode.INPUT_INVALID });
}

function atobSafe(b64) {
  const clean = b64.replace(/\s/g, '');
  try {
    return atob(clean);
  } catch (e) {
    // tolerate URL-safe base64 and missing padding (people paste these)
    const fixed = clean.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(clean.length / 4) * 4, '=');
    try {
      return atob(fixed);
    } catch (e2) {
      throw wrapError(e2, { code: ErrorCode.INPUT_BASE64, message: 'invalid base64 data URL' });
    }
  }
}

/**
 * Full probe: format + container metadata + decodability verdict.
 * @returns {Promise<{format:string,mime:string,width?:number,height?:number,durationMs?:number,
 *   fps?:number,frameCount?:number,codec?:string,hasAlpha?:boolean,loops?:number,
 *   samples?:Array,partial?:boolean,canDecode:boolean,reason?:string,bitrate?:number,size:number}>}
 */
export async function probe(input, opts = {}) {
  const size = typeof input === 'object' && input && typeof input.size === 'number' ? input.size : undefined;
  const bytes = opts.bytes || (await readBytes(input, { maxBytes: opts.headBytes || 0 }));
  const name = (input && (input.name || input.url)) || opts.name || '';
  const mimeHint = (input && input.type) || opts.mime || '';
  const id = identify(bytes, name, mimeHint);
  const out = { format: id.format, mime: id.mime, container: id.container, size: bytes.byteLength, ext: id.ext, confidence: id.confidence, note: id.note };
  if (size != null) out.inputSize = size;
  try {
    switch (id.format) {
      case 'mp4':
        out.container = 'mp4';
        Object.assign(out, summarizeContainer(parseMp4(bytes), 'mp4'));
        break;
      case 'webm':
      case 'mkv':
        out.container = 'webm';
        Object.assign(out, summarizeContainer(parseWebm(bytes), 'webm'));
        break;
      case 'avi':
      case 'riff':
        out.container = 'riff';
        Object.assign(out, summarizeContainer(parseRiff(bytes), 'riff'));
        break;
      case 'gif':
        Object.assign(out, summarizeGif(bytes));
        break;
      case 'png':
        Object.assign(out, summarizePng(bytes));
        break;
      case 'jpeg':
        Object.assign(out, summarizeJpeg(bytes));
        break;
      case 'webp':
        Object.assign(out, summarizeWebp(bytes));
        break;
      case 'bmp':
        Object.assign(out, summarizeBmp(bytes));
        break;
      case 'zip':
        Object.assign(out, { fileCount: countZipEntries(bytes), compressed: true });
        break;
      default:
        break;
    }
  } catch (e) {
    out.partial = true;
    out.parseError = e.message;
  }
  out.canDecode = decideDecodability(out, opts);
  if (!out.canDecode && !out.reason) out.reason = `no in-browser decoder for "${out.format}"`;
  if (out.width && out.height && out.durationMs) {
    out.bitrate = Math.round((out.size * 8) / (out.durationMs / 1000));
    if (out.frameCount) out.fps = out.fps || (out.frameCount * 1000) / out.durationMs;
  }
  return out;
}

function decideDecodability(info, opts) {
  const env = typeof globalThis !== 'undefined' ? globalThis : {};
  const videoFormats = ['mp4', 'webm', 'mkv', 'avi', 'h264-annexb'];
  const stillFormats = ['png', 'jpeg', 'gif', 'webp', 'bmp', 'avif', 'tiff', 'ico', 'svg'];
  if (opts.decode === false) return true; // caller just wanted metadata
  if (stillFormats.includes(info.format)) return true;
  if (videoFormats.includes(info.format)) {
    const hasWebCodecs = typeof env.VideoDecoder === 'function';
    const hasVideo = typeof env.HTMLVideoElement === 'function' || typeof document !== 'undefined';
    return hasWebCodecs || hasVideo;
  }
  if (info.format === 'zip') return true;
  return false;
}

/** Pull the fields we care about out of any container parser's result. */
function summarizeContainer(info, kind) {
  if (!info) return {};
  const out = {
    container: kind,
    durationMs: Math.round((info.durationMs || 0) * 1000) / 1000,
    width: info.width || 0,
    height: info.height || 0,
    frameCount: info.frameCount || 0,
    fps: info.fps ? Math.round(info.fps * 1000) / 1000 : 0,
    codec: info.codec || '',
    codecString: info.codecString || (info.video && info.video.codecString) || '',
    hasAlpha: !!info.hasAlpha,
    partial: !!info.partial,
    rotation: info.rotation || 0,
    fastStart: info.fastStart,
    fragmented: info.fragmented,
    encrypted: info.encrypted,
    muxer: info.muxer || '',
    brands: info.brands,
    doctype: info.doctype,
    indexSource: info.indexSource,
    oddities: info.oddities,
    lacing: info.lacing,
  };
  if (out.fps && out.frameCount && !out.durationMs) out.durationMs = Math.round((out.frameCount * 1000) / out.fps);
  if (out.durationMs && out.frameCount && !out.fps) out.fps = Math.round((out.frameCount * 1000) / out.durationMs);
  if (info.video?.colour || info.video?.color) out.colour = info.video.colour || info.video.color;
  if (info.video?.sampleAspectRatio) out.sampleAspectRatio = info.video.sampleAspectRatio;
  if (info.video?.bitDepth) out.bitDepth = info.video.bitDepth;
  return out;
}

/* --------------------------------------------------------------- image only */

function summarizeGif(u8) {
  if (u8.length < 13) return { partial: true };
  const width = u8[6] | (u8[7] << 8);
  const height = u8[8] | (u8[9] << 8);
  let frames = 0;
  let loops = 0;
  let delayCs = 0;
  for (let i = 0; i < u8.length; i++) {
    if (u8[i] === 0x21 && u8[i + 1] === 0xf9) {
      frames++;
      const d = u8[i + 4] | (u8[i + 5] << 8);
      delayCs += d;
    } else if (u8[i] === 0x21 && u8[i + 1] === 0xff && u8[i + 3] === 0x0b) {
      const net = u8[i + 12] === 0x4e && u8[i + 13] === 0x45 && u8[i + 14] === 0x54;
      if (net) loops = u8[i + 17] | (u8[i + 18] << 8);
    }
  }
  return {
    width,
    height,
    frameCount: frames,
    loops,
    durationMs: delayCs * 10,
    fps: delayCs ? (frames * 1000) / (delayCs * 10) : undefined,
    hasAlpha: !!(u8[10] & 0x80),
    globalColorTable: !!(u8[10] & 0x80),
    colorDepth: (u8[10] & 7) + 1,
  };
}

function summarizePng(u8) {
  if (u8.length < 33) return { partial: true };
  const width = (u8[16] << 24) | (u8[17] << 16) | (u8[18] << 8) | u8[19];
  const height = (u8[20] << 24) | (u8[21] << 16) | (u8[22] << 8) | u8[23];
  const bitDepth = u8[24];
  const colorType = u8[25];
  let acTL = null;
  for (let o = 8; o + 12 <= u8.length; ) {
    const len = (u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3];
    const type = String.fromCharCode(u8[o + 4], u8[o + 5], u8[o + 6], u8[o + 7]);
    if (type === 'acTL') acTL = { numFrames: (u8[o + 8] << 24) | (u8[o + 9] << 16) | (u8[o + 10] << 8) | u8[o + 11], numPlays: (u8[o + 12] << 24) | (u8[o + 13] << 16) | (u8[o + 14] << 8) | u8[o + 15] };
    if (type === 'IEND') break;
    o += 12 + Math.max(0, len);
  }
  return {
    width,
    height,
    bitDepth,
    colorType,
    hasAlpha: colorType === 4 || colorType === 6 || (colorType === 3 && !!findChunk(u8, 'tRNS')),
    interlaced: u8[28] === 1,
    animated: !!acTL,
    frameCount: acTL ? acTL.numFrames : 1,
    loops: acTL ? acTL.numPlays : 1,
  };
}

function findChunk(u8, name) {
  for (let o = 8; o + 12 <= u8.length; ) {
    const len = (u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3];
    const type = String.fromCharCode(u8[o + 4], u8[o + 5], u8[o + 6], u8[o + 7]);
    if (type === name) return u8.subarray(o + 8, o + 8 + len);
    if (type === 'IEND') return null;
    o += 12 + Math.max(0, len);
  }
  return null;
}

function summarizeJpeg(u8) {
  let o = 2;
  let width = 0;
  let height = 0;
  let progressive = false;
  let comments = [];
  let thumb = null;
  while (o + 4 < u8.length) {
    if (u8[o] !== 0xff) {
      o++;
      continue;
    }
    const marker = u8[o + 1];
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      o += 2;
      continue;
    }
    const len = (u8[o + 2] << 8) | u8[o + 3];
    if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2 || marker === 0xc3) {
      height = (u8[o + 5] << 8) | u8[o + 6];
      width = (u8[o + 7] << 8) | u8[o + 8];
      progressive = marker === 0xc2;
      break;
    }
    if (marker === 0xe1) {
      const exif = u8.subarray(o + 4, o + 2 + len);
      if (exif.length > 6 && exif[0] === 0x45 && exif[1] === 0x78) thumb = readExifOrientation(exif);
    }
    if (marker === 0xfe) comments.push(String.fromCharCode(...u8.subarray(o + 4, o + 2 + len)));
    if (marker === 0xda) break;
    o += 2 + len;
  }
  return { width, height, progressive, comments: comments.length ? comments.join(' | ') : undefined, orientation: thumb?.orientation, exif: thumb || undefined };
}

function readExifOrientation(exif) {
  const tiff = exif.subarray(6);
  if (tiff.length < 8) return {};
  const le = tiff[0] === 0x49;
  const u16 = (o) => (le ? tiff[o] | (tiff[o + 1] << 8) : (tiff[o] << 8) | tiff[o + 1]);
  const u32 = (o) => (le ? tiff[o] | (tiff[o + 1] << 8) | (tiff[o + 2] << 16) | (tiff[o + 3] << 24) : ((tiff[o] << 24) | (tiff[o + 1] << 16) | (tiff[o + 2] << 8) | tiff[o + 3]) >>> 0);
  const out = {};
  const ifds = [u32(4)];
  for (const base of ifds) {
    if (base + 2 > tiff.length) continue;
    const n = u16(base);
    for (let i = 0; i < n; i++) {
      const e = base + 2 + i * 12;
      if (e + 12 > tiff.length) break;
      const tag = u16(e);
      if (tag === 0x0112) out.orientation = u16(e + 8);
      else if (tag === 0x011a) out.xDensity = u32(e + 8) / u32(e + 12);
      else if (tag === 0x8825) ifds.push(u32(e + 8));
      else if (tag === 0x8298) out.make = true;
      else if (tag === 0x9003) out.dateTimeTag = true;
    }
  }
  return out;
}

function summarizeWebp(u8) {
  if (u8.length < 30) return { partial: true };
  const fourcc = String.fromCharCode(u8[12], u8[13], u8[14], u8[15]);
  let width = 0;
  let height = 0;
  let hasAlpha = false;
  let animated = false;
  let frames = 1;
  if (fourcc === 'VP8 ') {
    width = ((u8[26] | (u8[27] << 8)) & 0x3fff) || 0;
    height = ((u8[28] | (u8[29] << 8)) & 0x3fff) || 0;
  } else if (fourcc === 'VP8L') {
    const b = (u8[21] << 24) | (u8[22] << 16) | (u8[23] << 8) | u8[24];
    width = (b & 0x3fff) + 1;
    height = ((b >> 14) & 0x3fff) + 1;
    hasAlpha = ((b >> 28) & 1) === 1;
  } else if (fourcc === 'VP8X') {
    hasAlpha = !!(u8[20] & 0x10);
    animated = !!(u8[20] & 0x02);
    width = 1 + (u8[24] | (u8[25] << 8) | (u8[26] << 16));
    height = 1 + (u8[27] | (u8[28] << 8) | (u8[29] << 16));
    if (animated) frames = countWebpFrames(u8);
  }
  return { width, height, hasAlpha, animated, frameCount: frames, fourcc };
}

function countWebpFrames(u8) {
  let count = 0;
  for (let o = 12; o + 8 <= u8.length; ) {
    const tag = String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);
    const size = u8[o + 4] | (u8[o + 5] << 8) | (u8[o + 6] << 16) | (u8[o + 7] << 24);
    if (tag === 'ANMF') count++;
    o += 8 + size + (size & 1);
  }
  return count || 1;
}

function summarizeBmp(u8) {
  const width = (u8[18] | (u8[19] << 8) | (u8[20] << 16) | (u8[21] << 24)) | 0;
  const height = (u8[22] | (u8[23] << 8) | (u8[24] << 16) | (u8[25] << 24)) | 0;
  return { width: Math.abs(width), height: Math.abs(height), bitsPerPixel: u8[28] | (u8[29] << 8), bottomUp: height > 0 };
}

function countZipEntries(u8) {
  let count = 0;
  // scan the central directory by signature; cheaper than inflating the TOC
  for (let o = 0; o + 4 <= u8.length; o++) {
    if (u8[o] === 0x50 && u8[o + 1] === 0x4b && u8[o + 2] === 0x01 && u8[o + 3] === 0x02) count++;
  }
  return count;
}

export { FORMATS as PROBE_FORMATS };

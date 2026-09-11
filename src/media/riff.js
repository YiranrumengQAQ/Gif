/**
 * GIFX Kernel — RIFF parser (AVI video, WebP stills/animation, WAVE metadata).
 *
 * AVI is old enough that its files are frequently *broken*: missing `idx1`,
 * `dwTotalFrames` lying, offsets relative to the wrong base, or an OpenDML
 * (`dvix`/`AVIX`) 64-bit variant. A GIF tool has to survive all of that because
 * screen recorders and phone apps still emit AVI. Strategy:
 *   1. trust `idx1` when present and self-consistent;
 *   2. otherwise scan `movi` for '00dc'/'00db' chunks and *build* the index
 *      (bounded by `maxScanBytes`), which recovers every playable file.
 *
 * @module media/riff
 */
import { GifxError, ErrorCode } from '../core/errors.js';

const FOURCC = (u8, o) => String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);
const u16 = (v, o) => v.getUint16(o, true);
const u32 = (v, o) => v.getUint32(o, true);
const i32 = (v, o) => v.getInt32(o, true);

/**
 * @param {Uint8Array} u8
 * @param {object} [opts] `{scanIfNoIndex:boolean, maxScanBytes, keepFrames:boolean}`
 */
export function parseRiff(u8, opts = {}) {
  const v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.length < 12) throw new GifxError('file too small for RIFF', { code: ErrorCode.DEMUX_TOO_SMALL });
  if (FOURCC(u8, 0) !== 'RIFF' && FOURCC(u8, 0) !== 'RIFX') throw new GifxError('not a RIFF file', { code: ErrorCode.DEMUX_PARSE });
  const little = FOURCC(u8, 0) === 'RIFF';
  const wave32 = (o) => (little ? v.getUint32(o, true) : v.getUint32(o, false));
  const out = {
    format: 'riff',
    form: FOURCC(u8, 8),
    size: wave32(4),
    partial: false,
    streams: [],
    video: null,
    width: 0,
    height: 0,
    fps: 0,
    durationMs: 0,
    frameCount: 0,
    hasAlpha: false,
    chunks: [],
    frames: null,
    indexSource: 'none',
    openDml: false,
    oddities: [],
  };
  const total = Math.min(u8.length - 8, out.size + (out.size % 2));
  walkChunks(u8, v, 12, 12 + total, out, little, 0, opts);
  out.__u8 = u8;
  if (out.form === 'AVI ') summarizeAvi(out, opts, u8);
  else if (out.form === 'WEBP') summarizeWebp(out, v, little);
  else if (out.form === 'WAVE') out.format = 'wave';
  return out;
}

function walkChunks(u8, v, start, end, out, little, depth, opts) {
  let o = start;
  while (o + 8 <= end) {
    const id = FOURCC(u8, o);
    const size = v.getUint32(o + 4, little);
    const payload = o + 8;
    const next = payload + size + (size & 1);
    if (next > end + 8) {
      out.partial = true;
      out.chunks.push({ id, at: o, size, truncated: true });
      break;
    }
    out.chunks.push({ id, at: o, size });
    switch (id) {
      case 'LIST': {
        const listType = FOURCC(u8, payload);
        if (listType === 'hdrl' || listType === 'movi' || listType === 'idx1' || listType === 'rec ') {
          out.chunks[out.chunks.length - 1].list = listType;
          if (listType === 'movi') {
            // `payload` is the 'movi' fourcc — exactly the base AVI `idx1`
            // offsets are relative to (per the RIFF spec / every real muxer).
            out.moviStart = payload;
            out.moviSize = size;
          }
          walkChunks(u8, v, payload + 4, Math.min(end, next), out, little, depth + 1, opts);
        } else if (listType === 'INFO' || listType === 'idit') {
          parseInfoList(u8, v, payload + 4, next, out, little);
        } else walkChunks(u8, v, payload + 4, Math.min(end, next), out, little, depth + 1, opts);
        break;
      }
      case 'avih':
        parseAvih(v, payload, out, little);
        break;
      case 'strh':
        parseStrh(v, payload, size, out, little);
        break;
      case 'strf':
        parseStrf(v, payload, size, out, little);
        break;
      case 'idx1':
        parseIdx1(u8, v, payload, size, out, little);
        break;
      case 'JUNK':
      case 'pad_':
        break;
      case 'dmlh':
        out.openDml = true;
        {
          const junkFrameCount = v.getUint32(payload, little);
          if (junkFrameCount && (!out.video || !out.video.frameCount)) out.oddities.push('OpenDML frame count override');
          if (junkFrameCount) out.openDmlFrameCount = junkFrameCount;
        }
        break;
      case 'AVI ':
      case 'ON2 ':
        break;
      case 'VP8 ':
      case 'VP8L':
      case 'VP8X':
      case 'ANIM':
      case 'ANMF':
      case 'ALPH':
      case 'ICCP':
      case 'EXIF':
      case 'XMP ':
        out.webpChunks = out.webpChunks || [];
        out.webpChunks.push({ id, at: o, size, payload });
        if (id === 'VP8X') {
          const flags = u8[payload];
          out.hasAlpha = !!(flags & 0x10);
          out.animated = !!(flags & 0x02);
          out.width = 1 + (u8[payload + 4] | (u8[payload + 5] << 8) | (u8[payload + 6] << 16));
          out.height = 1 + (u8[payload + 7] | (u8[payload + 8] << 8) | (u8[payload + 9] << 16));
          out.bgColor = u8[payload + 10] | (u8[payload + 11] << 8) | (u8[payload + 12] << 16) | (u8[payload + 13] << 24);
        } else if (id === 'VP8 ') {
          out.width = (u8[payload + 6] | (u8[payload + 7] << 8)) & 0x3fff;
          out.height = (u8[payload + 8] | (u8[payload + 9] << 8)) & 0x3fff;
        } else if (id === 'VP8L') {
          const bits = v.getUint32(payload + 1, true);
          out.width = (bits & 0x3fff) + 1;
          out.height = ((bits >> 14) & 0x3fff) + 1;
          out.hasAlpha = ((bits >> 28) & 1) === 1;
        } else if (id === 'ANMF') {
          out.frameCount = (out.frameCount || 0) + 1;
        }
        break;
      case 'fmt ':
        out.format = 'wave';
        out.channels = v.getUint16(payload + 2, little);
        out.sampleRate = v.getUint32(payload + 4, little);
        break;
      case 'data':
        if (out.format === 'wave') out.dataSize = size;
        else if (out.form === 'AVI ') {
          // '00dc'/'00db' live inside LIST movi; bare `data` only in odd muxes
        }
        break;
      default:
        break;
    }
    o = next;
  }
}

function parseInfoList(u8, v, start, end, out, little) {
  let o = start;
  const info = (out.info = out.info || {});
  const map = { INAM: 'name', IART: 'artist', ICMT: 'comment', ICRD: 'date', ISFT: 'software', IGNR: 'genre', IPRD: 'product', IENG: 'engineer' };
  while (o + 8 <= end) {
    const id = FOURCC(u8, o);
    const size = v.getUint32(o + 4, little);
    if (map[id]) {
      let s = '';
      for (let i = 0; i < size && o + 8 + i < end; i++) {
        const c = u8[o + 8 + i];
        if (!c) break;
        s += String.fromCharCode(c);
      }
      info[map[id]] = s;
    }
    o += 8 + size + (size & 1);
  }
}

function parseAvih(v, p, out, little) {
  const usPerFrame = v.getUint32(p, little);
  out.maxBytesPerSec = v.getUint32(p + 4, little);
  out.flags = v.getUint32(p + 12, little);
  out.totalFrames = v.getUint32(p + 16, little);
  out.initialFrames = v.getUint32(p + 20, little);
  out.streamCount = v.getUint32(p + 24, little);
  out.width = v.getUint32(p + 32, little);
  out.height = v.getUint32(p + 36, little);
  out.usPerFrame = usPerFrame || 40000;
  if (usPerFrame) {
    out.fps = 1e6 / usPerFrame;
    out.durationMs = (usPerFrame * out.totalFrames) / 1000;
  }
  out.truncated = !!(out.flags & 0x10);
  if (out.truncated) out.oddities.push('AVI flagged truncated');
}

function parseStrh(v, p, size, out, little) {
  const fccType = FOURCC(new Uint8Array(v.buffer, v.byteOffset + p + 8, 4), 0);
  const fccHandler = FOURCC(new Uint8Array(v.buffer, v.byteOffset + p + 12, 4), 0);
  const s = {
    type: fccType,
    handler: fccHandler,
    scale: v.getUint32(p + 20, little),
    rate: v.getUint32(p + 24, little),
    start: v.getUint32(p + 28, little),
    length: v.getUint32(p + 32, little),
    quality: v.getUint32(p + 36, little),
    samplesPerChunk: v.getUint32(p + 48, little),
  };
  if (s.rate && s.scale) s.fps = s.rate / s.scale;
  out.streams.push(s);
  if (fccType === 'vids' && !out.video) out.video = s;
  else if (fccType === 'auds' && !out.audio) out.audio = s;
  void size;
}

function parseStrf(v, p, size, out, little) {
  const s = out.streams[out.streams.length - 1];
  if (!s) return;
  const biSize = v.getUint32(p, little);
  if (biSize >= 40) {
    s.biSize = biSize;
    s.width = v.getInt32(p + 4, little);
    s.height = v.getInt32(p + 8, little);
    s.planes = v.getUint16(p + 12, little);
    s.bitCount = v.getUint16(p + 14, little);
    s.compression = FOURCC(new Uint8Array(v.buffer, v.byteOffset + p + 16, 4), 0);
    s.imageSize = v.getUint32(p + 20, little);
    s.xPelsPerMeter = v.getInt32(p + 24, little);
    s.yPelsPerMeter = v.getInt32(p + 28, little);
    s.bottomUp = s.height > 0;
    if (biSize >= 56) {
      s.clrUsed = v.getUint32(p + 40, little);
      s.important = v.getUint32(p + 44, little);
      if (s.compression === 'JPEG') {
        s.quality = v.getUint16(p + 48, little);
        s.pixelFormat = FOURCC(new Uint8Array(v.buffer, v.byteOffset + p + 50, 4), 0);
      }
    }
    if (s.type === 'vids' || out.video === s) {
      out.width = out.width || Math.abs(s.width);
      out.height = out.height || Math.abs(s.height);
      out.codec = s.compression;
      s.hasAlpha = s.bitCount === 32;
      out.hasAlpha = s.bitCount === 32;
    }
  } else if (biSize === 0) {
    // WAVEFORMATEX for audio streams
    s.formatTag = v.getUint16(p, little);
    s.channels = v.getUint16(p + 2, little);
    s.sampleRate = v.getUint32(p + 4, little);
    s.bitsPerSample = v.getUint16(p + 14, little);
  }
  void size;
}

function parseIdx1(u8, v, p, size, out, little) {
  const n = Math.floor(size / 16);
  if (!n) return;
  // `idx1` offsets are relative to the start of the `movi` LIST's *data*
  // (i.e. its fourcc). Some muxers get this wrong by 4 bytes, so probe both and
  // keep whichever base makes every entry land inside the file.
  const moviStart = out.moviStart;
  const bases = moviStart != null ? [moviStart, moviStart + 4, moviStart - 4, 12] : [12];
  const list = [];
  for (let i = 0; i < n && i < 500000; i++) {
    const at = p + i * 16;
    if (at + 16 > p + size) break;
    const ckid = FOURCC(u8, at);
    if (!/^[0-9][0-9](dc|db|pc|pb)$/.test(ckid)) continue;
    const flags = v.getUint32(at + 4, little);
    const offset = v.getUint32(at + 8, little);
    const length = v.getUint32(at + 12, little);
    list.push({ index: list.length, fourcc: ckid, isKey: (flags & 0x10) !== 0, offsetBase: -1, offset, size: length });
  }
  if (!list.length) return;
  // A base is only right if it also puts each entry's *chunk header* at the
  // expected fourcc — offsets that merely land inside the file are not enough.
  const okAt = (base, f) => {
    const at = f.offset + base;
    if (at < 8 || at + f.size > u8.length) return false;
    return FOURCC(u8, at - 8) === f.fourcc && v.getUint32(at - 4, little) === f.size;
  };
  for (const base of bases) {
    if (list.every((f) => okAt(base, f))) {
      for (const f of list) {
        f.absoluteOffset = f.offset + base;
        f.offset = f.absoluteOffset;
      }
      out.frames = list;
      out.indexSource = 'idx1';
      out.indexBase = base;
      return;
    }
  }
  out.oddities.push('idx1 offsets out of range — rescanning movi');
  out.indexBroken = true;
}

/** Rebuild an index by walking `movi` (handles files written without `idx1`). */
function scanMovi(u8, out, opts) {
  if (!(opts.scanIfNoIndex !== false) || out.moviStart == null) return;
  const v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const end = Math.min(u8.length, out.moviStart - 4 + 8 + (out.moviSize || u8.length));
  const list = [];
  let o = out.moviStart + 4; // skip the 'movi' fourcc, land on the first chunk header
  const budget = opts.maxScanBytes || 512 * 1024 * 1024;
  let scanned = 0;
  while (o + 8 <= end && scanned < budget) {
    const id = FOURCC(u8, o);
    const size = v.getUint32(o + 4, true);
    if (size > 0x08000000 || o + 8 + size > end) {
      out.partial = true;
      break;
    }
    if (/^[0-9][0-9](dc|db)$/.test(id)) {
      // Every frame in an AVI we can actually read (uncompressed DIB or MJPG) is
      // independently decodable, so treat them all as keyframes.
      list.push({ index: list.length, fourcc: id, isKey: true, offset: o + 8, size });
    }
    o += 8 + size + (size & 1);
    scanned += 8 + size;
  }
  if (list.length) {
    out.frames = list;
    out.indexSource = 'scan';
    out.oddities.push('index rebuilt by scanning movi');
  }
}
function summarizeAvi(out, opts, u8) {
  const u8For = () => out.__u8 || new Uint8Array(0);
  if (out.video) {
    if (out.video.fps && !out.fps) {
      out.fps = out.video.fps;
      out.durationMs = out.video.length ? (out.video.length / out.video.rate) * 1000 / (1 / out.video.scale) : out.durationMs;
      if (out.video.rate && out.video.scale) out.durationMs = (out.video.length * out.video.scale * 1000) / out.video.rate;
    }
    if (!out.width && out.video.width) out.width = Math.abs(out.video.width);
    if (!out.height && out.video.height) out.height = Math.abs(out.video.height);
  }
  if (!out.frames || !out.frames.length) scanMovi(u8 || u8For(out), out, opts);
  if (out.frames && out.frames.length) {
    out.frameCount = out.frames.length;
    if (out.fps) out.durationMs = (out.frames.length * 1000) / out.fps;
    out.keyframeCount = out.frames.reduce((a, f) => a + (f.isKey ? 1 : 0), 0);
  } else if (out.totalFrames) out.frameCount = out.totalFrames;
  out.format = 'avi';
  if (out.openDml) out.format = 'avi-opendml';
  if (opts.strict && (!out.video || !out.frames)) throw new GifxError('AVI has no usable video index', { code: ErrorCode.DEMUX_SAMPLE_TABLE, details: { chunks: out.chunks.map((c) => c.id) } });
}

function summarizeWebp(out, v, little) {
  out.format = 'webp';
  out.mime = 'image/webp';
}

/** Bytes of one AVI frame (video chunk payload). */
export function riffFrameBytes(u8, frame) {
  return u8.subarray(frame.offset, frame.offset + frame.size);
}

/**
 * Encode an RGB raster sequence into an uncompressed AVI (24-bit biRGB) — used by
 * `save({type:'avi'})` for tools that refuse to read GIF, and as the fallback
 * path in environments without a WebM muxer. ~1.2 MB/s of payload at 320x240@10,
 * so it's a *last resort*; we warn rather than silently produce huge files.
 */
export function buildUncompressedAvi(frames, opts = {}) {
  const width = opts.width || frames[0].width;
  const height = opts.height || frames[0].height;
  const fps = opts.fps || 10;
  const rowSize = ((width * 3 + 3) >> 2) << 2;
  const imgSize = rowSize * height;
  const n = frames.length;
  const moviSize = n * (8 + imgSize);
  const idx1Size = n * 16;
  const total = 4 + 4 + (4 + 8 + 88 + 4 + (4 + 8 + 124) + (4 + 8 + 40) + (4 + 8 + 4 + 1176)) + (4 + 8 + moviSize) + (4 + 8 + idx1Size) + 8 + 8;
  const buf = new Uint8Array(8 + total);
  const v = new DataView(buf.buffer);
  let o = 0;
  const put = (s) => {
    for (let i = 0; i < s.length; i++) buf[o++] = s.charCodeAt(i);
  };
  const u32at = (val) => {
    v.setUint32(o, val, true);
    o += 4;
  };
  const u16at = (val) => {
    v.setUint16(o, val, true);
    o += 2;
  };
  put('RIFF');
  u32at(total);
  put('AVI ');
  put('LIST');
  const hdrlStart = o;
  u32at(0);
  put('hdrl');
  put('avih');
  u32at(56);
  const avihStart = o;
  u32at(Math.round(1e6 / fps));
  u32at(Math.round((imgSize * n) / (n / fps)));
  u32at(0);
  u32at(0x10);
  u32at(n);
  u32at(n);
  u32at(2);
  u32at(0);
  u32at(width);
  u32at(height);
  buf.fill(0, o, o + 16);
  o += 16;
  put('LIST');
  const strlStart = o;
  u32at(0);
  put('strl');
  put('strh');
  u32at(56);
  put('vids');
  put('DIB ');
  u32at(0);
  u16at(0);
  u16at(0);
  u32at(0);
  u32at(1);
  u32at(Math.round(fps) || 10);
  u32at(0);
  u32at(n);
  u32at(0xffffffff);
  u32at(0);
  u32at(1);
  u32at(0);
  put('DIB ');
  u32at(0);
  u16at(0);
  u16at(0);
  put('strf');
  u32at(40);
  u32at(40);
  u32at(width);
  u32at(-height);
  u16at(1);
  u16at(24);
  u32at(0);
  u32at(imgSize);
  u32at(2835);
  u32at(2835);
  u32at(0);
  u32at(0);
  v.setUint32(strlStart, o - strlStart - 4, true);
  v.setUint32(hdrlStart, o - hdrlStart - 4, true);
  put('LIST');
  const moviStart = o;
  u32at(0);
  put('movi');
  const index = [];
  for (let i = 0; i < n; i++) {
    const f = frames[i];
    put('00db');
    u32at(imgSize);
    index.push({ offset: o - (moviStart + 4), size: imgSize });
    const d = f.data;
    for (let y = 0; y < height; y++) {
      const dy = height - 1 - y; // AVI DIB is bottom-up
      const srow = dy * f.stride;
      const drow = o + y * rowSize;
      for (let x = 0; x < width; x++) {
        const si = srow + x * 4;
        buf[drow + x * 3] = d[si + 2];
        buf[drow + x * 3 + 1] = d[si + 1];
        buf[drow + x * 3 + 2] = d[si];
      }
    }
    o += imgSize;
  }
  v.setUint32(moviStart, o - moviStart - 4, true);
  put('idx1');
  u32at(idx1Size);
  for (let i = 0; i < n; i++) {
    put('00db');
    u32at(i % 15 === 0 ? 0x10 : 0);
    u32at(index[i].offset);
    u32at(index[i].size);
  }
  v.setUint32(4, o - 8, true);
  void avihStart;
  return buf.subarray(0, o);
}

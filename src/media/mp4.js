/**
 * GIFX Kernel — ISO-BMFF (MP4 / MOV / M4V) demuxer.
 *
 * Enough of the spec to do the things a GIF encoder actually needs:
 *  - duration/timescale/fps *without decoding* (mvhd + stts)
 *  - frame-accurate sample timestamps + per-sample byte ranges (stts/ctts/
 *    stss/stsz/stsc/stco) so we can feed WebCodecs directly from a Worker with
 *    no `<video>` element and no seek jitter
 *  - the codec configuration record (avcC/hvcC/vpcC/av1C) for `VideoDecoder`
 *  - keyframe positions for fast starts and accurate trimming
 *
 * Progressive-download files (moov at the end) are detected and reported as
 * `needsFullFile`, because a partial read cannot yield a sample table.
 *
 * @module media/mp4
 */
import { GifxError, ErrorCode } from '../core/errors.js';

const CONTAINERS = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl', 'dinf', 'edts', 'mvex', 'sinf', 'schi', 'moof', 'traf', 'mdat', 'udta', 'meta', 'ilst', '©too']);
/** Boxes we descend into that are not "full" containers. */
const LEAFS_WE_CARE_ABOUT = new Set(['ftyp', 'mdat', 'free', 'skip', 'wide', 'uuid', 'pssh']);

/**
 * @typedef {Object} Mp4Sample
 * @property {number} offset byte offset of the sample inside the file
 * @property {number} size
 * @property {number} dts decode timestamp in timescale units
 * @property {number} cts composition timestamp offset
 * @property {number} pts presentation timestamp (dts + cts)
 * @property {boolean} key
 */

/**
 * Parse a whole (or head of a) MP4 file.
 * @param {Uint8Array} u8
 * @param {object} [opts] `{maxSamples, keepSamples:boolean, framesOnly:boolean}`
 */
export function parseMp4(u8, opts = {}) {
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = {
    format: 'mp4',
    partial: false,
    needsFullFile: false,
    brands: [],
    majorBrand: '',
    tracks: [],
    video: null,
    audio: null,
    fragments: 0,
    fastStart: false,
    encrypted: false,
    width: 0,
    height: 0,
    durationMs: 0,
    timescale: 0,
    frameCount: 0,
    fps: 0,
    rotation: 0,
    codec: '',
    mimeCodecs: '',
    hasAlpha: false,
    mdat: null,
  };
  if (u8.length < 8) throw new GifxError('file too small to be MP4', { code: ErrorCode.DEMUX_TOO_SMALL, details: { size: u8.length } });
  const boxes = readBoxes(view, 0, u8.length, out, 0, opts);
  if (!boxes.length) throw new GifxError('no boxes found — not an MP4/MOV', { code: ErrorCode.DEMUX_PARSE, details: { head: Array.from(u8.subarray(0, 12)) } });
  out.boxes = boxes.map((b) => ({ type: b.type, start: b.start, size: b.size }));
  out.fastStart = boxes.some((b) => b.type === 'moov') && boxes.some((b) => b.type === 'mdat') && boxes.findIndex((b) => b.type === 'moov') < boxes.findIndex((b) => b.type === 'mdat');
  const mdatBox = boxes.find((b) => b.type === 'mdat');
  if (mdatBox) out.md = out.mdat = { start: mdatBox.start + mdatBox.headerSize, end: mdatBox.end };
  return out;
}

function readBoxes(view, start, end, out, depth, opts) {
  const list = [];
  let o = start;
  while (o + 8 <= end) {
    let size = view.getUint32(o);
    const type = str(view, o + 4, 4);
    let headerSize = 8;
    if (size === 1) {
      if (o + 16 > end) break;
      const hi = view.getUint32(o + 8);
      const lo = view.getUint32(o + 12);
      size = hi * 4294967296 + lo;
      headerSize = 16;
    } else if (size === 0) size = end - o;
    if (size < headerSize || o + size > end) {
      // truncated tail (very common with aborted downloads): keep what we have
      out.partial = true;
      size = Math.max(headerSize, Math.min(size, end - o));
      if (o + size > end) {
        out.truncatedAt = o;
        break;
      }
    }
    const box = { type, start: o, headerSize, size, end: o + size, payloadStart: o + headerSize, payloadEnd: o + size };
    list.push(box);
    if (type === 'ftyp') parseFtyp(box, view, out);
    else if (type === 'moov') parseMoov(box, view, out, depth, opts);
    else if (type === 'moof') {
      out.fragments++;
      out.fragmented = true;
    } else if (type === 'pssh' || type === 'sinf') out.encrypted = true;
    else if (type === 'uuid' && out.partial) void 0;
    o += size;
  }
  if (o < end && depth === 0) out.partial = true;
  return list;
}

const str = (view, off, len) => {
  let s = '';
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
  return s;
};

function parseFtyp(box, view, out) {
  out.majorBrand = str(view, box.payloadStart, 4).trim();
  const brands = [];
  for (let o = box.payloadStart + 8; o + 4 <= box.payloadEnd; o += 4) brands.push(str(view, o, 4).trim());
  out.brands = brands;
  out.mime = brands.some((b) => /^qt/.test(b)) ? 'video/quicktime' : 'video/mp4';
}

function parseMoov(box, view, out, depth, opts) {
  let o = box.payloadStart;
  while (o + 8 <= box.payloadEnd) {
    const size = view.getUint32(o);
    const type = str(view, o + 4, 4);
    if (size < 8 || o + size > box.payloadEnd) break;
    const p = o + 8;
    switch (type) {
      case 'mvhd': {
        const v = view.getUint8(p);
        if (v === 1) {
          out.timescale = view.getUint32(p + 20);
          out.duration = view.getUint32(p + 24);
        } else {
          out.timescale = view.getUint32(p + 12);
          out.duration = view.getUint32(p + 16);
        }
        if (out.timescale > 0) out.durationMs = (out.duration / out.timescale) * 1000;
        out.creationTime = readTime(view, p, v);
        break;
      }
      case 'trak':
        parseTrak({ payloadStart: p, payloadEnd: o + size }, view, out, depth + 1, opts);
        break;
      case 'mvex':
        out.fragmented = true;
        break;
      default:
        break;
    }
    o += size;
  }
  const v = out.video;
  if (v) {
    out.width = v.width;
    out.height = v.height;
    out.rotation = v.rotation;
    out.codec = v.codec;
    out.mimeCodecs = v.codecString;
    out.hasAlpha = !!v.hasAlpha;
    out.frameCount = v.samples ? v.samples.length : v.sampleCount || 0;
    if (v.durationMs) out.durationMs = v.durationMs;
    if (out.durationMs && out.frameCount) out.fps = (out.frameCount * 1000) / out.durationMs;
    else if (v.timescale && v.avgDuration) out.fps = v.timescale / v.avgDuration;
  }
}

function readTime(view, p, v) {
  try {
    return v === 1 ? Number(view.getBigUint64(p + 4)) : view.getUint32(p + 4);
  } catch {
    return 0;
  }
}

function parseTrak(box, view, out, depth, opts) {
  const track = { handler: '', samples: null, timescale: 0, width: 0, height: 0, codec: '', codecString: '', hasAlpha: false, trackId: 0, enabled: true };
  let o = box.payloadStart;
  while (o + 8 <= box.payloadEnd) {
    const size = view.getUint32(o);
    const type = str(view, o + 4, 4);
    if (size < 8 || o + size > box.payloadEnd) break;
    const p = o + 8;
    if (type === 'tkhd') {
      const v = view.getUint8(p);
      const isV1 = v === 1;
      track.flags = view.getUint8(p + 3);
      track.enabled = (track.flags & 1) === 1;
      track.trackId = view.getUint32(p + (isV1 ? 20 : 12));
      track.duration = isV1 ? readU64(view, p + 28) : view.getUint32(p + 20);
      track.rotation = readMatrixRotation(view, p + (isV1 ? 52 : 40));
      // Matrix can scale as well as rotate; a negative determinant is a mirror,
      // which we must NOT silently drop or the GIF comes out flipped.
      const dimAt = p + (isV1 ? 88 : 76);
      track.width = view.getUint32(dimAt) >> 16;
      track.height = view.getUint32(dimAt + 4) >> 16;
      if (track.flags & 2) track.disabled = true;
      if (track.flags & 4) track.inMovie = true;
    } else if (type === 'mdia') {
      parseMdia({ payloadStart: p, payloadEnd: o + size }, view, out, track, depth, opts);
    } else if (type === 'edts') {
      const el = findBox(view, p, o + size, 'elst');
      if (el) {
        const entryCount = view.getUint32(el + 8);
        if (entryCount > 0) {
          const v = view.getUint8(el);
          track.editDelay = v === 1 ? Number(view.getBigInt64(el + 20)) : view.getInt32(el + 16);
        }
      }
    }
    o += size;
  }
  out.tracks.push(track);
  if (track.handler === 'vide' && (!out.video || (track.samples && track.samples.length))) {
    out.video = track;
    // honour track width override: prefer the sample description, which is not
    // affected by transform matrices
    if (!track.width || !track.height) {
      track.width = track.stsdWidth || track.width;
      track.height = track.stsdHeight || track.height;
    }
  } else if (track.handler === 'soun' && !out.audio) out.audio = track;
}

function readMatrixRotation(view, at) {
  try {
    const a = view.getInt32(at) / 65536;
    const b = view.getInt32(at + 4) / 65536;
    if (a === 0 && b === 0) return 0;
    let deg = Math.round((Math.atan2(b, a) * 180) / Math.PI);
    if (deg < 0) deg += 360;
    return deg;
  } catch {
    return 0;
  }
}

function parseMdia(box, view, out, track, depth, opts) {
  let o = box.payloadStart;
  while (o + 8 <= box.payloadEnd) {
    const size = view.getUint32(o);
    const type = str(view, o + 4, 4);
    if (size < 8 || o + size > box.payloadEnd) break;
    const p = o + 8;
    if (type === 'mdhd') {
      const v = view.getUint8(p);
      track.timescale = v === 1 ? view.getUint32(p + 20) : view.getUint32(p + 12);
      track.mdhdDuration = v === 1 ? Number(view.getBigUint64(p + 24)) : view.getUint32(p + 16);
      if (track.timescale > 0) track.durationMs = (track.mdhdDuration / track.timescale) * 1000;
    } else if (type === 'hdlr') {
      track.handler = str(view, p + 8, 4);
      // HandlerBox: version/flags(4) + predefined(4) + handlerType(4) +
      // reserved(12) → name starts at p + 24.
      const nameLen = size - 8 - 4 - 4 - 4 - 12 - 1;
      if (nameLen > 0 && nameLen < 128) {
        let nm = '';
        for (let i = 0; i < nameLen; i++) nm += String.fromCharCode(view.getUint8(p + 24 + i));
        track.handlerName = nm.replace(/\0.*/g, '');
      }
    } else if (type === 'minf') {
      parseMinf({ payloadStart: p, payloadEnd: o + size }, view, out, track, opts);
    }
    o += size;
  }
}

function parseMinf(box, view, out, track, opts) {
  const stbl = findBoxRange(view, box.payloadStart, box.payloadEnd, 'stbl');
  if (!stbl) return;
  parseStbl(stbl, view, out, track, opts);
}

function parseStbl(box, view, out, track, opts) {
  let o = box.payloadStart;
  const tables = {};
  while (o + 8 <= box.payloadEnd) {
    const size = view.getUint32(o);
    const type = str(view, o + 4, 4);
    if (size < 8 || o + size > box.payloadEnd) break;
    tables[type] = { start: o + 8, end: o + size, at: o };
    o += size;
  }
  if (tables.stsd) parseStsd(tables.stsd, view, out, track);
  if (tables.stsz) parseStsz(tables.stsz, view, out, track);
  if (tables.stts) parseStts(tables.stts, view, out, track);
  if (tables.stss) parseStss(tables.stss, view, out, track);
  if (tables.ctts) parseCtts(tables.ctts, view, out, track);

  const chunkOffsets = tables.co64 || tables.stco;
  if (chunkOffsets && track.sampleSizes) {
    buildSampleOffsets(view, out, track, tables, chunkOffsets, opts);
  }
  if (track.samples && track.samples.length && track.timescale) {
    let last = track.samples[track.samples.length - 1];
    track.lastPts = (last.pts + (last.duration || 0)) / track.timescale;
    if (!track.durationMs) track.durationMs = track.lastPts * 1000;
    const keyCount = track.samples.reduce((a, s) => a + (s.key ? 1 : 0), 0);
    track.keyframeCount = keyCount;
  }
}

function readU64(view, at) {
  try {
    return Number(view.getBigUint64(at));
  } catch {
    return view.getUint32(at) * 4294967296 + view.getUint32(at + 4);
  }
}

function parseStsd(t, view, out, track) {
  const count = view.getUint32(t.start + 4);
  let o = t.start + 8;
  for (let i = 0; i < count && o + 8 <= t.end; i++) {
    const size = view.getUint32(o);
    const format = str(view, o + 4, 4);
    track.sampleCount = count;
    if (format === 'avc1' || format === 'avc3') parseAvc(view, o, size, track, format);
    else if (format === 'hvc1' || format === 'hev1') parseHevc(view, o, size, track, format);
    else if (format === 'av01') parseAv1(view, o, size, track);
    else if (format === 'vp08' || format === 'vp09') parseVpcC(view, o, size, track, format);
    else {
      track.codec = format;
      track.codecString = format;
    }
    if (['mp4a', 'mp4v', 'ac-3', 'ec-3', 'alac', 'Opus', 'opus', 'twos', 'sowt', 'text', 'sbtl', 'clap'].includes(format)) {
      // non-video sample entries have a different layout; don't try to read dims
    } else {
      track.stsdWidth = view.getUint16(o + VISUAL_W);
      track.stsdHeight = view.getUint16(o + VISUAL_W + 2);
      track.width = track.stsdWidth || track.width;
      track.height = track.stsdHeight || track.height;
    }
    o += Math.max(8, size);
  }
}

/**
 * VisualSampleEntry layout: 8 header + 6 reserved + 2 dataRefIdx + 16 predefs +
 * 2 rev + 4 reserved + 12 predefs → width/height at +24 (after header).
 */
/** VisualSampleEntry: 86 bytes before its child boxes; width/height at +32. */
const VISUAL_CHILDREN = 86;
const visualEntryBase = (o) => o + VISUAL_CHILDREN;
const VISUAL_W = 32;

function parseAvc(view, o, size, track, format) {
  const base = visualEntryBase(o);
  track.width = view.getUint16(o + VISUAL_W);
  track.height = view.getUint16(o + VISUAL_W + 2);
  track.codec = format;
  // walk the child boxes after the VisualSampleEntry for avcC / colr / pasp
      let p = base;
  const extra = {};
  while (p + 8 <= o + size) {
    const csz = view.getUint32(p);
    const ctype = str(view, p + 4, 4);
    if (csz < 8 || p + csz > o + size) break;
    if (ctype === 'avcC') {
      track.description = copyBytes(view, p + 8, csz - 8);
      // AVCDecoderConfigurationRecord: 0=ver 1=profile 2=compat 3=level
      // 4=lengthSizeMinusOne(2 bits) 5=numOfSPS
      track.nalLengthSize = (view.getUint8(p + 12) & 3) + 1;
      track.profile = view.getUint8(p + 9);
      track.profileCompat = view.getUint8(p + 10);
      track.level = view.getUint8(p + 11);
      extra.avcC = true;
    } else if (ctype === 'pasp') {
      track.hSpacing = view.getUint32(p + 8);
      track.vSpacing = view.getUint32(p + 12);
      track.sampleAspectRatio = track.vSpacing ? track.hSpacing / track.vSpacing : 1;
    } else if (ctype === 'clap') {
      track.cleanAperture = readAperture(view, p);
    } else if (ctype === 'colr' || ctype === 'nclx') {
      track.colour = readColour(view, p, csz);
    } else if (ctype === 'senc' || ctype === 'saiz') {
      track.encrypted = true;
      out.encrypted = true;
    }
    p += csz;
  }
  track.codecString = avcCodecString(track);
  track.hasAlpha = format === 'apple' || false;
  void extra;
}

function readAperture(view, p) {
  const r = (at) => {
    const c = view.getUint32(at);
    const d = view.getUint32(at + 4);
    return d ? c / d : 0;
  };
  return {
    width: r(p + 8),
    height: r(p + 12),
    hOffset: r(p + 16),
    vOffset: r(p + 20),
  };
}

function readColour(view, p, size) {
  if (size < 15) return null;
  const type = str(view, p + 8, 4);
  if (type !== 'nclx') return { type };
  return {
    type: 'nclx',
    primaries: view.getUint16(p + 12),
    transfer: view.getUint16(p + 14),
    matrix: view.getUint16(p + 16),
    fullRange: (view.getUint8(p + 18) & 1) === 1,
  };
}

function parseHevc(view, o, size, track, format) {
  const base = visualEntryBase(o);
  track.width = view.getUint16(o + VISUAL_W);
  track.height = view.getUint16(o + VISUAL_W + 2);
  track.codec = format;
      let p = base;
  while (p + 8 <= o + size) {
    const csz = view.getUint32(p);
    const ctype = str(view, p + 4, 4);
    if (csz < 8 || p + csz > o + size) break;
    if (ctype === 'hvcC' || ctype === 'hvc1') {
      track.description = copyBytes(view, p + 8, csz - 8);
      // hvcC: 0=ver 1=space(2)|tier(1)|profileIdc(5) 12=levelIdc
      // 21 bit0-1 = lengthSizeMinusOne 22>>1 chroma 23>>5 bitDepth
      const b1 = view.getUint8(p + 9);
      track.nalLengthSize = (view.getUint8(p + 29) & 3) + 1;
      track.profileSpace = (b1 >> 6) & 3;
      track.tier = (b1 >> 5) & 1;
      track.profile = b1 & 0x1f;
      track.level = view.getUint8(p + 20);
      track.chromaFormat = (view.getUint8(p + 30) >> 1) & 3;
      track.bitDepth = (view.getUint8(p + 31) >> 1) & 7;
    } else if (ctype === 'colr') track.colour = readColour(view, p, csz);
    else if (ctype === 'pasp') {
      track.hSpacing = view.getUint32(p + 8);
      track.vSpacing = view.getUint32(p + 12);
    }
    p += csz;
  }
  if (track.profile != null) {
    // `hvc1.<space><profile>.<compat>.<constraint>.0x<level>` — the shape
    // isTypeSupported() wants; a wrong guess only costs us the fallback path.
    const space = track.profileSpace === 0 ? '' : String.fromCharCode(65 + track.profileSpace);
    const compat = track.description && track.description.length > 6 ? Array.from(track.description.subarray(2, 6)).map((b) => b.toString(16).padStart(2, '0')).join('.') : '00.00.00.00';
    track.codecString = `${track.codec === 'hev1' ? 'hvc1' : 'hev1'}.${space}${track.profile}.${compat}.0x${(track.level || 0x5d).toString(16)}`;
  }
}

function parseVpcC(view, o, size, track, format) {
  const base = visualEntryBase(o);
  track.width = view.getUint16(o + VISUAL_W);
  track.height = view.getUint16(o + VISUAL_W + 2);
  track.codec = format;
  track.codecString = format === 'vp09' ? 'vp09.00.10.08' : format;
      let p = base;
  while (p + 8 <= o + size) {
    const csz = view.getUint32(p);
    const ctype = str(view, p + 4, 4);
    if (csz < 8 || p + csz > o + size) break;
    if (ctype === 'vpcC') {
      track.description = null; // WebM-style VP9 in MP4 needs the size-prefixed NAL format
      track.bitDepth = view.getUint8(p + 11) >> 4;
      track.colorProfile = view.getUint8(p + 11) & 7;
      track.chromaSubsampling = (view.getUint8(p + 12) >> 4) & 7;
      track.videoFullRange = ((view.getUint8(p + 12) >> 3) & 1) === 1;
    }
    p += csz;
  }
}

function parseAv1(view, o, size, track) {
  const base = visualEntryBase(o);
  track.width = view.getUint16(o + VISUAL_W);
  track.height = view.getUint16(o + VISUAL_W + 2);
  track.codec = 'av01';
      let p = base;
  while (p + 8 <= o + size) {
    const csz = view.getUint32(p);
    const ctype = str(view, p + 4, 4);
    if (csz < 8 || p + csz > o + size) break;
    if (ctype === 'av1C') {
      const b0 = view.getUint8(p + 8);
      track.description = copyBytes(view, p + 8, Math.min(csz - 8, 4));
      track.codecString = `av01.${b0 >> 5}.${((b0 & 31) >> 2).toString(16).padStart(2, '0')}.${((b0 & 2) << 2) | (view.getUint8(p + 9) >> 6)}`;
      track.seqLevelProfile = b0;
    } else if (ctype === 'colr') track.colour = readColour(view, p, csz);
    p += csz;
  }
  if (!track.codecString) track.codecString = 'av01.0.01M.0';
  // Alpha in AV1/HEVC is carried as a second ("auxiliary") track; note it.
  track.hasAlpha = false;
}

function avcCodecString(track) {
  if (track.profile == null) return 'avc1.42001e';
  const hex = (n) => n.toString(16).padStart(2, '0');
  return `avc1.${hex(track.profile)}${hex(track.profileCompat)}${hex(track.level)}`;
}

function parseStsz(t, view, out, track) {
  const sampleSize = view.getUint32(t.start + 4);
  const count = view.getUint32(t.start + 8);
  if (count > 1e7) {
    out.partial = true;
    return;
  }
  const sizes = new Uint32Array(count);
  if (sampleSize) sizes.fill(sampleSize);
  else {
    if (t.start + 12 + count * 4 > t.end + 4) {
      out.partial = true;
      return;
    }
    for (let i = 0; i < count; i++) sizes[i] = view.getUint32(t.start + 12 + i * 4);
  }
  track.sampleSizes = sizes;
  track.sampleCount = count;
  track.bytesPerSample = count ? sizes.reduce((a, b) => a + b, 0) / count : 0;
}

function parseStts(t, view, out, track) {
  const count = view.getUint32(t.start + 4);
  const sizes = track.sampleSizes;
  const n = sizes ? sizes.length : 0;
  if (n === 0) return;
  const samples = track.samples || (track.samples = new Array(n));
  const durations = track.durations || (track.durations = new Float64Array(n));
  let idx = 0;
  let dts = 0;
  let totalDur = 0;
  let maxDur = 0;
  for (let i = 0; i < count && idx < n; i++) {
    const at = t.start + 8 + i * 8;
    if (at + 8 > t.end) break;
    const cnt = view.getUint32(at);
    const dur = view.getUint32(at + 4);
    for (let j = 0; j < cnt && idx < n; j++, idx++) {
      durations[idx] = dur;
      if (samples[idx]) samples[idx].dts = dts;
      else samples[idx] = { dts, size: sizes[idx], offset: 0, key: true, duration: dur };
      dts += dur;
      totalDur += dur;
      if (dur > maxDur) maxDur = dur;
    }
  }
  // fill remaining samples (malformed stts) with the last duration
  for (; idx < n; idx++) {
    durations[idx] = maxDur || 1;
    if (!samples[idx]) samples[idx] = { dts, size: sizes[idx], offset: 0, key: false, duration: durations[idx] };
    dts += durations[idx];
    totalDur += durations[idx];
  }
  track.avgDuration = n ? totalDur / n : 1;
  track.maxDuration = maxDur;
  if (track.timescale) track.durationMs = (dts / track.timescale) * 1000;
}

function parseStss(t, view, out, track) {
  const count = view.getUint32(t.start + 4);
  if (!track.samples) {
    track.keyframeOnly = true;
    track.syncSamples = new Uint32Array(Math.min(count, 1e6));
    for (let i = 0; i < count && i < 1e6; i++) track.syncSamples[i] = view.getUint32(t.start + 8 + i * 4) - 1;
    return;
  }
  for (let i = 0; i < track.samples.length; i++) track.samples[i].key = false;
  const n = Math.min(count, track.samples.length * 4);
  for (let i = 0; i < n; i++) {
    const at = t.start + 8 + i * 4;
    if (at + 4 > t.end) break;
    const idx = view.getUint32(at) - 1;
    if (idx >= 0 && idx < track.samples.length) track.samples[idx].key = true;
  }
  track.allSync = false;
}

function parseCtts(t, view, out, track) {
  const count = view.getUint32(t.start + 4);
  const version = view.getUint8(t.at || t.start);
  const signed = version === 1;
  if (!track.samples) return;
  let idx = 0;
  for (let i = 0; i < count; i++) {
    const at = t.start + 8 + i * 8;
    if (at + 8 > t.end) break;
    const cnt = view.getUint32(at);
    const off = signed ? view.getInt32(at + 4) : view.getUint32(at + 4);
    for (let j = 0; j < cnt && idx < track.samples.length; j++, idx++) {
      track.samples[idx].cts = off;
      track.samples[idx].pts = track.samples[idx].dts + off;
    }
  }
  track.hasBFrames = true;
}

function parseStsc(t, view, out, track) {
  const count = view.getUint32(t.start + 4);
  const table = [];
  for (let i = 0; i < count; i++) {
    const at = t.start + 8 + i * 12;
    if (at + 12 > t.end) break;
    table.push({ firstChunk: view.getUint32(at), samplesPerChunk: view.getUint32(at + 4), sampleDescriptionIndex: view.getUint32(at + 8) });
  }
  return table;
}

function buildSampleOffsets(view, out, track, tables, chunkOffsets, opts) {
  // Simple muxers (and some streaming fragmenters) omit `stsc` when every
  // sample is its own chunk or when there is exactly one chunk. Recover in those
  // two cases and flag the guess, instead of leaving every offset at 0.
  const singleChunkish = !tables.stsc;
  const stsc = tables.stsc ? parseStsc(tables.stsc, view, out, track) : [];
  if (singleChunkish) track.offsetsGuessed = true;
  const isCo64 = !!tables.co64;
  const step = isCo64 ? 8 : 4;
  // payload = version/flags(4) + entry_count(4) + entries[]
  const entryCount = view.getUint32(chunkOffsets.start + 4);
  const entriesAt = chunkOffsets.start + 8;
  const chunkCount = Math.max(0, Math.min(entryCount, Math.floor((chunkOffsets.end - entriesAt) / step)));
  if (singleChunkish) {
    const perChunk = chunkCount === 1 ? (track.samples ? track.samples.length : 0) || entryCount : 1;
    stsc.length = 0;
    stsc.push({ firstChunk: 1, samplesPerChunk: Math.max(1, perChunk), sampleDescriptionIndex: 1 });
  }
  const sizes = track.sampleSizes;
  if (!sizes || !stsc.length || !chunkCount) return;
  const samples = track.samples;
  const mdat = out.md;
  let sampleIndex = 0;
  let maxOffset = 0;
  let failed = false;
  for (let chunk = 0; chunk < chunkCount && sampleIndex < samples.length; chunk++) {
    // which stsc row applies to this chunk?
    let row = 0;
    for (let r = 0; r < stsc.length; r++) {
      if (stsc[r].firstChunk <= chunk + 1) row = r;
      else break;
    }
    if (!stsc.length) break;
    const perChunk = Math.max(1, stsc[row].samplesPerChunk);
    const raw = entriesAt + chunk * step;
    let offset = 0;
    if (isCo64) offset = view.getUint32(raw) * 4294967296 + view.getUint32(raw + 4);
    else offset = view.getUint32(raw);
    maxOffset = Math.max(maxOffset, offset);
    for (let s = 0; s < perChunk && sampleIndex < samples.length; s++, sampleIndex++) {
      samples[sampleIndex].offset = offset;
      offset += sizes[sampleIndex];
    }
    void mdat;
  }
  if (sampleIndex < samples.length) failed = true;
  // sanity: offsets must be inside the file
  if (maxOffset >= view.byteLength || failed) {
    out.partial = true;
    track.offsetsInvalid = true;
    if (opts.strict) throw new GifxError('MP4 stco/stsc do not resolve inside the file (truncated or streamed?)', { code: ErrorCode.DEMUX_SAMPLE_TABLE, details: { maxOffset, size: view.byteLength } });
  }
}

function findBox(view, start, end, type) {
  let o = start;
  while (o + 8 <= end) {
    const size = view.getUint32(o);
    const t = str(view, o + 4, 4);
    if (t === type) return o;
    if (size < 8) break;
    o += size;
  }
  return -1;
}

function findBoxRange(view, start, end, type) {
  const at = findBox(view, start, end, type);
  if (at < 0) return null;
  const size = view.getUint32(at);
  return { payloadStart: at + 8, payloadEnd: Math.min(end, at + size) };
}

const copyBytes = (view, off, len) => new Uint8Array(view.buffer.slice(view.byteOffset + off, view.byteOffset + off + len));

/* ------------------------------------------------------------ sample access */

/**
 * Extract one sample's bytes (NAL length prefixes included, i.e. exactly what
 * `EncodedVideoChunk` wants for avc1/avc3/hvc1/av01).
 */
export function mp4SampleBytes(u8, track, index) {
  const s = track.samples[index];
  if (!s) return null;
  return u8.subarray(s.offset, s.offset + s.size);
}

/**
 * Convert the sample table to millisecond timestamps, applying the edit list and
 * normalizing to presentation order (B-frames make decode order differ!).
 */
export function mp4FrameTimes(track, opts = {}) {
  const ts = track.timescale || 1000;
  const list = (track.samples || []).map((s, i) => ({
    index: i,
    pts: (s.pts ?? s.dts ?? 0) / ts * 1000,
    dts: (s.dts ?? 0) / ts * 1000,
    duration: (s.duration || 1) / ts * 1000,
    key: s.key !== false,
    size: s.size,
    offset: s.offset,
  }));
  list.sort((a, b) => a.pts - b.pts || a.index - b.index);
  const shift = track.editDelay ? (track.editDelay / ts) * 1000 : 0;
  for (let i = 0; i < list.length; i++) {
    list[i].pts -= shift;
    // duration from the following frame is more truthful for VFR sources
    if (i + 1 < list.length) list[i].durationToNext = list[i + 1].pts - list[i].pts;
  }
  if (opts.mergeShortFrames) {
    const minMs = opts.minFrameMs || 20;
    for (let i = list.length - 2; i >= 0; i--) {
      if (list[i].durationToNext < minMs) {
        // keep the *later* frame of a sub-threshold pair (more up to date)
        list.splice(i, 1);
      }
    }
  }
  if (opts.dedupeKeyframesOnly) return list.filter((f) => f.key);
  return list;
}

/**
 * Range of samples that must be *fed* to a decoder to reach `ptsMs` with
 * accurate output: start at the previous keyframe, end at the target.
 */
export function mp4SeekRange(track, ptsMs) {
  const ts = track.timescale || 1000;
  const times = mp4FrameTimes(track);
  let target = 0;
  for (let i = 0; i < times.length; i++) {
    if (times[i].pts >= ptsMs - 1) {
      target = i;
      break;
    }
    target = i;
  }
  // map back to decode order
  const want = times[target];
  let start = want.index;
  while (start > 0 && !track.samples[start].key) start--;
  const decodeCount = Math.max(1, want.index - start + 1);
  return { firstDecodeSample: start, lastSample: want.index, sampleCount: decodeCount, frames: times, targetTimeMs: want.pts, warmupFrames: decodeCount };
}

/**
 * Minimal fMP4/MP4 writer used for the "wrap raw Annex-B H.264" path and for
 * lossless re-mux of trimmed clips.
 */
export function buildMp4InitSegment(track) {
  const boxes = [];
  boxes.push(box('ftyp', cat(u32be(0x69736f6d), u32be(0x200), u32be(0x69736f6d), u32be(0x69736f32))));
  const mvhd = cat(
    u8(0), u8(0), u8(0), u8(0),
    u32be(0), u32be(0),
    u32be(track.timescale || 1000),
    u32be(0),
    u32be(0x00010000), u32be(0), u32be(0), u32be(0),
    u32be(0x00010000), u32be(0), u32be(0), u32be(0),
    u32be(0), u32be(0x00010000), u32be(0), u32be(0), u32be(0), u32be(0),
    new Uint8Array(24),
    u32be(2), u32be(0), u32be(0)
  );
  boxes.push(box('mvhd', mvhd));
  return boxes;
}

/* --------------------------------------------------------------- box utils */

export function box(type, payload) {
  const head = new Uint8Array(8);
  const v = new DataView(head.buffer);
  v.setUint32(0, 8 + payload.byteLength);
  for (let i = 0; i < 4; i++) v.setUint8(4 + i, type.charCodeAt(i));
  return cat(head, payload);
}
export function fullBox(type, version, flags, payload) {
  const head = new Uint8Array(12);
  const v = new DataView(head.buffer);
  v.setUint32(0, 12 + payload.byteLength);
  for (let i = 0; i < 4; i++) v.setUint8(4 + i, type.charCodeAt(i));
  v.setUint8(8, version);
  v.setUint8(9, (flags >> 16) & 255);
  v.setUint8(10, (flags >> 8) & 255);
  v.setUint8(11, flags & 255);
  return cat(head, payload);
}
const u8 = (n) => new Uint8Array([n & 255]);
const u16be = (n) => new Uint8Array([(n >> 8) & 255, n & 255]);
export const u32be = (n) => new Uint8Array([(n / 16777216) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255]);
export const u64be = (n) => cat(u32be(Math.floor(n / 4294967296)), u32be(n >>> 0));
export function cat(...arrs) {
  let len = 0;
  for (const a of arrs) len += a.byteLength;
  const out = new Uint8Array(len);
  let o = 0;
  for (const a of arrs) {
    out.set(a instanceof Uint8Array ? a : new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o);
    o += a.byteLength;
  }
  return out;
}
export { CONTAINERS as MP4_CONTAINERS, LEAFS_WE_CARE_ABOUT as MP4_LEAVES };

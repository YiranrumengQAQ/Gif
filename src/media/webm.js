/**
 * GIFX Kernel — EBML (WebM / Matroska) demuxer.
 *
 * Parses just enough of the (brilliantly extensible but annoyingly variable-int
 * encoded) EBML tree for a GIF pipeline:
 *  - duration + timecode scale (Info)
 *  - the video track: codec ID, pixel size, `DefaultDuration`, cropping,
 *    projection (for VR), colour metadata and alpha (`ALPH`-style BlockAdditional)
 *  - the block index: timestamps + keyframe flags + file offsets, either from
 *    `Cues` (cheap) or by scanning clusters (accurate, capped by a byte budget)
 *
 * Real-world quirks handled here: unknown/large element sizes (streaming
 * "unknown length" clusters), 1-byte padding in vints (technically invalid,
 * produced by some encoders), negative block relative timestamps, BlockGroup
 * vs SimpleBlock, BlockMore/BlockAdditional (alpha!), lace-1 blocks (we flag
 * `lacing: true` rather than mis-counting frames), and void/padding.
 *
 * @module media/webm
 */
import { GifxError, ErrorCode } from '../core/errors.js';

const ID = {
  EBML: 0x1a45dfa3,
  DocType: 0x4282,
  Segment: 0x18538067,
  Info: 0x1549a966,
  TimecodeScale: 0x2ad7b1,
  Duration: 0x4489,
  MuxingApp: 0x4d80,
  WritingApp: 0x5741,
  Title: 0x7ba9,
  Tracks: 0x1654ae6b,
  TrackEntry: 0xae,
  TrackNumber: 0xd7,
  TrackType: 0x83,
  CodecID: 0x86,
  CodecPrivate: 0x63a2,
  DefaultDuration: 0x23e383,
  FlagLacing: 0x9c,
  Language: 0x22b59c,
  Name: 0x536e,
  Video: 0xe0,
  PixelWidth: 0xb0,
  PixelHeight: 0xba,
  DisplayWidth: 0x54b0,
  DisplayHeight: 0x54ba,
  CropLeft: 0x55aa,
  CropRight: 0x55bb,
  CropTop: 0x55cc,
  CropBottom: 0x55dd,
  FlagInterlaced: 0x9a,
  FieldOrder: 0x9d,
  StereoMode: 0x53b8,
  AlphaMode: 0x53c0,
  Projection: 0x7670,
  ProjectionType: 0x7671,
  Colour: 0x55b0,
  ColourPrimaries: 0x55b1,
  TransferCharacteristics: 0x55b2,
  MatrixCoefficients: 0x55b3,
  BitDepth: 0x6240,
  BlockAdditionMapping: 0x41e4,
  Cluster: 0x1f43b675,
  Timecode: 0xe7,
  SimpleBlock: 0xa3,
  BlockGroup: 0xa1,
  Block: 0xa2,
  BlockDuration: 0x9b,
  ReferenceBlock: 0xfb,
  BlockAdditions: 0x75a1,
  BlockMore: 0xa6,
  BlockAddID: 0xee,
  BlockAdditional: 0xa5,
  Keyframe: 0x88,
  Cues: 0x1c53bb6b,
  CuePoint: 0xbb,
  CueTime: 0xb3,
  CueTrackPositions: 0xb7,
  CueTrack: 0xf7,
  CueClusterPosition: 0xf1,
  CueRelativePosition: 0xf0,
  SeekHead: 0x114d9b74,
  Void: 0xec,
  Padding: 0xed,
  Tags: 0x1254c367,
  Attachments: 0x1941a469,
  ChapterTrack: 0x1043c77,
};

class Reader {
  constructor(u8) {
    this.u8 = u8;
    this.v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    this.pos = 0;
    this.error = null;
  }
  get eof() {
    return this.pos >= this.u8.length;
  }
  /** EBML variable size: returns {value, bytes} or null on truncation. */
  readVint(maxBytes = 8, signed = false) {
    if (this.pos >= this.u8.length) return null;
    const first = this.u8[this.pos];
    if (!first) {
      // 0x00 start byte: invalid marker length; bail so the caller can resync
      this.error = 'invalid vint';
      return null;
    }
    let len = 1;
    let mask = 0x80;
    while (len <= 8 && !(first & mask)) {
      mask >>= 1;
      len++;
    }
    if (len > 8 || this.pos + len > this.u8.length) return null;
    if (maxBytes && len > maxBytes) return null;
    let value = first & (0xff >> len);
    for (let i = 1; i < len; i++) value = value * 256 + this.u8[this.pos + i];
    if (signed && value >= 2 ** (7 * len - 1)) value -= 2 ** (8 * len - 1);
    this.pos += len;
    return { value, bytes: len };
  }
  readId() {
    const start = this.pos;
    const first = this.u8[this.pos];
    if (first == null) return null;
    let len = 1;
    let mask = 0x80;
    while (len < 4 && !(first & mask)) {
      mask >>= 1;
      len++;
    }
    if (this.pos + len > this.u8.length) return null;
    let value = 0;
    for (let i = 0; i < len; i++) value = value * 256 + this.u8[this.pos + i];
    this.pos += len;
    return { value, bytes: len, at: start };
  }
  readUInt(n) {
    let v = 0;
    for (let i = 0; i < n && this.pos < this.u8.length; i++, this.pos++) v = v * 256 + this.u8[this.pos];
    return v;
  }
  readInt(n) {
    if (!n) return 0;
    let v = 0;
    const start = this.pos;
    for (let i = 0; i < n && this.pos < this.u8.length; i++, this.pos++) v = v * 256 + this.u8[this.pos];
    const bits = n * 8;
    if (bits < 53 && v >= 2 ** (bits - 1)) v -= 2 ** bits;
    void start;
    return v;
  }
  readFloat(n) {
    if (n === 4) {
      const v = this.v.getFloat32(this.pos);
      this.pos += 4;
      return v;
    }
    if (n === 8) {
      const v = this.v.getFloat64(this.pos);
      this.pos += 8;
      return v;
    }
    return this.readUInt(n);
  }
  readBinary(n) {
    const out = this.u8.subarray(this.pos, Math.min(this.u8.length, this.pos + n));
    this.pos += n;
    return out;
  }
  readString(n) {
    const bytes = this.u8.subarray(this.pos, Math.min(this.u8.length, this.pos + n));
    let s = '';
    for (let i = 0; i < bytes.length; i++) {
      if (!bytes[i]) break;
      s += String.fromCharCode(bytes[i]);
    }
    this.pos += n;
    try {
      return typeof TextDecoder !== 'undefined' ? new TextDecoder().decode(bytes.subarray(0, s.length)) : s;
    } catch {
      return s;
    }
  }
}

/**
 * @param {Uint8Array} u8
 * @param {object} [opts] `{scanBlocks:'auto'|'always'|'never', maxBytes, sampleLimit}`
 */
export function parseWebm(u8, opts = {}) {
  const r = new Reader(u8);
  const out = {
    format: 'webm',
    doctype: '',
    durationMs: 0,
    timecodeScale: 1000000,
    tracks: [],
    video: null,
    audio: null,
    width: 0,
    height: 0,
    codec: '',
    frameCount: 0,
    fps: 0,
    hasAlpha: false,
    interlaced: false,
    lacing: false,
    clusters: 0,
    keyframeCount: 0,
    partial: false,
    seekable: false,
    muxer: '',
    title: '',
    blocks: null,
  };
  let firstId = r.readId();
  if (!firstId || firstId.value !== ID.EBML) {
    // Some files start with padding/voids; skip forward to a recognizable ID.
    let guard = 0;
    while (guard++ < 64 && !r.eof) {
      const at = r.pos;
      const id = r.readId();
      if (id && id.value === ID.EBML) {
        firstId = id;
        break;
      }
      r.pos = at + 1;
    }
    if (!firstId || firstId.value !== ID.EBML) throw new GifxError('not an EBML/WebM file (no EBML header)', { code: ErrorCode.DEMUX_PARSE, details: { first: firstId?.value?.toString(16) } });
  }
  {
    const sz = r.readVint();
    if (!sz) throw new GifxError('truncated EBML header', { code: ErrorCode.DEMUX_PARSE });
    const end = Math.min(u8.length, r.pos + sz.value);
    while (r.pos < end) {
      const id = r.readId();
      if (!id) break;
      const s = r.readVint();
      if (!s) break;
      if (id.value === ID.DocType) out.doctype = r.readString(s.value);
      else r.pos += s.value;
    }
    r.pos = end;
  }
  const seg = r.readId();
  if (!seg || seg.value !== ID.Segment) {
    out.partial = true;
  } else {
    const sz = r.readVint();
    const unknown = !sz || sz.value === 2 ** (7 * sz.bytes) - 1;
    const segEnd = unknown ? u8.length : Math.min(u8.length, r.pos + sz.value);
    parseSegment(r, out, segEnd, opts);
    if (!unknown && segEnd < u8.length) out.partial = true;
  }
  if (out.video) {
    const v = out.video;
    out.width = v.displayWidth || v.width;
    out.height = v.displayHeight || v.height;
    out.codec = v.codecId;
    out.hasAlpha = !!v.alphaMode || !!v.blockAdditionMapping;
    out.interlaced = !!v.flagInterlaced;
    if (v.defaultDuration) {
      out.fps = 1e9 / v.defaultDuration;
      out.frameCount = out.durationMs ? Math.round(out.durationMs / (v.defaultDuration / 1e6)) : 0;
    }
    if (out.blocks && out.blocks.length) {
      out.frameCount = out.blocks.length;
      if (out.durationMs > 0) out.fps = (out.blocks.length * 1000) / out.durationMs;
      out.keyframeCount = out.blocks.reduce((a, b) => a + (b.key ? 1 : 0), 0);
    }
  }
  return out;
}

function parseSegment(r, out, segEnd, opts) {
  const scanBlocks = opts.scanBlocks || 'auto';
  let blocksFromCues = null;
  let cueEntries = 0;
  let scanned = 0;
  const clusterLimit = opts.maxClusters || 4000;
  const byteBudget = opts.maxBytes || 64 * 1024 * 1024;
  while (r.pos < segEnd && !r.eof) {
    const id = r.readId();
    if (!id) break;
    const sz = r.readVint();
    if (!sz) {
      out.partial = true;
      break;
    }
    const unknown = sz.value === 2 ** (7 * sz.bytes) - 1;
    const start = r.pos;
    const end = unknown ? segEnd : Math.min(segEnd, start + sz.value);
    switch (id.value) {
      case ID.Info:
        parseInfo(r, out, end);
        break;
      case ID.Tracks:
        parseTracks(r, out, end);
        break;
      case ID.Cues:
        blocksFromCues = parseCues(r, end);
        cueEntries = blocksFromCues ? blocksFromCues.length : 0;
        break;
      case ID.Cluster: {
        out.clusters++;
        if (scanBlocks !== 'never' && (!blocksFromCues || scanBlocks === 'always') && out.clusters <= clusterLimit && scanned < byteBudget) {
          scanned += parseCluster(r, out, end, start);
        }
        break;
      }
      case ID.SeekHead:
        out.seekHead = parseSeekHead(r, end);
        break;
      case ID.Tags:
        break;
      case ID.Void:
      case ID.Padding:
        break;
      default:
        break;
    }
    r.pos = unknown ? start : end;
    if (!unknown && r.pos <= start) r.pos = end;
    if (r.pos >= segEnd) break;
  }
  // Cues are always worth reporting when present (they are the seek index), and
  // `blocksFromCues` fills the gap when cluster scanning was skipped/capped.
  if (blocksFromCues && blocksFromCues.length) {
    out.cues = blocksFromCues;
    out.seekable = true;
    if (!out.blocks || !out.blocks.length) {
      out.blocks = blocksFromCues.map((c, i) => ({ pts: c.time, index: i, key: true, clusterPosition: c.clusterPosition, fromCue: true }));
    }
  }
  void cueEntries;
  if (out.blocks && out.blocks.length) out.blocks.sort((a, b) => a.pts - b.pts || a.index - b.index);
}

function parseInfo(r, out, end) {
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    switch (id.value) {
      case ID.TimecodeScale:
        out.timecodeScale = r.readUInt(s.value);
        break;
      case ID.Duration:
        out.durationMs = r.readFloat(s.value) * (out.timecodeScale / 1e6);
        break;
      case ID.MuxingApp:
        out.muxer = r.readString(s.value);
        break;
      case ID.Title:
        out.title = r.readString(s.value);
        break;
      default:
        break;
    }
    r.pos = e;
  }
}

function parseTracks(r, out, end) {
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.TrackEntry) {
      const t = parseTrackEntry(r, e, out);
      out.tracks.push(t);
      if (t.trackType === 1 && !out.video) out.video = t;
      else if (t.trackType === 2 && !out.audio) out.audio = t;
    }
    r.pos = e;
  }
}

function parseTrackEntry(r, end, out) {
  const t = { trackNumber: 1, trackType: 0, codecId: '', defaultDuration: 0, width: 0, height: 0, flagLacing: true, alphaMode: false };
  let pendingVideo = null;
  let pendingColour = null;
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    switch (id.value) {
      case ID.TrackNumber:
        t.trackNumber = r.readUInt(s.value);
        break;
      case ID.TrackType:
        t.trackType = r.readUInt(s.value);
        break;
      case ID.CodecID:
        t.codecId = r.readString(s.value);
        break;
      case ID.CodecPrivate:
        t.codecPrivate = r.readBinary(s.value).slice();
        break;
      case ID.DefaultDuration:
        t.defaultDuration = r.readUInt(s.value);
        break;
      case ID.Language:
        t.language = r.readString(s.value);
        break;
      case ID.Name:
        t.name = r.readString(s.value);
        break;
      case ID.FlagLacing:
        t.flagLacing = r.readUInt(s.value) !== 0;
        if (t.flagLacing) out.lacing = true;
        break;
      case ID.Video:
        pendingVideo = { at: r.pos, end: e };
        break;
      case ID.BlockAdditionMapping:
        t.blockAdditionMapping = true;
        break;
      case ID.Projection: {
        const p = parseProjection(r, e);
        if (p) t.projection = p;
        break;
      }
      case ID.Colour:
        pendingColour = { at: r.pos, end: e };
        break;
      default:
        break;
    }
    r.pos = e;
  }
  if (pendingVideo) {
    r.pos = pendingVideo.at;
    parseVideo(r, pendingVideo.end, t);
    r.pos = pendingVideo.end;
  }
  if (pendingColour) {
    r.pos = pendingColour.at;
    t.colour = parseColour(r, pendingColour.end);
    r.pos = pendingColour.end;
  }
  // cropping then display size (the number a player actually shows)
  if (t.cropLeft || t.cropRight || t.cropTop || t.cropBottom) {
    t.width = Math.max(1, t.width - (t.cropLeft || 0) - (t.cropRight || 0));
    t.height = Math.max(1, t.height - (t.cropTop || 0) - (t.cropBottom || 0));
  }
  t.pixelAspectRatio = t.width && t.height && t.displayWidth && t.displayHeight ? t.displayWidth / t.height === 0 ? 1 : (t.displayWidth / t.width) * (t.height / t.displayHeight) : 1;
  return t;
}

function parseVideo(r, end, t) {
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    switch (id.value) {
      case ID.PixelWidth:
        t.width = r.readUInt(s.value);
        break;
      case ID.PixelHeight:
        t.height = r.readUInt(s.value);
        break;
      case ID.DisplayWidth:
        t.displayWidth = r.readUInt(s.value);
        break;
      case ID.DisplayHeight:
        t.displayHeight = r.readUInt(s.value);
        break;
      case ID.CropLeft:
        t.cropLeft = r.readUInt(s.value);
        break;
      case ID.CropRight:
        t.cropRight = r.readUInt(s.value);
        break;
      case ID.CropTop:
        t.cropTop = r.readUInt(s.value);
        break;
      case ID.CropBottom:
        t.cropBottom = r.readUInt(s.value);
        break;
      case ID.FlagInterlaced:
        t.flagInterlaced = r.readUInt(s.value) !== 0;
        break;
      case ID.FieldOrder:
        t.fieldOrder = r.readUInt(s.value);
        break;
      case ID.StereoMode:
        t.stereoMode = r.readUInt(s.value);
        break;
      case ID.AlphaMode:
        t.alphaMode = r.readUInt(s.value) !== 0;
        break;
      case ID.BitDepth:
        t.bitDepth = r.readUInt(s.value);
        break;
      case ID.MatrixCoefficients:
        t.colourMatrix = r.readUInt(s.value);
        break;
      default:
        break;
    }
    r.pos = e;
  }
  if (!t.displayWidth) t.displayWidth = t.width;
  if (!t.displayHeight) t.displayHeight = t.height;
}

function parseColour(r, end) {
  const c = {};
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.ColourPrimaries) c.primaries = r.readUInt(s.value);
    else if (id.value === ID.TransferCharacteristics) c.transfer = r.readUInt(s.value);
    else if (id.value === ID.MatrixCoefficients) c.matrix = r.readUInt(s.value);
    r.pos = e;
  }
  return c;
}

function parseProjection(r, end) {
  const p = {};
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.ProjectionType) p.type = r.readUInt(s.value);
    r.pos = e;
  }
  return p;
}

function parseCues(r, end) {
  const list = [];
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.CuePoint) {
      const cue = { time: 0, clusterPosition: 0, relativePosition: 0, track: 1 };
      while (r.pos < e) {
        const cid = r.readId();
        if (!cid) break;
        const cs = r.readVint();
        if (!cs) break;
        const ce = r.pos + cs.value;
        if (cid.value === ID.CueTime) cue.time = r.readUInt(cs.value);
        else if (cid.value === ID.CueTrackPositions) {
          while (r.pos < ce) {
            const tid = r.readId();
            if (!tid) break;
            const ts = r.readVint();
            if (!ts) break;
            const te = r.pos + ts.value;
            if (tid.value === ID.CueTrack) cue.track = r.readUInt(ts.value);
            else if (tid.value === ID.CueClusterPosition) cue.clusterPosition = r.readUInt(ts.value);
            else if (tid.value === ID.CueRelativePosition) cue.relativePosition = r.readUInt(ts.value);
            r.pos = te;
          }
        }
        r.pos = ce;
      }
      list.push(cue);
    }
    r.pos = e;
  }
  return list;
}

function parseSeekHead(r, end) {
  const list = [];
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === 0x4dbb) {
      const seek = { id: 0, position: 0 };
      while (r.pos < e) {
        const sid = r.readId();
        if (!sid) break;
        const ss = r.readVint();
        if (!ss) break;
        const se = r.pos + ss.value;
        if (sid.value === 0x53ab) seek.id = r.readUInt(ss.value);
        else if (sid.value === 0x53ac) seek.position = r.readUInt(ss.value);
        r.pos = se;
      }
      list.push(seek);
    }
    r.pos = e;
  }
  return list;
}

function parseCluster(r, out, clusterEnd, clusterStart) {
  let clusterTime = 0;
  let count = 0;
  if (!out.blocks) out.blocks = [];
  const limit = out.blocks.length + (out.opts && out.opts.sampleLimit ? out.opts.sampleLimit : 1e6);
  while (r.pos < clusterEnd && r.pos < clusterEnd) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = Math.min(clusterEnd, r.pos + s.value);
    if (id.value === ID.Timecode) {
      clusterTime = r.readUInt(s.value);
    } else if (id.value === ID.SimpleBlock) {
      const track = r.readVint(4)?.value || 1;
      const rel = r.readInt(2);
      const flagsByte = r.pos < e ? r.u8[r.pos++] : 0;
      out.blocks.push({ pts: clusterTime + rel, index: out.blocks.length, key: (flagsByte & 0x80) !== 0, invisible: (flagsByte & 0x08) !== 0, laced: (flagsByte & 0x06) >> 1, track, offset: clusterStart, dataStart: r.pos, size: e - r.pos, simple: true });
      count++;
    } else if (id.value === ID.BlockGroup) {
      const b = parseBlockGroup(r, e, out, clusterTime, clusterStart);
      if (b) {
        out.blocks.push(b);
        count++;
      }
    }
    r.pos = e;
    if (out.blocks.length >= limit) break;
  }
  return count;
}

function parseBlockGroup(r, end, out, clusterTime, clusterStart) {
  let block = null;
  let duration = 0;
  let key = true;
  let additional = null;
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.Block) {
      const track = r.readVint(4)?.value || 1;
      const rel = r.readInt(2);
      const flags = r.pos < e ? r.u8[r.pos++] : 0;
      block = { pts: clusterTime + rel, index: out.blocks ? out.blocks.length : 0, key: key && (flags & 0x80) !== 0 || (flags & 0x80) !== 0, track, offset: clusterStart, dataStart: r.pos, size: e - r.pos, simple: false };
    } else if (id.value === ID.ReferenceBlock) {
      r.readInt(s.value);
    } else if (id.value === ID.Keyframe) {
      key = r.readUInt(s.value) !== 0;
      if (block) block.key = key;
    } else if (id.value === ID.BlockDuration) {
      duration = r.readUInt(s.value);
      if (block) block.duration = duration;
    } else if (id.value === ID.BlockAdditions) {
      additional = parseBlockAdditions(r, e);
    }
    r.pos = e;
  }
  if (block && additional) block.additional = additional;
  if (block && duration && out.video?.defaultDuration) block.duration = duration * (out.timecodeScale / 1e6);
  return block;
}

function parseBlockAdditions(r, end) {
  let out = null;
  while (r.pos < end) {
    const id = r.readId();
    if (!id) break;
    const s = r.readVint();
    if (!s) break;
    const e = r.pos + s.value;
    if (id.value === ID.BlockMore) {
      let idnum = 1;
      while (r.pos < e) {
        const mid = r.readId();
        if (!mid) break;
        const ms = r.readVint();
        if (!ms) break;
        const me = r.pos + ms.value;
        if (mid.value === ID.BlockAddID) idnum = r.readUInt(ms.value);
        else if (mid.value === ID.BlockAdditional) out = { id: idnum, data: r.readBinary(ms.value).slice() };
        r.pos = me;
      }
    }
    r.pos = e;
  }
  return out;
}

/**
 * Map a WebM codec ID to something `VideoDecoder.isConfigSupported` accepts, and
 * produce the config object. VP8/VP9/AV1 in WebM use *IVF-style* framing: a
 * partition/packet as-is (no length prefixes), which is what EncodedVideoChunk wants.
 */
export function webmDecoderConfig(video) {
  if (!video) return null;
  const id = String(video.codecId || '').toUpperCase();
  let codec = id;
  if (id === 'V_VP8') codec = 'vp8';
  else if (id === 'V_VP9') codec = video.profile != null ? `vp09.${String(video.profile).padStart(2, '0')}.10.08` : 'vp09.00.10.08';
  else if (id === 'V_AV1') codec = 'av01.0.01M.0';
  else if (id === 'V_MPEG4/ISO/AVC') codec = 'avc1.42001e';
  else if (id === 'V_MS/VFW/FOURCC') codec = '';
  const cfg = { codec, description: video.codecPrivate || undefined };
  if (video.codecPrivate && (id === 'V_AV1' || id === 'V_VP9')) {
    // AV1 in WebM stores an OBU sequence header; VP9 stores nothing useful.
    if (id === 'V_AV1') cfg.description = video.codecPrivate;
  }
  return { codec: cfg.codec, codedWidth: video.width, codedHeight: video.height, description: cfg.description };
}

/** Extract block bytes for a sample (with its alpha if BlockAdditions carried it). */
export function webmBlockBytes(u8, block) {
  return u8.subarray(block.dataStart, block.dataStart + block.size);
}

export { ID as WEBM_IDS };

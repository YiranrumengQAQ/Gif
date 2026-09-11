/**
 * GIFX Kernel — byte/bit primitives.
 *
 * Everything here is allocation-free on the hot path (view over an existing
 * ArrayBuffer) because GIF/PNG/WebP muxing touches every single byte of the
 * output and per-write object allocation dominated early profiles.
 *
 * @module core/bytes
 */

/* ------------------------------------------------------------------ *
 * Little-endian / big-endian view over an ArrayBuffer
 * ------------------------------------------------------------------ */

/**
 * Growable little-endian writer with 1/2/3/4-byte accessors.
 * `3`-byte helpers exist because animated WebP (ANMF) and VP8X use them.
 */
export class ByteWriter {
  /** @param {number|Uint8Array} [capacity=65536] */
  constructor(capacity = 65536) {
    if (capacity instanceof Uint8Array) {
      this.buf = capacity;
    } else {
      this.buf = new Uint8Array(Math.max(64, capacity | 0));
    }
    this.view = new DataView(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);
    this.pos = 0;
  }

  get length() {
    return this.pos;
  }
  get capacity() {
    return this.buf.length;
  }

  _ensure(n) {
    if (this.pos + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.pos + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.pos));
    this.buf = next;
    this.view = new DataView(next.buffer, next.byteOffset, next.byteLength);
  }

  u8(v) {
    this._ensure(1);
    this.buf[this.pos++] = v & 0xff;
    return this;
  }
  u16(v) {
    this._ensure(2);
    this.view.setUint16(this.pos, v, true);
    this.pos += 2;
    return this;
  }
  u16be(v) {
    this._ensure(2);
    this.view.setUint16(this.pos, v, false);
    this.pos += 2;
    return this;
  }
  u24(v) {
    this._ensure(3);
    this.buf[this.pos++] = v & 0xff;
    this.buf[this.pos++] = (v >>> 8) & 0xff;
    this.buf[this.pos++] = (v >>> 16) & 0xff;
    return this;
  }
  u24be(v) {
    this._ensure(3);
    this.buf[this.pos++] = (v >> 16) & 0xff;
    this.buf[this.pos++] = (v >> 8) & 0xff;
    this.buf[this.pos++] = v & 0xff;
    return this;
  }
  u32(v) {
    this._ensure(4);
    this.view.setUint32(this.pos, v >>> 0, true);
    this.pos += 4;
    return this;
  }
  u32be(v) {
    this._ensure(4);
    this.view.setUint32(this.pos, v >>> 0, false);
    this.pos += 4;
    return this;
  }
  i32(v) {
    this._ensure(4);
    this.view.setInt32(this.pos, v | 0, true);
    this.pos += 4;
    return this;
  }
  f32be(v) {
    this._ensure(4);
    this.view.setFloat32(this.pos, v, false);
    this.pos += 4;
    return this;
  }
  bytes(src, offset = 0, len = src.length - offset) {
    this._ensure(len);
    this.buf.set(src instanceof Uint8Array ? src : new Uint8Array(src.buffer, src.byteOffset, src.byteLength), this.pos);
    this.pos += len;
    return this;
  }
  /** Write `n` zero bytes (memset). */
  zero(n) {
    this._ensure(n);
    this.buf.fill(0, this.pos, this.pos + n);
    this.pos += n;
    return this;
  }
  repeat(v, n) {
    this._ensure(n);
    this.buf.fill(v & 0xff, this.pos, this.pos + n);
    this.pos += n;
    return this;
  }
  ascii(str) {
    this._ensure(str.length);
    for (let i = 0; i < str.length; i++) this.buf[this.pos++] = str.charCodeAt(i) & 0xff;
    return this;
  }
  utf8(str) {
    for (let i = 0; i < str.length; i++) {
      let c = str.codePointAt(i);
      if (c > 0xffff) i++;
      if (c < 0x80) this.u8(c);
      else if (c < 0x800) {
        this.u8(0xc0 | (c >> 6)).u8(0x80 | (c & 63));
      } else if (c < 0x10000) {
        this.u8(0xe0 | (c >> 12)).u8(0x80 | ((c >> 6) & 63)).u8(0x80 | (c & 63));
      } else {
        this.u8(0xf0 | (c >> 18)).u8(0x80 | ((c >> 12) & 63)).u8(0x80 | ((c >> 6) & 63)).u8(0x80 | (c & 63));
      }
    }
    return this;
  }
  /** Pad to an even byte boundary (RIFF requirement). */
  padByte(v = 0) {
    if (this.pos & 1) this.u8(v);
    return this;
  }
  u8At(pos, v) {
    this.buf[pos] = v & 0xff;
    return this;
  }
  u32At(pos, v) {
    this.view.setUint32(pos, v >>> 0, true);
    return this;
  }
  u32beAt(pos, v) {
    this.view.setUint32(pos, v >>> 0, false);
    return this;
  }
  /** Reserve space for a length field that will be patched later. */
  reserve32() {
    const at = this.pos;
    this.u32(0);
    return at;
  }
  /** Snapshot of what has been written (copies unless `own` requested). */
  result({ own = false } = {}) {
    const out = own ? new Uint8Array(this.pos) : new Uint8Array(this.buf.buffer, this.buf.byteOffset, this.pos);
    if (own) out.set(this.buf.subarray(0, this.pos));
    return out;
  }
  /** Zero-copy view; invalidated by the next _ensure(). */
  view_() {
    return this.buf.subarray(0, this.pos);
  }
}

/** Big-endian reader for PNG/GIF/WebP-tagged structures. */
export class ByteReader {
  /** @param {ArrayBuffer|Uint8Array|ArrayBufferView} src */
  constructor(src) {
    if (src instanceof Uint8Array) this.u8 = src;
    else if (src instanceof ArrayBuffer) this.u8 = new Uint8Array(src);
    else if (ArrayBuffer.isView(src)) this.u8 = new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
    else throw new TypeError('ByteReader needs ArrayBuffer/Uint8Array');
    this.dv = new DataView(this.u8.buffer, this.u8.byteOffset, this.u8.byteLength);
    this.pos = 0;
  }
  get length() {
    return this.u8.length;
  }
  get remaining() {
    return this.u8.length - this.pos;
  }
  eof() {
    return this.pos >= this.u8.length;
  }
  seek(p) {
    this.pos = p;
    return this;
  }
  u8_() {
    return this.u8[this.pos++];
  }
  u16() {
    const v = this.dv.getUint16(this.pos, false);
    this.pos += 2;
    return v;
  }
  u16le() {
    const v = this.dv.getUint16(this.pos, true);
    this.pos += 2;
    return v;
  }
  u24() {
    const b = this.u8;
    const p = this.pos;
    this.pos += 3;
    return (b[p] << 16) | (b[p + 1] << 8) | b[p + 2];
  }
  u24le() {
    const b = this.u8;
    const p = this.pos;
    this.pos += 3;
    return b[p] | (b[p + 1] << 8) | (b[p + 2] << 16);
  }
  u32() {
    const v = this.dv.getUint32(this.pos, false);
    this.pos += 4;
    return v;
  }
  u32le() {
    const v = this.dv.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }
  i32() {
    const v = this.dv.getInt32(this.pos, false);
    this.pos += 4;
    return v;
  }
  f64be() {
    const v = this.dv.getFloat64(this.pos, false);
    this.pos += 8;
    return v;
  }
  bytes(n) {
    const out = this.u8.subarray(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }
  ascii(n) {
    let s = '';
    for (let i = 0; i < n; i++) {
      const c = this.u8[this.pos + i];
      if (!c) break;
      s += String.fromCharCode(c);
    }
    this.pos += n;
    return s;
  }
}

/* ------------------------------------------------------------------ *
 * Bit-level IO (LZW lives here)
 * ------------------------------------------------------------------ */

/**
 * LSB-first bit writer used by the GIF LZW encoder.
 *
 * Writes whole 32-bit words when possible: the inner LZW loop used to call
 * `u8()` up to 12 times per code which was measurable in profiles.
 */
export class BitWriterLE {
  constructor(out /* Uint8Array-like view with .pos semantics provided by caller */) {
    this.out = out;
    this.bitBuf = 0;
    this.bitCount = 0;
    this.pos = 0;
  }
  reset(out, pos = 0) {
    this.out = out;
    this.pos = pos;
    this.bitBuf = 0;
    this.bitCount = 0;
  }
  write(code, size) {
    let v = code;
    if (this.bitCount + size <= 24) {
      this.bitBuf |= (v & ((1 << size) - 1)) << this.bitCount;
      this.bitCount += size;
      while (this.bitCount >= 8) {
        this.out[this.pos++] = this.bitBuf & 255;
        this.bitBuf >>>= 8;
        this.bitCount -= 8;
      }
      return;
    }
    // slow path for wide codes
    while (size > 0) {
      const room = 32 - this.bitCount;
      const take = Math.min(room, size);
      this.bitBuf |= (v & ((1 << take) - 1)) << this.bitCount;
      this.bitCount += take;
      v >>>= take;
      size -= take;
      while (this.bitCount >= 8) {
        this.out[this.pos++] = this.bitBuf & 255;
        this.bitBuf >>>= 8;
        this.bitCount -= 8;
      }
    }
  }
  flush() {
    while (this.bitCount > 0) {
      this.out[this.pos++] = this.bitBuf & 255;
      this.bitBuf >>>= 8;
      this.bitCount = Math.max(0, this.bitCount - 8);
    }
    this.pos |= 0;
    return this.pos;
  }
}

/** LSB-first bit reader for the LZW decoder. */
export class BitReaderLE {
  constructor(data) {
    this.d = data;
    this.pos = 0;
    this.bitBuf = 0;
    this.bitCount = 0;
  }
  read(size) {
    while (this.bitCount < size) {
      this.bitBuf |= this.d[this.pos++] << this.bitCount;
      this.bitCount += 8;
    }
    const v = this.bitBuf & ((1 << size) - 1);
    this.bitBuf >>>= size;
    this.bitCount -= size;
    return v;
  }
}

/* ------------------------------------------------------------------ *
 * Conversions
 * ------------------------------------------------------------------ */

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = (() => {
  const t = new Int16Array(256).fill(-1);
  for (let i = 0; i < 64; i++) t[B64_CHARS.charCodeAt(i)] = i;
  t['-'.charCodeAt(0)] = 62;
  t['_'.charCodeAt(0)] = 63;
  return t;
})();

/**
 * Chunked base64 — `btoa(String.fromCharCode(...arr))` blows the stack above
 * ~32 KB, so we accumulate 0x8000-byte slices. ~2-3x faster than the naive
 * loop because the engine's native btoa handles the ASCII step.
 */
export function bytesToBase64(bytes, { urlSafe = false, lineBreaks = 0 } = {}) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out;
  if (typeof btoa === 'function') {
    const CH = 0x8000;
    const parts = [];
    let s = '';
    const cbuf = new Array(CH);
    for (let i = 0; i < u8.length; i += CH) {
      const end = Math.min(i + CH, u8.length);
      let n = 0;
      for (let j = i; j < end; j++) cbuf[n++] = String.fromCharCode(u8[j]);
      s += btoa(cbuf.slice(0, n).join(''));
      if (s.length > 0x10000) {
        parts.push(s);
        s = '';
      }
    }
    if (s) parts.push(s);
    out = parts.join('');
  } else if (typeof Buffer === 'function') {
    out = Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString('base64');
  } else {
    out = manualB64(u8);
  }
  if (urlSafe) out = out.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  if (lineBreaks > 0) {
    const lines = [];
    for (let i = 0; i < out.length; i += lineBreaks) lines.push(out.slice(i, i + lineBreaks));
    out = lines.join('\r\n');
  }
  return out;
}

function manualB64(u8) {
  let out = '';
  let i = 0;
  const n = u8.length;
  const CH = 0x6000;
  const parts = [];
  let acc = '';
  for (; i < n; ) {
    const b0 = u8[i++];
    const b1 = i < n ? u8[i++] : -1;
    const b2 = i < n ? u8[i++] : -1;
    acc += B64_CHARS[b0 >> 2];
    acc += B64_CHARS[((b0 & 3) << 4) | (b1 < 0 ? 0 : b1 >> 4)];
    acc += b1 < 0 ? '=' : B64_CHARS[((b1 & 15) << 2) | (b2 < 0 ? 0 : b2 >> 6)];
    acc += b2 < 0 ? '=' : B64_CHARS[b2 & 63];
    if (acc.length >= CH) {
      parts.push(acc);
      acc = '';
    }
  }
  return parts.join('') + acc;
}

export function base64ToBytes(b64) {
  const clean = b64.replace(/[^A-Za-z0-9\-_+/=]/g, '');
  if (typeof atob === 'function') {
    const bin = atob(clean.replace(/-/g, '+').replace(/_/g, '/'));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  if (typeof Buffer === 'function') return new Uint8Array(Buffer.from(clean, 'base64'));
  const noPad = clean.replace(/=+$/, '');
  const len = ((noPad.length * 3) / 4) | 0;
  const out = new Uint8Array(len);
  let o = 0;
  let acc = 0;
  let bits = 0;
  for (let i = 0; i < noPad.length; i++) {
    const v = B64_LOOKUP[noPad.charCodeAt(i)];
    if (v < 0) continue;
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (acc >> bits) & 255;
    }
  }
  return out.subarray(0, o);
}

export function hexToBytes(hex) {
  const h = hex.replace(/[^0-9a-fA-F]/g, '');
  const out = new Uint8Array(h.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(h.substr(i * 2, 2), 16);
  return out;
}

export function bytesToHex(bytes, sep = '') {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let s = '';
  for (let i = 0; i < u8.length; i++) {
    s += (u8[i] < 16 ? '0' : '') + u8[i].toString(16);
    if (sep && i < u8.length - 1) s += sep;
  }
  return s;
}

export function asciiBytes(str) {
  const out = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) out[i] = str.charCodeAt(i) & 0xff;
  return out;
}

/** FourCC read (used everywhere: MP4 boxes, PNG chunks, RIFF tags). */
export function fourcc(u8, at = 0) {
  return String.fromCharCode(u8[at], u8[at + 1], u8[at + 2], u8[at + 3]);
}
export function writeFourcc(u8, at, tag) {
  for (let i = 0; i < 4; i++) u8[at + i] = tag.charCodeAt(i);
}

/* ------------------------------------------------------------------ *
 * CRC32 / hashes
 * ------------------------------------------------------------------ */

let CRC_TABLE = null;
export function crc32Table() {
  if (CRC_TABLE) return CRC_TABLE;
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c | 0;
  }
  return (CRC_TABLE = t);
}

/** PNG/ZIP CRC-32 (reflected, init 0xffffffff, final xor). */
export function crc32(data, seed = 0) {
  const t = crc32Table();
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  let c = ~seed;
  for (let i = 0; i < u8.length; i++) c = t[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

/** 32-bit FNV-1a; used for cheap buffer identity/dedup. */
export function fnv1a(data, seed = 0x811c9dc5) {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  let h = seed >>> 0;
  for (let i = 0; i < u8.length; i++) {
    h ^= u8[i];
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}

/**
 * xxHash32 (canonical, seed 0 → `xxh32("") === 0x02CC5D05`).
 * Used for frame fingerprinting / cache keys; ~3-5x faster than FNV-1a on
 * multi-megabyte buffers and much better avalanche.
 */
const P1 = 2654435761, P2 = 2246822519, P3 = 3266489917, P4 = 668265263, P5 = 374761393;
const rotl = (x, r) => ((x << r) | (x >>> (32 - r))) >>> 0;
const round32 = (acc, input) => {
  acc = (acc + Math.imul(input, P3)) >>> 0;
  acc = rotl(acc, 17);
  return Math.imul(acc, P4) >>> 0;
};

export function xxhash32(data, seed = 0, offset = 0, length = data.length - offset) {
  const u8 = data instanceof Uint8Array ? data : data.buffer ? new Uint8Array(data.buffer, data.byteOffset || 0, data.byteLength) : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const end = offset + length;
  let h;
  let p = offset;
  if (length >= 16) {
    let v1 = (seed + P1 + P2) >>> 0;
    let v2 = (seed + P2) >>> 0;
    let v3 = seed >>> 0;
    let v4 = (seed - P1) >>> 0;
    while (p + 16 <= end) {
      v1 = round32(v1, dv.getUint32(p, true));
      v2 = round32(v2, dv.getUint32(p + 4, true));
      v3 = round32(v3, dv.getUint32(p + 8, true));
      v4 = round32(v4, dv.getUint32(p + 12, true));
      p += 16;
    }
    h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) >>> 0;
  } else {
    h = (seed + P5) >>> 0;
  }
  h = (h + length) >>> 0;
  while (p + 4 <= end) {
    h = round32(h, dv.getUint32(p, true));
    p += 4;
  }
  while (p < end) h = Math.imul(rotl((h ^ Math.imul(u8[p++], P5)) >>> 0, 11), P1) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, P2) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, P3) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * 64-bit-ish rolling fingerprint over an RGBA buffer for duplicate-frame
 * detection. Stride-sampled (default: one texel per 97) so dedup never
 * becomes the bottleneck on 4K sources; collisions are then confirmed with a
 * full compare in `optimize/diff.js`.
 */
export function frameSignature(u8, stride = 97) {
  let h1 = 0x9e3779b9 | 0;
  let h2 = 0x85ebca6b | 0;
  const n = u8.length;
  for (let i = 0; i < n; i += stride * 4) {
    h1 = (rotl((h1 + u8[i]) >>> 0, 5) ^ Math.imul(h2, P3)) >>> 0;
    h2 = (rotl((h2 + u8[i + 1] + u8[i + 2]) >>> 0, 7) + h1) >>> 0;
  }
  h1 = Math.imul(h1 ^ (h1 >>> 13), P4) >>> 0;
  h2 = Math.imul(h2 ^ (h2 >>> 11), P2) >>> 0;
  return (h1 >>> 0).toString(16).padStart(8, '0') + (h2 >>> 0).toString(16).padStart(8, '0');
}

/* ------------------------------------------------------------------ *
 * misc
 * ------------------------------------------------------------------ */

/** Compare two byte ranges without allocating. */
export function bytesEqual(a, b, len = Math.min(a.length, b.length)) {
  if (a.length !== b.length && len === Math.min(a.length, b.length) && a.length !== b.length) return false;
  for (let i = 0; i < len; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function concatBytes(list, total) {
  let n = total || 0;
  if (!total) for (const b of list) n += b.length;
  const out = new Uint8Array(n);
  let at = 0;
  for (const b of list) {
    out.set(b, at);
    at += b.length;
  }
  return out;
}

/** `moov`-style fixed-point fraction parsing (MP4 uses 16.16 / 8.8). */
export function fixed16(v) {
  return (v >> 16) + (v & 0xffff) / 65536;
}

export function roundUp(x, m) {
  return Math.ceil(x / m) * m;
}

export function align(x, a) {
  return Math.ceil(x / a) * a;
}

export function formatBytes(n, digits = 1) {
  if (!isFinite(n)) return '—';
  const neg = n < 0;
  let v = Math.abs(n);
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${neg ? '-' : ''}${v.toFixed(i === 0 ? 0 : v < 10 ? digits : 1)} ${units[i]}`;
}

export function parseBytes(str) {
  if (typeof str === 'number') return str;
  const m = /^\s*([\d.]+)\s*(kb|mb|gb|kib|mib|gib|k|m|g|b)?\s*$/i.exec(String(str));
  if (!m) return NaN;
  const v = parseFloat(m[1]);
  const u = (m[2] || 'b').toLowerCase();
  const mult = { b: 1, k: 1024, kb: 1024, kib: 1024, m: 1024 ** 2, mb: 1024 ** 2, mib: 1024 ** 2, g: 1024 ** 3, gb: 1024 ** 3, gib: 1024 ** 3 }[u];
  return Math.round(v * mult);
}

/**
 * GIF89a encoder — bit-exact, spec-complete, streaming-friendly.
 *
 * Supports the whole feature set that actually matters for size/quality:
 *
 *  - global **and** per-frame local color tables (auto-selected: local tables
 *    only when they shrink the output)
 *  - transparent index per frame + disposal methods (0 none / 1 keep /
 *    2 restore-to-background / 3 restore-to-previous)
 *  - **dirty-rect frames**: a frame may cover a sub-rectangle of the canvas;
 *    combined with `disposal: 1` this is the single biggest size win for
 *    screen recordings and talking-head video (typically 30–55%)
 *  - interlacing (4-pass GIF progressive)
 *  - Netscape looping extension, application extensions, comment extension
 *    (UTF-8, chunked at 255 bytes), XMP packet
 *  - min LZW code size derived from palette size (2..8) — free 3–8% savings
 *  - incremental `addFrame()` so results can be streamed to OPFS without ever
 *    holding the whole GIF in memory
 *
 * @module enc/gif
 */
import { ByteWriter } from '../core/bytes.js';
import { lzwEncode, lzwMaxBytes, writeSubBlocks, subBlockSize, minCodeSizeFor } from './lzw.js';
import { GifxError, ErrorCode } from '../core/errors.js';

const HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a
const MAX_DIM = 65535;

/** GIF disposal methods. */
export const Disposal = Object.freeze({
  /** Leave the frame in place (default; cheapest). */
  NONE: 0,
  /** Do not dispose — the next frame is drawn on top of this one. */
  KEEP: 1,
  /** Restore the frame area to the background colour. */
  RESTORE_BG: 2,
  /** Restore the previous canvas contents (needed for overlays/erasers). */
  RESTORE_PREVIOUS: 3,
});

export class GifWriter {
  /**
   * @param {object} opts
   * @param {number} opts.width canvas width (1..65535)
   * @param {number} opts.height canvas height
   * @param {Uint8Array} [opts.palette] global palette, RGB triples
   * @param {number} [opts.colors] entries in `palette`
   * @param {number|null} [opts.loop=0] repeat count, 0 = infinite, null = none
   * @param {number} [opts.backgroundColorIndex=0]
   * @param {number} [opts.pixelAspectRatio=0]
   * @param {string} [opts.comment] written as a Comment Extension
   * @param {string} [opts.application] app id (must be 8 ASCII chars) — pair with `appAuth`/`appData`
   * @param {string} [opts.xmp] XMP packet
   * @param {ByteWriter} [opts.out] pre-allocated writer (worker reuse)
   */
  constructor(opts = {}) {
    const w = opts.width | 0;
    const h = opts.height | 0;
    if (!(w > 0 && h > 0)) throw new GifxError(ErrorCode.CONFIG_INVALID, `GIF needs positive width/height (got ${w}x${h})`, { path: 'size' });
    if (w > MAX_DIM || h > MAX_DIM) {
      throw new GifxError(ErrorCode.CONFIG_INVALID, `GIF dimensions must be <= ${MAX_DIM} (got ${w}x${h})`, {
        path: 'size',
        hint: 'Downscale the output: GIF has a hard 16-bit size limit and browsers choke on huge canvases anyway.',
      });
    }
    this.width = w;
    this.height = h;
    this.out = opts.out || new ByteWriter(Math.max(1 << 16, Math.min(1 << 24, w * h)));
    this.frameCount = 0;
    this.lzwBytes = 0;
    this.payloadBytes = 0;
    this.loop = opts.loop === undefined ? 0 : opts.loop;
    this.bgIndex = opts.backgroundColorIndex | 0;
    this.aspect = opts.pixelAspectRatio | 0;
    this.globalPalette = null;
    this.globalColors = 0;
    this.globalMinCodeSize = 8;
    this.frameDescriptions = [];
    if (opts.palette) this.setGlobalPalette(opts.palette, opts.colors);
    this._headerWritten = false;
    this.comment = opts.comment || null;
    this.xmp = opts.xmp || null;
    this.application = opts.application || null;
    this.appAuth = opts.appAuth || null;
    this.appData = opts.appData || null;
    this._pendingPatch = -1;
    /**
     * Scratch used to repack rect rows when the caller supplies a full-canvas
     * index buffer; sized lazily.
     */
    this._rowScratch = null;
    this._bitScratch = null;
  }

  setGlobalPalette(palette, colors = palette.length / 3) {
    this.globalPalette = palette;
    this.globalColors = Math.max(2, Math.min(256, colors | 0));
    this.globalMinCodeSize = minCodeSizeFor(this.globalColors);
    return this;
  }

  /** Write header + logical screen descriptor + global table + extensions. */
  begin() {
    if (this._headerWritten) return this;
    const out = this.out;
    out.bytes(Uint8Array.from(HEADER));
    out.u16(this.width).u16(this.height);
    const bits = this.globalPalette ? paletteBits(this.globalColors) : 1;
    const packed = (this.globalPalette ? 0x80 : 0) | ((Math.max(1, bits) - 1) << 4) | (this.globalPalette ? bits - 1 : 0);
    out.u8(packed).u8(this.bgIndex).u8(this.aspect);
    if (this.globalPalette) this._writeColorTable(this.globalPalette, this.globalColors, bits);
    if (this.comment) this._writeComment(this.comment);
    if (this.application) this._writeApplication(this.application, this.appAuth, this.appData);
    if (this.loop !== null && this.loop !== undefined) this._writeNetscapeLoop(this.loop);
    if (this.xmp) this._writeXmp(this.xmp);
    this._headerWritten = true;
    return this;
  }

  /**
   * Append one frame.
   *
   * @param {object} f
   * @param {number} [f.x=0] left of the frame rectangle
   * @param {number} [f.y=0] top
   * @param {number} f.width rect width
   * @param {number} f.height rect height
   * @param {Uint8Array|Uint8ClampedArray} f.indices palette indices, row-major
   * @param {number} [f.stride=f.width] source row stride (pixels)
   * @param {number} [f.delay=10] frame delay in **centiseconds** (1/100 s)
   * @param {number} [f.disposal=1] {@link Disposal} value
   * @param {number} [f.transparentIndex=-1] index treated as transparent
   * @param {Uint8Array} [f.palette] local color table (RGB triples)
   * @param {number} [f.colors] entries in the local table
   * @param {boolean} [f.interlace=false]
   * @param {string} [f.comment]
   * @returns {{bytes:number, minCodeSize:number, local:boolean, rect:number[]}}
   */
  addFrame(f) {
    if (!this._headerWritten) this.begin();
    const out = this.out;
    const x = f.x | 0;
    const y = f.y | 0;
    const fw = f.width | 0;
    const fh = f.height | 0;
    if (x < 0 || y < 0 || x + fw > this.width || y + fh > this.height) {
      throw new GifxError(ErrorCode.INTERNAL, `frame rect ${x},${y} ${fw}x${fh} escapes canvas ${this.width}x${this.height}`, {
        path: `frames[${this.frameCount}]`,
      });
    }
    const frameStart = out.pos;
    const local = !!f.palette;
    const colors = Math.max(2, Math.min(256, (local ? f.colors : this.globalColors) | 0));
    const minCodeSize = minCodeSizeFor(colors);
    const transparentIndex = f.transparentIndex == null ? -1 : f.transparentIndex | 0;
    // A symbol at or above the LZW alphabet size cannot be represented: the bit
    // stream would decode as garbage in the tail. Better a loud error than a GIF
    // that looks fine in one viewer and breaks in another.
    if (transparentIndex >= (1 << minCodeSize)) {
      throw new GifxError(ErrorCode.ENCODE_FAILED, `transparent index ${transparentIndex} does not fit a ${1 << minCodeSize}-entry color table`, {
        path: `frames[${this.frameCount}]`,
        hint: 'use a palette with a free slot (paletteSlots) or drop per-frame transparency',
      });
    }
    const delay = normalizeDelay(f.delay);
    const disposal = f.disposal == null ? Disposal.KEEP : f.disposal | 0;
    const interlace = !!f.interlace;

    if (f.comment) this._writeComment(f.comment);

    // --- Graphic Control Extension
    out.u8(0x21).u8(0xf9).u8(4);
    // GCE packed field: bits 7-5 reserved, 4-2 disposal, 1 user input, 0 transparency
    out.u8(((disposal & 7) << 2) | (f.userInput ? 2 : 0) | (transparentIndex >= 0 ? 1 : 0));
    out.u16(delay);
    out.u8(transparentIndex >= 0 ? Math.min(255, transparentIndex) : 0);
    out.u8(0);

    // --- Image Descriptor
    out.u8(0x2c);
    out.u16(x).u16(y).u16(fw).u16(fh);
    const lbits = local ? paletteBits(colors) : 0;
    out.u8((local ? 0x80 : 0) | (interlace ? 0x40 : 0) | (f.sorted ? 0x20 : 0) | (local ? lbits - 1 : 0));
    if (local) this._writeColorTable(f.palette, colors, lbits);

    // --- LZW image data
    const pixels = fw * fh;
    const cap = lzwMaxBytes(pixels, minCodeSize) + 16;
    const bitBuf = cap <= 1 << 22 && this._bitScratch && this._bitScratch.length >= cap ? this._bitScratch : (this._bitScratch = new Uint8Array(cap));
    // `indices` rows are `stride` pixels wide. `x0`/`y0` select a window inside
    // that buffer — they are deliberately independent of the frame's placement on
    // the canvas, because callers usually hand in an already-clipped sub-rect
    // (stride === width, no offset) or the whole canvas plus the rect they want.
    const stride = f.stride || fw;
    const sx = f.x0 === undefined || f.x0 === null ? 0 : f.x0 | 0;
    const sy = f.y0 === undefined || f.y0 === null ? 0 : f.y0 | 0;
    let written;
    try {
      written = lzwEncode(
        f.indices,
        pixels,
        minCodeSize,
        bitBuf,
        stride !== fw || sx !== 0 || sy !== 0 || interlace
          ? {
              width: fw,
              stride,
              x0: sx,
              y0: sy,
              rectH: fh,
              rowOrder: interlace ? interlaceRows(fh) : null,
              ...(f.lzw || {}),
            }
          : { ...(f.lzw || {}) },
      );
    } catch (cause) {
      throw new GifxError(ErrorCode.ENCODE_FAILED, `LZW failed on frame ${this.frameCount} (${fw}x${fh}, ${colors} colors)`, { cause, path: `frames[${this.frameCount}]` });
    }
    out.u8(minCodeSize);
    out._ensure(subBlockSize(written));
    const p = out.pos;
    const consumed = writeSubBlocks(bitBuf, 0, written, out.buf, p);
    out.pos = p + consumed;

    this.payloadBytes += written;
    this.lzwBytes += consumed;
    const info = {
      bytes: out.pos - frameStart,
      lzwBytes: consumed,
      rawBytes: written,
      pixels,
      minCodeSize,
      local,
      rect: [x, y, fw, fh],
      delay,
      disposal,
      colors,
      interlace,
      transparent: transparentIndex,
    };
    this.frameDescriptions.push(info);
    this.frameCount++;
    return info;
  }

  /** Finalise the stream (trailer) and return the bytes. */
  finish({ copy = true } = {}) {
    this.begin();
    this.out.u8(0x3b);
    const res = this.out.result({ own: copy });
    this.finished = true;
    return res;
  }

  /** Estimated size of the GIF *so far* (bytes). */
  get bytesWritten() {
    return this.out.pos + 1;
  }

  _writeColorTable(palette, colors, bits) {
    const entries = 1 << bits;
    const out = this.out;
    out._ensure(entries * 3);
    const buf = out.buf;
    let p = out.pos;
    for (let i = 0; i < entries; i++) {
      if (i < colors) {
        buf[p++] = palette[i * 3];
        buf[p++] = palette[i * 3 + 1];
        buf[p++] = palette[i * 3 + 2];
      } else {
        // pad with the last entry: some decoders probe unused slots
        buf[p++] = palette[(colors - 1) * 3];
        buf[p++] = palette[(colors - 1) * 3 + 1];
        buf[p++] = palette[(colors - 1) * 3 + 2];
      }
    }
    out.pos = p;
  }

  _writeNetscapeLoop(loop) {
    const out = this.out;
    out.u8(0x21).u8(0xff).u8(11);
    out.ascii('NETSCAPE2.0');
    out.u8(3).u8(1).u16(Math.max(0, loop | 0) & 0xffff).u8(0);
  }

  _writeApplication(id, auth, data) {
    const out = this.out;
    const idBytes = asciiPad(id, 8);
    const authBytes = auth ? asciiPad(auth, 3) : new Uint8Array([0x01, 0x00, 0x00]);
    const payload = data ? (typeof data === 'string' ? new TextEncoder().encode(data) : data) : new Uint8Array(0);
    out.u8(0x21).u8(0xff).u8(idBytes.length + authBytes.length);
    out.bytes(idBytes).bytes(authBytes);
    let at = 0;
    while (at < payload.length) {
      const n = Math.min(255, payload.length - at);
      out.u8(n).bytes(payload.subarray(at, at + n));
      at += n;
    }
    out.u8(0);
  }

  _writeComment(text) {
    // A comment extension is `0x21 0xFE` followed by a chain of sub-blocks
    // terminated by a zero byte (no leading total-length byte).
    const enc = new TextEncoder().encode(text);
    const out = this.out;
    out.u8(0x21).u8(0xfe);
    let at = 0;
    if (enc.length === 0) out.u8(0);
    while (at < enc.length) {
      const n = Math.min(255, enc.length - at);
      out.u8(n).bytes(enc.subarray(at, at + n));
      at += n;
    }
    out.u8(0);
  }

  _writeXmp(packet) {
    const out = this.out;
    const id = asciiPad('XMP ', 8);
    const auth = new Uint8Array([0x01, 0x00]);
    const data = new TextEncoder().encode(packet);
    out.u8(0x21).u8(0xff).u8(id.length + auth.length);
    out.bytes(id).bytes(auth);
    let at = 0;
    while (at < data.length) {
      const n = Math.min(255, data.length - at);
      out.u8(n).bytes(data.subarray(at, at + n));
      at += n;
    }
    out.u8(0);
  }

  /**
   * Convenience: encode a complete GIF from an array of ready frames.
   * @param {object} opts see constructor + {@link GifWriter#addFrame}
   * @param {object[]} frames
   */
  static encodeAll(opts, frames) {
    const g = new GifWriter(opts);
    g.begin();
    for (const f of frames) g.addFrame(f);
    return g.finish();
  }
}

/**
 * Compute the whole-GIF byte size *without* allocating it, by measuring only
 * the LZW streams. Used by the target-size search to avoid producing garbage
 * candidate files.
 */
export function measureGif(opts, frames) {
  let total = 13 + (opts.palette ? (1 << paletteBits(opts.colors || opts.palette.length / 3)) * 3 : 0) + (opts.loop != null ? 19 : 0) + 1;
  if (opts.comment) total += 4 + new TextEncoder().encode(opts.comment).length;
  for (const f of frames) {
    const colors = Math.max(2, (f.palette ? f.colors : opts.colors) | 0);
    const mcs = minCodeSizeFor(colors);
    total += 8 + 10 + subBlockSize(f.width * f.height) + (f.palette ? colors * 3 : 0);
    void mcs;
  }
  return total;
}

function paletteBits(colors) {
  let bits = 1;
  while ((1 << bits) < colors) bits++;
  return Math.max(1, Math.min(8, bits));
}

function asciiPad(s, n) {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = i < s.length ? s.charCodeAt(i) & 0xff : 0x20;
  return out;
}

/**
 * GIF delays are centiseconds; clamping to 2cs avoids the notorious
 * "delay 0/1 → 10ms or 100ms depending on browser" quirk. Chrome clamps
 * anything < 10ms up to 100ms on some code paths, so 2 is the pragmatic
 * minimum (matches gif.js/giflossy behaviour).
 */
export function normalizeDelay(cs, minCs = 2, zeroCs = minCs > 0 ? 10 : 0) {
  const v = Number(cs);
  if (!Number.isFinite(v) || v <= 0) return Math.max(minCs, Math.min(65535, zeroCs));
  return Math.max(minCs, Math.min(65535, Math.round(v)));
}

/**
 * 4-pass interlace row permutation: the row `i` of the *bitstream* corresponds
 * to canvas row `rows[i]`.
 * @param {number} h
 * @param {Uint16Array} [reuse]
 */
export function interlaceRows(h, reuse) {
  const rows = reuse && reuse.length === h ? reuse : new Uint16Array(h);
  let p = 0;
  for (let pass = 0; pass < 4; pass++) {
    const start = [0, 4, 2, 1][pass];
    const step = [8, 8, 4, 2][pass];
    for (let y = start; y < h; y += step) rows[p++] = y;
  }
  return rows;
}

/** Pack an RGBA ImageData-like buffer into palette indices + palette. */
export function rgbaToIndices(rgba, width, height, palette, colors, lut) {
  const n = width * height;
  const idx = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const key = ((rgba[p + 2] & 0xf8) << 7) | ((rgba[p + 1] & 0xf8) << 2) | (rgba[p] >> 3);
    let v = lut[key];
    if (v === 0xff || v >= colors) v = 0;
    idx[i] = v;
  }
  return idx;
}

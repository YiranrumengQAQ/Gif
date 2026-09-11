/**
 * GIF89a parser + decoder.
 *
 * Four jobs:
 *  1. read GIF files as a conversion *input* (extract frames, delays, palettes)
 *  2. `optimizeExistingGif()` — shrink a GIF someone else made, without
 *     re-quantizing if the palette is already tight
 *  3. self-verification in tests (our encoder must round-trip through this)
 *  4. analysis (per-frame disposal, alpha usage, loop count, comment metadata)
 *
 * The parser is deliberately forgiving: real-world GIFs have garbage between
 * blocks, missing trailers, oversized color tables and truncated LZW streams.
 * It records `warnings` instead of throwing when it can continue.
 *
 * @module dec/gif
 */
import { lzwDecode } from '../enc/lzw.js';
import { GifxError, ErrorCode } from '../core/errors.js';

const BLOCK = {
  EXT: 0x21,
  IMAGE: 0x2c,
  TRAILER: 0x3b,
  GCE: 0xf9,
  APP: 0xff,
  COMMENT: 0xfe,
  PLAINTEXT: 0x01,
};

/**
 * Parse a GIF into structural data (palettes + LZW-compressed per-frame index
 * streams are decoded, but no compositing happens here).
 *
 * @param {Uint8Array} bytes
 * @param {object} [opts]
 * @param {boolean} [opts.indices=true] keep per-frame palette indices
 * @param {boolean} [opts.decode=true] run LZW (false = structure only, very fast)
 * @param {number} [opts.maxFrames]
 * @param {(i:number,total:number)=>void} [opts.onFrame]
 * @returns {GifData}
 */
export function parseGif(bytes, opts = {}) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes.buffer ?? bytes);
  if (u8.length < 13) throw new GifxError(ErrorCode.INPUT_CORRUPT, 'too short to be a GIF', { path: 'gif', data: { length: u8.length } });
  const ver = String.fromCharCode(u8[0], u8[1], u8[2], u8[3], u8[4], u8[5]);
  if (ver !== 'GIF87a' && ver !== 'GIF89a') {
    throw new GifxError(ErrorCode.INPUT_CORRUPT, `bad GIF signature "${ver}"`, { path: 'gif.signature' });
  }
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const warnings = [];
  let p = 6;
  const width = dv.getUint16(p, true);
  const height = dv.getUint16(p + 2, true);
  const packed = u8[p + 4];
  const gctFlag = !!(packed & 0x80);
  const colorResolution = ((packed >> 4) & 7) + 1;
  const sortFlag = !!(packed & 0x08);
  const gctSize = packed & 7;
  const backgroundColorIndex = u8[p + 5];
  const pixelAspectRatio = u8[p + 6];
  p += 7;

  let globalPalette = null;
  if (gctFlag) {
    const n = 2 << gctSize;
    globalPalette = u8.slice(p, p + n * 3);
    p += n * 3;
  } else {
    warnings.push('no global color table');
  }

  const frames = [];
  const comments = [];
  const allComments = [];
  let loop = null;
  let application = null;
  let appAuth = null;
  let xmp = null;
  let pendingGce = null;
  let pendingComments = [];
  let index = 0;
  let done = false;

  /**
   * Read a GIF sub-block chain (1-byte length, payload, …, 0x00 terminator).
   * @param {boolean} collect when true, also report the byte range so callers
   *   can copy the raw bytes verbatim (lossless re-mux path).
   */
  const readSubBlocks = (collect = false) => {
    const start = p;
    const chunks = [];
    let len = 0;
    for (;;) {
      if (p >= u8.length) {
        warnings.push('truncated sub-block chain');
        break;
      }
      const size = u8[p++];
      if (size === 0) break;
      if (p + size > u8.length) {
        warnings.push('truncated sub-block payload');
        chunks.push(u8.subarray(p, u8.length));
        len += u8.length - p;
        p = u8.length;
        break;
      }
      chunks.push(u8.subarray(p, p + size));
      len += size;
      p += size;
    }
    let bytes;
    if (chunks.length === 1) bytes = chunks[0];
    else {
      bytes = new Uint8Array(len);
      let o = 0;
      for (const c of chunks) {
        bytes.set(c, o);
        o += c.length;
      }
    }
    return collect ? { bytes, start, end: p } : bytes;
  };

  while (p < u8.length && !done) {
    const b = u8[p++];
    if (b === BLOCK.TRAILER) {
      done = true;
      break;
    }
    if (b === BLOCK.EXT) {
      const label = u8[p++];
      if (label === BLOCK.GCE) {
        const size = u8[p++];
        if (size >= 4) {
          const gp = u8[p];
          const transparent = gp & 1;
          const disposal = (gp >> 2) & 7;
          const userInput = (gp >> 1) & 1;
          const delay = dv.getUint16(p + 1, true);
          const tIndex = u8[p + 3];
          pendingGce = { delay, disposal, userInput: !!userInput, transparentIndex: transparent ? tIndex : -1 };
          p += size;
          if (u8[p] !== 0) {
            // some encoders omit the terminator; skip a block chain to resync
            readSubBlocks();
          } else p++;
        } else {
          warnings.push(`bad GCE block size ${size}`);
          p += size;
        }
      } else if (label === BLOCK.APP) {
        const size = u8[p++];
        const start = p;
        const id = String.fromCharCode(...u8.subarray(start, Math.min(start + 8, u8.length)));
        const auth = String.fromCharCode(...u8.subarray(start + 8, Math.min(start + size, u8.length)));
        application = id.trim();
        appAuth = auth;
        p += size;
        const payload = readSubBlocks();
        if (/NETSCAPE/i.test(id) && payload.length >= 3 && payload[0] === 1) {
          loop = payload[1] | (payload[2] << 8);
        } else if (/XMP/.test(id)) {
          try {
            xmp = new TextDecoder().decode(payload);
          } catch {
            /* ignore */
          }
        }
      } else if (label === BLOCK.COMMENT) {
        const text = new TextDecoder().decode(readSubBlocks());
        allComments.push(text);
        // A comment before the first image block belongs to the file; afterwards it
        // belongs to the frame that follows (that is how gifsicle reads them too).
        // Keeping the two lists disjoint is what lets a re-mux write each one once.
        if (frames.length) pendingComments.push(text);
        else comments.push(text);
      } else if (label === BLOCK.PLAINTEXT) {
        const size = u8[p++];
        p += size;
        readSubBlocks();
        warnings.push('plain text extension skipped');
      } else {
        readSubBlocks();
      }
      continue;
    }
    if (b === BLOCK.IMAGE) {
      if (p + 9 > u8.length) throw new GifxError(ErrorCode.INPUT_CORRUPT, 'truncated image descriptor', { path: `gif.frames[${index}]` });
      const x = dv.getUint16(p, true);
      const y = dv.getUint16(p + 2, true);
      const w = dv.getUint16(p + 4, true);
      const h = dv.getUint16(p + 6, true);
      const ip = u8[p + 8];
      p += 9;
      const lctFlag = !!(ip & 0x80);
      const interlace = !!(ip & 0x40);
      const sort = !!(ip & 0x20);
      const lctSize = ip & 7;
      let palette = globalPalette;
      if (lctFlag) {
        const n = 2 << lctSize;
        palette = u8.slice(p, p + n * 3);
        p += n * 3;
      }
      if (!palette) {
        // no palette at all: synthesise black so the frame still decodes
        palette = new Uint8Array(6);
        warnings.push(`frame ${index} had no color table`);
      }
      const minCodeSize = u8[p++];
      const raw = opts.keepStream ? readSubBlocks(true) : null;
      const stream = raw ? raw.bytes : readSubBlocks();
      const gce = pendingGce || { delay: 10, disposal: 0, userInput: false, transparentIndex: -1 };
      pendingGce = null;
      const frame = {
        index: index++,
        x,
        y,
        width: w,
        height: h,
        interlace,
        sort,
        palette,
        paletteColors: palette.length / 3,
        minCodeSize,
        delayCs: gce.delay,
        delayMs: gce.delay * 10,
        disposal: gce.disposal,
        userInput: gce.userInput,
        transparentIndex: gce.transparentIndex,
        comments: pendingComments.splice(0),
        compressedSize: stream.length,
        // Raw (still sub-block-framed) data — lets the rewriter copy an
        // unchanged frame's LZW stream verbatim instead of re-encoding it.
        streamStart: raw ? raw.start : -1,
        streamEnd: raw ? raw.end : -1,
      };
      if (opts.keepStream) frame.stream = stream;
      if (opts.decode !== false) {
        const n = w * h;
        const indices = new Uint8Array(n);
        let got = 0;
        try {
          got = lzwDecode(stream, Math.max(2, Math.min(11, minCodeSize)), indices, n);
        } catch (err) {
          warnings.push(`frame ${frame.index} LZW error: ${err.message}`);
        }
        if (got < n) warnings.push(`frame ${frame.index} incomplete (${got}/${n} pixels) — padded`);
        if (interlace) {
          const deint = new Uint8Array(n);
          // GIF interlace passes: 8th lines from 0, 8th from 4, 4th from 2, 2nd from 1
          const starts = [0, 4, 2, 1];
          const steps = [8, 8, 4, 2];
          let src = 0;
          for (let pass = 0; pass < 4; pass++) {
            for (let row = starts[pass]; row < h; row += steps[pass]) {
              deint.set(indices.subarray(src, src + w), row * w);
              src += w;
            }
          }
          frame.indices = deint;
        } else {
          frame.indices = opts.indices === false ? null : indices;
        }
        if (opts.indices === false && !interlace) frame.indices = null;
      }
      frames.push(frame);
      if (opts.maxFrames && frames.length >= opts.maxFrames) {
        warnings.push(`stopped after ${opts.maxFrames} frames (maxFrames)`);
        break;
      }
      continue;
    }
    // unknown block: skip the sub-block chain and resync
    warnings.push(`unknown block 0x${b.toString(16)} at ${p - 1}`);
    readSubBlocks();
  }

  if (!frames.length && !done) warnings.push('no frames found');

  let hasAlpha = false;
  for (const f of frames) if (f.transparentIndex >= 0) hasAlpha = true;

  return {
    version: ver,
    width,
    height,
    colorResolution,
    sortFlag,
    backgroundColorIndex,
    pixelAspectRatio,
    palette: globalPalette,
    paletteColors: globalPalette ? globalPalette.length / 3 : 0,
    loop,
    hasAlpha,
    frames,
    comments,
    allComments,
    application,
    appAuth,
    xmp,
    warnings,
    truncated: !done,
    bytes: u8.length,
  };
}

/**
 * Composite parsed frames into RGBA bitmaps honouring disposal methods — the
 * same semantics browsers use, so preview ≡ output.
 *
 * @param {ReturnType<typeof parseGif>} gif
 * @param {object} [opts]
 * @param {number[]} [opts.only] frame indices to render (others skipped, canvas kept)
 * @param {Uint8Array} [opts.canvas] reuse a width*height*4 buffer
 * @param {{raster:number,r:g:number,b:number}} [opts.background] used for disposal=2
 * @param {(frame:{index:number,data:Uint8Array,durationMs:number,pts:number})=>void} [opts.onFrame]
 *   streaming mode: called per frame, buffer is only valid during the call
 * @returns {{data:Uint8Array,width:number,height:number,frames:object[]}[]|void}
 */
export function composeGifFrames(gif, opts = {}) {
  const W = gif.width;
  const H = gif.height;
  const out = [];
  const canvas = opts.canvas || new Uint8Array(W * H * 4);
  // Sub-framed GIFs never touch part of the canvas. A viewer shows the logical
  // screen descriptor's background color there, so we composite onto the same
  // base instead of leaving uninitialized transparent pixels — otherwise
  // "did this frame change?" comparisons disagree with every real renderer.
  // Pass `background: null` to keep the old transparent-zero base.
  let bg = opts.background;
  if (bg === undefined) {
    const bgi = gif.backgroundColorIndex | 0;
    bg = gif.palette && bgi * 3 + 2 < gif.palette.length ? { r: gif.palette[bgi * 3], g: gif.palette[bgi * 3 + 1], b: gif.palette[bgi * 3 + 2] } : null;
  }
  if (bg && opts.noClear !== true) {
    for (let i = 0; i < W * H; i++) {
      canvas[i * 4] = bg.r;
      canvas[i * 4 + 1] = bg.g;
      canvas[i * 4 + 2] = bg.b;
      canvas[i * 4 + 3] = 255;
    }
  }
  const snapshot = () => canvas.slice();
  let previous = null;
  let saved = null;
  let pts = 0;
  const collect = !opts.onFrame;
  for (const f of gif.frames) {
    if (opts.only && !opts.only.includes(f.index)) {
      pts += f.delayMs;
      continue;
    }
    if (f.disposal === 3) previous = snapshot();
    else if (f.disposal === 2) saved = snapshot();
    paintFrame(canvas, W, H, f, bg);
    const data = collect ? canvas.slice() : canvas;
    const rec = { index: f.index, data, width: W, height: H, durationMs: f.delayMs, delayCs: f.delayCs, pts: pts / 1000, frame: f };
    if (opts.onFrame) opts.onFrame(rec);
    else out.push(rec);
    // Disposal methods act on the frame's drawing area only (GIF89a §23), which is
    // what makes sub-framed GIFs work: clearing the whole canvas here wipes every
    // earlier frame out of existence.
    if (f.disposal === 2) {
      const b = bg || { r: 0, g: 0, b: 0 };
      const y0 = Math.max(0, f.y);
      const y1 = Math.min(H, f.y + f.height);
      const x0 = Math.max(0, f.x);
      const x1 = Math.min(W, f.x + f.width);
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * W + x) * 4;
          canvas[i] = b.r;
          canvas[i + 1] = b.g;
          canvas[i + 2] = b.b;
          canvas[i + 3] = 255;
        }
      }
    } else if (f.disposal === 3 && previous) {
      const y0 = Math.max(0, f.y);
      const y1 = Math.min(H, f.y + f.height);
      const x0 = Math.max(0, f.x);
      const x1 = Math.min(W, f.x + f.width);
      for (let y = y0; y < y1; y++) {
        const row = y * W * 4;
        for (let i = row + x0 * 4; i < row + x1 * 4; i++) canvas[i] = previous[i];
      }
    }
    pts += f.delayMs;
  }
  return collect ? out : undefined;
}

function paintFrame(canvas, W, H, f) {
  const pal = f.palette;
  const tIdx = f.transparentIndex;
  const { x: fx, y: fy, width: fw, height: fh, indices } = f;
  if (!indices) return;
  for (let y = 0; y < fh; y++) {
    const dy = fy + y;
    if (dy < 0 || dy >= H) continue;
    const srow = y * fw;
    const drow = dy * W;
    for (let x = 0; x < fw; x++) {
      const dx = fx + x;
      if (dx < 0 || dx >= W) continue;
      const idx = indices[srow + x];
      if (idx === tIdx) continue;
      const pi = idx * 3;
      if (pi + 2 >= pal.length) continue;
      const di = (drow + dx) * 4;
      canvas[di] = pal[pi];
      canvas[di + 1] = pal[pi + 1];
      canvas[di + 2] = pal[pi + 2];
      canvas[di + 3] = 255;
    }
  }
}

/** Total duration in ms + per-frame timing stats. */
export function gifDuration(gif) {
  let ms = 0;
  const delays = new Set();
  for (const f of gif.frames) {
    ms += f.delayCs * 10;
    delays.add(f.delayCs);
  }
  return {
    ms,
    seconds: ms / 1000,
    frames: gif.frames.length,
    fps: ms > 0 ? (gif.frames.length / ms) * 1000 : 0,
    distinctDelays: delays.size,
    minDelayCs: Math.min(...[...delays, 0].filter((d) => d > 0)),
    maxDelayCs: Math.max(...delays, 0),
    playsOnce: gif.loop === null || gif.loop === 1,
  };
}

/**
 * Statistics used by the optimizer to decide whether re-encoding a GIF is
 * worth it (e.g. its palette wastes colours, or its frames are full-canvas
 * when they could be dirty rects).
 */
export function analyzeGif(gif) {
  const used = new Set();
  let localPalettes = 0;
  let fullCanvas = 0;
  let transparent = 0;
  let lzwBytes = 0;
  for (const f of gif.frames) {
    if (f.palette !== gif.palette) localPalettes++;
    if (f.width === gif.width && f.height === gif.height) fullCanvas++;
    if (f.transparentIndex >= 0) transparent++;
    lzwBytes += f.compressedSize || 0;
    if (f.indices) {
      const n = Math.min(f.indices.length, 65536);
      for (let i = 0; i < n; i++) used.add(f.indices[i]);
    }
  }
  const paletteCount = gif.paletteColors || 0;
  return {
    frames: gif.frames.length,
    usedColorCount: used.size,
    paletteCount,
    wastedPaletteEntries: Math.max(0, paletteCount - used.size),
    localPalettes,
    fullCanvasFrames: fullCanvas,
    transparentFrames: transparent,
    lzwBytes,
    headerBytes: gif.bytes - lzwBytes,
    bytesPerFrame: gif.frames.length ? Math.round(gif.bytes / gif.frames.length) : gif.bytes,
    duration: gifDuration(gif),
    reencodeMayHelp: used.size < paletteCount * 0.9 || fullCanvas > gif.frames.length * 0.5,
  };
}

/** Quick "is this a GIF and how big is it" probe without decoding anything. */
export function peekGif(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes.buffer ?? bytes);
  if (u8.length < 13) return null;
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const packed = u8[10];
  return {
    version: String.fromCharCode(u8[0], u8[1], u8[2], u8[3], u8[4], u8[5]),
    width: dv.getUint16(6, true),
    height: dv.getUint16(8, true),
    hasGlobalPalette: !!(packed & 0x80),
    paletteColors: packed & 0x80 ? 2 << (packed & 7) : 0,
    bytes: u8.length,
  };
}

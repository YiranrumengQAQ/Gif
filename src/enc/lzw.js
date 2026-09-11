/**
 * GIF89a LZW codec.
 *
 * The encoder is the single hottest function in a video→GIF pipeline: for a
 * 640×360 × 60-frame GIF it touches 13.8M texels. Implementations that
 * (a) rebuild their dictionary with fill() on every clear code, (b) call a
 * byte-writer per emitted code, or (c) re-pack pixels into a temp array, are
 * 2–4× slower. This one avoids all three:
 *
 *  - epoch-stamped hash table → clear code costs O(1), not O(table size)
 *  - 24-bit accumulator bit writer → ≤2 stores per code instead of ≤12
 *  - rect + row-stride input → dirty-rect frames encode with zero copies
 *  - optional "clear-code interval" tuning to keep the dictionary dense
 *
 * Bit-exact against the GIF spec (verified by round-trip in test/lzw.test.js
 * and by `dec/gif.js` decoding its own output).
 *
 * @module enc/lzw
 */

const MAX_BITS = 12;
const MAX_CODE = 1 << MAX_BITS; // 4096
const HT_SIZE = 1 << 16; // power of two → masking instead of %
const HT_MASK = HT_SIZE - 1;

/**
 * Encode `count` indices as a GIF LZW bit stream.
 *
 * @param {Uint8Array|Uint8ClampedArray} pixels palette indices
 * @param {number} count number of pixels to read from `pixels`
 * @param {number} minCodeSize bits per index (2..8, normally 8; ≤16 colors → 4 etc.)
 * @param {Uint8Array} out destination (must have room; see {@link lzwMaxBytes})
 * @param {object} [opts]
 * @param {number} [opts.offset=0] start position inside `out`
 * @param {number} [opts.width] when given with `stride`, pixels are read as a
 *   rectangle: `pixels[y*stride + x + x0]` for y in 0..height-1, x in 0..width-1
 * @param {number} [opts.stride]
 * @param {number} [opts.x0=0] rect left
 * @param {number} [opts.y0=0] rect top
 * @param {number} [opts.rectH] rect height (defaults to rows needed for count)
 * @param {number[]} [opts.rowOrder] interlace row permutation (values are row
 *   indices within the rect)
 * @param {number} [opts.clearInterval=0] emit a clear code every N pixels
 *   (0 = only when the dictionary fills). Small gains on noisy content.
 * @returns {number} bytes written
 */
export function lzwEncode(pixels, count, minCodeSize, out, opts = {}) {
  if (minCodeSize < 2) minCodeSize = 2;
  if (minCodeSize > 8) minCodeSize = 8;
  const offset = opts.offset | 0;
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;

  const useRect = opts.width > 0 && opts.stride > 0;
  const rectW = useRect ? opts.width | 0 : count;
  const rectH = useRect ? (opts.rectH ?? Math.ceil(count / rectW)) | 0 : 1;
  const stride = useRect ? opts.stride | 0 : rectW;
  const x0 = opts.x0 | 0;
  const y0 = opts.y0 | 0;
  const rowOrder = opts.rowOrder || null;
  const clearIntervalPx = opts.clearInterval > 0 ? opts.clearInterval | 0 : 0;
  /** hard pixel cap (used by the streaming/abortable path) */
  const stopAfter = opts.stopAfter > 0 ? opts.stopAfter | 0 : 0;
  // GIF's "deferred change" convention: the code width is raised on the code
  // *after* the dictionary entry that needs it was created. Encoders that bump
  // eagerly produce a stream no conforming decoder can read past code 7.
  // `earlyWidthBump` exists only so tools/oracle-lzw.mjs can demonstrate that.
  const earlyBump = opts.earlyWidthBump === true;

  const hashTab = getHashTab();
  const keyTab = getKeyTab();
  const stamp = getStamp();
  let epoch = nextEpoch();

  const codeSizeBits = new Int32Array(1);

  // --- bit writer state (locals, not an object: keeps the hot loop monomorphic)
  let pos = offset;
  let bitBuf = 0;
  let bitCount = 0;

  const dictBits = minCodeSize + 1;
  let codeSize = dictBits;
  let nextCode = eoiCode + 1;

  const write = (code, bits) => {
    if (bitCount + bits <= 24) {
      bitBuf |= (code & ((1 << bits) - 1)) << bitCount;
      bitCount += bits;
      // Drain only *complete* bytes; leftover bits stay in the accumulator for
      // the next code. Using do/while here corrupts the stream badly.
      while (bitCount >= 8) {
        out[pos++] = bitBuf & 255;
        bitBuf >>>= 8;
        bitCount -= 8;
      }
    } else {
      let v = code;
      let b = bits;
      while (b > 0) {
        const room = 24 - bitCount;
        const take = room < b ? room : b;
        bitBuf |= (v & ((1 << take) - 1)) << bitCount;
        bitCount += take;
        v >>>= take;
        b -= take;
        while (bitCount >= 8) {
          out[pos++] = bitBuf & 255;
          bitBuf >>>= 8;
          bitCount -= 8;
        }
      }
    }
  };

  const reset = () => {
    // The Clear code must be written at the width the decoder is *currently*
    // reading at (which includes any deferred bump from the entry we just added),
    // and only then does the width go back to its initial value. Writing it at
    // `dictBits` desynchronises every code that follows — visible as a stream that
    // decodes to a fraction of the pixels, and only for frames large enough to
    // have bumped their width, which is why small unit tests never caught it.
    epoch = nextEpoch();
    write(clearCode, codeSize);
    nextCode = eoiCode + 1;
    codeSize = dictBits;
  };

  write(clearCode, codeSize);
  codeSizeBits[0] = codeSize;

  let ent = -1;
  let sinceClear = 0;
  let n = 0;

  // Row-major walk without any per-pixel division: precompute each stream row's
  // starting offset (this is also where interlacing and dirty-rect offsets are
  // applied, so the inner loop is a single `pixels[at++]` read).
  const rowStart = new Int32Array(rectH);
  if (useRect) {
    for (let ry = 0; ry < rectH; ry++) {
      const srcRow = (rowOrder ? rowOrder[ry] : ry) + y0;
      rowStart[ry] = srcRow * stride + x0;
    }
  } else {
    rowStart[0] = 0;
  }

  outer: for (let ry = 0; ry < rectH; ry++) {
    let at = rowStart[ry];
    const rowEnd = at + rectW;
    for (; at < rowEnd && n < count; at++, n++) {
      const p = pixels[at];
      if (ent < 0) {
        ent = p;
        continue;
      }
      const key = ((ent << 8) | p) >>> 0;
      // Multiplicative hash; xor-shifts of the 20-bit key spread better than a
      // plain mask on palettes with clustered indices.
      let idx = Math.imul(key, 0x9e3779b1) & HT_MASK;
      let found = -1;
      for (;;) {
        if (stamp[idx] !== epoch) break; // empty slot
        if (keyTab[idx] === key) {
          found = hashTab[idx];
          break;
        }
        idx = (idx + 1) & HT_MASK;
      }
      if (found >= 0) {
        ent = found;
      } else {
        // "early" variant (negative control only): raises the width one code too
        // early, which is what naive rewrites of GIF LZW keep getting wrong.
        if (earlyBump && codeSize < MAX_BITS && nextCode + 1 >= (1 << codeSize)) codeSize++;
        write(ent, codeSize);
        if (nextCode < MAX_CODE) {
          // Standard deferred change: bump after this code has been written and
          // immediately before the entry that would not fit in the old width is
          // inserted. This mirrors the decoder's `avail > maxcode` rule.
          if (!earlyBump && codeSize < MAX_BITS && nextCode >= (1 << codeSize)) codeSize++;
          hashTab[idx] = nextCode;
          keyTab[idx] = key;
          stamp[idx] = epoch;
          nextCode++;
        } else {
          reset();
          sinceClear = 0;
        }
        ent = p;
        if (clearIntervalPx && ++sinceClear >= clearIntervalPx && nextCode > eoiCode + 32) {
          reset();
          sinceClear = 0;
        }
      }
      if (stopAfter && n >= stopAfter) break outer;
    }
  }
  if (ent >= 0) write(ent, codeSize);
  write(eoiCode, codeSize);
  if (bitCount > 0) out[pos++] = bitBuf & 255;
  return pos - offset;
}

/**
 * Hard upper bound on the encoded size, used to size the destination buffer
 * without measuring twice. The output width is `minCodeSize+1` initially but
 * grows to 12 bits as the dictionary fills — and it can reach 12 bits on any
 * input long enough to fill the table, so the bound must assume 12 bits per
 * emitted code regardless of `minCodeSize`. Getting this wrong silently
 * truncates typed-array writes (they are no-ops past the end), which is the
 * worst possible failure mode — hence the generous +64.
 *
 * Emit count ≤ pixels + 2 (one code per pixel plus trailing + EOI), plus one
 * clear code per 4090-ish entries.
 */
export function lzwMaxBytes(pixels, minCodeSize = 8) {
  const codes = pixels + 2 + Math.ceil(pixels / 255) + 1;
  return Math.ceil((codes * MAX_BITS) / 8) + Math.ceil(codes / 255) + 64;
}

/* ----------------------------------------------------------- dictionary */

let _hashTab = null;
let _keyTab = null;
let _stamp = null;
let _epoch = 0;
const EPOCH_MAX = 0x7ffffffe;

function getHashTab() {
  if (!_hashTab) _hashTab = new Int32Array(HT_SIZE);
  return _hashTab;
}
function getKeyTab() {
  if (!_keyTab) _keyTab = new Int32Array(HT_SIZE);
  return _keyTab;
}
function getStamp() {
  if (!_stamp) _stamp = new Int32Array(HT_SIZE);
  return _stamp;
}
function nextEpoch() {
  _epoch++;
  if (_epoch > EPOCH_MAX) {
    _stamp.fill(0);
    _epoch = 1;
  }
  return _epoch;
}

/** Release module-level scratch (worker teardown / `arena.dispose()`). */
export function lzwResetScratch() {
  _hashTab = _keyTab = _stamp = null;
  _epoch = 0;
}

/* --------------------------------------------------------------- decode */

/**
 * Decode a GIF LZW bit stream into `out`.
 *
 * Uses the classic prefix/suffix + stack-walk decoder with an explicit
 * `first[]` table so the KwKwK special case stays O(1) instead of re-walking
 * the chain. Tolerant of truncated streams (returns what it could produce)
 * because half of the GIFs on the internet are truncated.
 *
 * @param {Uint8Array} data bitstream bytes (already de-sub-blocked)
 * @param {number} minCodeSize
 * @param {Uint8Array} out destination, at least the expected pixel count
 * @param {number} [limit] stop after this many pixels
 * @returns {number} pixels written
 */
export function lzwDecode(data, minCodeSize, out, limit = out.length) {
  if (minCodeSize < 2) minCodeSize = 2;
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;
  let bits = minCodeSize + 1;
  let maxCode = (1 << bits) - 1;
  let nextCode = eoiCode + 1;

  const prefix = new Int32Array(MAX_CODE);
  const suffix = new Int32Array(MAX_CODE);
  const first = new Int32Array(MAX_CODE);
  const stack = new Uint8Array(MAX_CODE);

  let pos = 0;
  let bitBuf = 0;
  let bitCount = 0;
  const readCode = () => {
    while (bitCount < bits) {
      if (pos >= data.length) return -1;
      bitBuf |= data[pos++] << bitCount;
      bitCount += 8;
    }
    const c = bitBuf & ((1 << bits) - 1);
    bitBuf >>>= bits;
    bitCount -= bits;
    return c;
  };

  let n = 0;
  let oldCode = -1;
  let sp = 0;
  // Every non-clear code emits ≥1 pixel, so a valid stream can never need more
  // than `limit` code-steps. Bounding it means a hostile or truncated GIF cannot
  // spin the tab (omggif, for one, loops forever on malformed input).
  const maxSteps = limit + data.length * 3 + 64;
  let steps = 0;

  for (;;) {
    if (++steps > maxSteps) break;
    const code = readCode();
    if (code < 0) break;
    if (code === clearCode) {
      bits = minCodeSize + 1;
      maxCode = (1 << bits) - 1;
      nextCode = eoiCode + 1;
      oldCode = -1;
      sp = 0;
      continue;
    }
    if (code === eoiCode) break;

    // Walk the chain of `code` onto the stack (strings are stored as
    // prefix+suffix links, so bytes come out backwards).
    let walk;
    if (code < nextCode) {
      walk = code;
    } else if (code === nextCode && oldCode >= 0) {
      // KwKwK: the code we were just about to define is string(oldCode) +
      // firstByte(string(oldCode)). Push the extra byte now; the shared
      // insertion path below then defines the entry, exactly like giflib.
      const inCode = oldCode > eoiCode ? first[oldCode] : oldCode;
      stack[sp++] = inCode;
      walk = oldCode;
    } else {
      break; // invalid code — stop, keep the partial frame
    }

    while (walk > eoiCode) {
      stack[sp++] = suffix[walk];
      walk = prefix[walk];
      if (sp >= MAX_CODE) break;
    }
    if (walk < clearCode) stack[sp++] = walk;
    // `walk` is now the FIRST literal byte of the string just decoded, which is
    // exactly what the next dictionary entry needs as its suffix.

    while (sp > 0) {
      out[n++] = stack[--sp];
      if (n >= limit) return n; // caller's buffer full
    }

    if (oldCode >= 0 && nextCode < MAX_CODE) {
      prefix[nextCode] = oldCode;
      suffix[nextCode] = walk; // first byte of the string just decoded
      // The new entry's first byte is the first byte of *oldCode*'s string —
      // not of the current one. Getting this wrong corrupts every later
      // KwKwK code that references the entry.
      first[nextCode] = oldCode > eoiCode ? first[oldCode] : oldCode;
      nextCode++;
      if (nextCode > maxCode && bits < MAX_BITS) {
        bits++;
        maxCode = (1 << bits) - 1;
      }
    }
    oldCode = code;
  }
  return n;
}

/* --------------------------------------------------- sub-block packing */

/**
 * Wrap a raw bit stream into GIF sub-blocks (1..255-byte chunks terminated by
 * a 0x00 byte). Returns bytes written (== input length + ceil(len/255) + 1).
 */
export function writeSubBlocks(src, srcOffset, srcLen, out, outOffset, maxChunk = 255) {
  let p = outOffset;
  let remaining = srcLen;
  let at = srcOffset;
  while (remaining > maxChunk) {
    out[p++] = maxChunk;
    out.set(src.subarray(at, at + maxChunk), p);
    p += maxChunk;
    at += maxChunk;
    remaining -= maxChunk;
  }
  if (remaining > 0) {
    out[p++] = remaining;
    out.set(src.subarray(at, at + remaining), p);
    p += remaining;
  }
  out[p++] = 0;
  return p - outOffset;
}

/**
 * Size of the sub-block framing for `len` payload bytes.
 * @param {number} len
 * @param {number} [maxChunk=255]
 */
export function subBlockSize(len, maxChunk = 255) {
  return len + (len === 0 ? 0 : Math.floor((len - 1) / maxChunk) + 1) * 1 + 1;
}

/**
 * Choose the smallest legal minCodeSize for a palette, which shaves 1–2 bits
 * per code on ≤16-colour GIFs (typically another 3–6% smaller).
 */
export function minCodeSizeFor(colors) {
  if (colors <= 4) return 2;
  if (colors <= 8) return 3;
  if (colors <= 16) return 4;
  if (colors <= 32) return 5;
  if (colors <= 64) return 6;
  if (colors <= 128) return 7;
  return 8;
}

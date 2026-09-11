/**
 * LZW / GIF conformance oracle (dev-only, needs the `omggif` devDependency).
 *
 * Proves two things in both directions:
 *   1. our LZW bit stream is decodable by an independent, battle-tested
 *      implementation (omggif is what countless npm GIF tools ship), and
 *   2. our decoder reads streams produced by that implementation.
 *
 * It also asserts that the "immediate/early" code-width bump variant FAILS,
 * which documents why `lzwEncode` uses the deferred-change convention.
 *
 * Usage: node tools/oracle-lzw.mjs
 */
import { GifWriter as OggWriter, GifReader } from 'omggif';
import { GifWriter } from '../src/enc/gif.js';
import { lzwEncode } from '../src/enc/lzw.js';
import { parseGif } from '../src/dec/gif.js';

function rng(seed) { let s = seed >>> 0 || 1; return () => ((s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff) / 0x7fffffff); }
function paletteFor(colors) { const pal = new Uint8Array(colors * 3); for (let i = 0; i < colors; i++) { pal[i*3] = (i * 37) & 255; pal[i*3+1] = (i * 91) & 255; pal[i*3+2] = (i * 151 + 7) & 255; } return pal; }

const CASES = [[16, 16, 2], [64, 32, 4], [100, 50, 16], [233, 7, 64], [640, 4, 200], [37, 91, 256]];

/** GIF color-table field value n ⟹ 2^(n+1) table entries. */
function colorTableSize(colors) {
  let n = 0;
  while ((1 << (n + 1)) < colors) n++;
  return n;
}
/** Number of palette slots a conforming writer must emit for `colors`. */
function colorTableEntries(colors) {
  return Math.max(2, 1 << (colorTableSize(colors) + 1));
}

function padPalette(pal, colors) {
  const entries = colorTableEntries(colors);
  if (pal.length >= entries * 3) return pal;
  const out = new Uint8Array(entries * 3);
  out.set(pal.subarray(0, colors * 3));
  for (let i = colors; i < entries; i++) {
    out[i * 3] = pal[(colors - 1) * 3];
    out[i * 3 + 1] = pal[(colors - 1) * 3 + 1];
    out[i * 3 + 2] = pal[(colors - 1) * 3 + 2];
  }
  return out;
}

/**
 * Encode one full-canvas frame with our real GifWriter, optionally toggling the
 * LZW width-bump mode so the negative control exercises the same code path.
 */
function ourGifWithMode(w, h, px, colors, early) {
  const pal = paletteFor(colors);
  const g = new GifWriter({ width: w, height: h, palette: pal, colors, loop: 0 });
  g.begin();
  g.addFrame({ width: w, height: h, indices: px, delay: 10, disposal: 1, lzw: { earlyWidthBump: early } });
  return { bytes: g.finish(), pal };
}

function omggifDecode(bytes, w, h) {
  const rd = new GifReader(bytes);
  const img = new Uint8Array(w * h * 4);
  rd.decodeAndBlitFrameRGBA(0, img);
  return img;
}

let failures = 0;
const check = (name, cond, extra = '') => { if (!cond) { failures++; console.log(`  ✗ ${name} ${extra}`); } else console.log(`  ✓ ${name} ${extra}`); };

console.log('[1/3] our LZW stream → omggif decoder (deferred change: must pass)');
for (const [w, h, colors] of CASES) {
  const r = rng(w * 7919 + h * 104729 + colors);
  const px = new Uint8Array(w * h);
  for (let i = 0; i < px.length; i++) px[i] = Math.floor(r() * colors);
  const { bytes, pal } = ourGifWithMode(w, h, px, colors, false);
  let ok = true; let why = '';
  try {
    const img = omggifDecode(bytes, w, h);
    for (let i = 0; i < px.length; i++) {
      const o = i * 4;
      if (img[o] !== pal[px[i]*3] || img[o+1] !== pal[px[i]*3+1] || img[o+2] !== pal[px[i]*3+2]) { ok = false; why = `pixel ${i}: got ${img[o]},${img[o+1]},${img[o+2]} want ${pal[px[i]*3]},${pal[px[i]*3+1]},${pal[px[i]*3+2]}`; break; }
    }
  } catch (e) { ok = false; why = (e && e.message) || String(e); }
  check(`${w}x${h}/${colors}c`, ok, ok ? `(${bytes.length} B)` : why);
}

console.log('[2/3] negative control: early width bump must be rejected');
// NOTE: we deliberately do NOT hand the malformed stream to omggif's decoder —
// its read loop is unbounded and spins forever on some corrupt inputs (that is
// exactly why our own decoder carries a step budget; see lzwDecode).
for (const [w, h, colors] of CASES.slice(0, 4)) {
  const r = rng(w + h * 31 + colors);
  const px = new Uint8Array(w * h);
  for (let i = 0; i < px.length; i++) px[i] = Math.floor(r() * colors);
  const standard = ourGifWithMode(w, h, px, colors, false);
  const early = ourGifWithMode(w, h, px, colors, true);
  // (a) the two bitstreams must differ, otherwise the flag is dead code
  let differs = standard.bytes.length !== early.bytes.length;
  if (!differs) for (let i = 0; i < standard.bytes.length; i++) if (standard.bytes[i] !== early.bytes[i]) { differs = true; break; }
  // (b) our spec-conforming decoder must fail to recover the pixels from the
  //     early-bump stream while succeeding on the standard one
  const decodePixels = (bytes) => {
    const gif = parseGif(bytes);
    const f = gif.frames[0];
    if (!f || !f.indices) return false;
    for (let i = 0; i < px.length; i++) if (f.indices[i] !== px[i]) return false;
    return true;
  };
  const stdOk = decodePixels(standard.bytes);
  const earlyBad = !decodePixels(early.bytes);
  check(`${w}x${h}/${colors}c bump schedule differs & non-standard stream rejected`, differs && stdOk && earlyBad, `differs=${differs} stdOk=${stdOk} earlyBad=${earlyBad}`);
}

console.log('[3/3] omggif encoder → our parser/decoder');
for (const [w, h, colors] of CASES) {
  const r = rng(w * 13 + h * 17 + colors * 5);
  const px = new Uint8Array(w * h);
  for (let i = 0; i < px.length; i++) px[i] = Math.floor(r() * colors);
  const pal = paletteFor(colors);
  // omggif's palette is an array of packed 0xRRGGBB ints whose *length* is the
  // color count and must be a power of two.
  const entries = colorTableEntries(colors);
  const oggPal = new Array(entries);
  for (let i = 0; i < entries; i++) {
    const src = Math.min(i, colors - 1);
    oggPal[i] = (pal[src * 3] << 16) | (pal[src * 3 + 1] << 8) | pal[src * 3 + 2];
  }
  const buf = new Uint8Array(w * h * 3 + 8192);
  const gw = new OggWriter(buf, w, h, { loop: 0 });
  gw.addFrame(0, 0, w, h, px, { palette: oggPal, delay: 7 });
  const bytes = buf.subarray(0, gw.end());
  const gif = parseGif(new Uint8Array(bytes));
  const f = gif.frames[0];
  let ok = !!f && f.width === w && f.height === h && f.delayCs === 7 && f.indices.length === w * h;
  let why = '';
  if (ok) for (let i = 0; i < px.length; i++) if (f.indices[i] !== px[i]) { ok = false; why = `pixel ${i}: ${f.indices[i]} != ${px[i]}`; break; }
  else why = `frames=${gif.frames.length} delay=${f && f.delayCs}`;
  check(`${w}x${h}/${colors}c`, ok, why);
}

console.log(failures ? `\n${failures} ORACLE FAILURE(S)` : '\nLZW/GIF oracle: all checks passed');
process.exit(failures ? 1 : 0);

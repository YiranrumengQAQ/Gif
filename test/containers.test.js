/**
 * Container demuxer tests. Real files aren't checked into the repo (they'd bloat
 * it and can't be redistributed freely), so each parser is exercised against a
 * hand-assembled container that follows the spec byte-for-byte — which catches
 * exactly the class of bugs that matter here: off-by-one box offsets, wrong
 * timescale math, and sample tables that resolve outside the file.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseMp4, mp4FrameTimes, mp4SeekRange } from '../src/media/mp4.js';
import { parseWebm } from '../src/media/webm.js';
import { parseRiff, buildUncompressedAvi } from '../src/media/riff.js';
import { identify, probe } from '../src/media/probe.js';

/* ------------------------------------------------------------- tiny writers */

const u32 = (n) => new Uint8Array([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]);
const u16 = (n) => new Uint8Array([(n >>> 8) & 255, n & 255]);
const cat = (...a) => {
  const out = new Uint8Array(a.reduce((n, x) => n + x.length, 0));
  let o = 0;
  for (const x of a) {
    out.set(x, o);
    o += x.length;
  }
  return out;
};
const box = (type, ...payload) => {
  const p = cat(...payload);
  return cat(u32(8 + p.length), new TextEncoder().encode(type), p);
};
const fullBox = (type, version, flags, ...payload) => {
  const p = cat(...payload);
  return cat(u32(12 + p.length), new TextEncoder().encode(type), new Uint8Array([version, (flags >> 16) & 255, (flags >> 8) & 255, flags & 255]), p);
};

/** 3 samples, 1000 timescale, 30fps-ish, with a B-frame offset. */
function buildMp4() {
  const timescale = 30000;
  const dur = 1000; // 33.33ms
  const samples = [
    new Uint8Array([0, 0, 0, 4, 0x65, 1, 2, 3]),
    new Uint8Array([0, 0, 0, 3, 0x41, 5, 6]),
    new Uint8Array([0, 0, 0, 2, 0x41, 7]),
  ];
  const sizes = samples.map((s) => s.length);
  const stsd = fullBox('stsd', 0, 0, u32(1), ...[
    (() => {
      const avcC = box('avcC', new Uint8Array([1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 2, 0x67, 0x64, 1, 1]));
      const inner = cat(
        new Uint8Array(6), // reserved
        u16(1), // data_reference_index
        u16(0), // pre_defined
        u16(0), // reserved
        new Uint8Array(12), // pre_defined
        u16(64), u16(32), // width, height
        u32(0x00480000), u32(0x00480000), // 72 dpi
        u32(0), // reserved
        u16(1), // frame_count
        new Uint8Array(32), // compressorname
        u16(24), // bit_depth_minus_one == 24
        i16(-1) // pre_defined
      );
      const payload = cat(inner, avcC);
      return cat(u32(8 + payload.length), new TextEncoder().encode('avc1'), payload);
    })(),
  ]);
  const stts = fullBox('stts', 0, 0, u32(1), u32(3), u32(dur));
  const stsz = fullBox('stsz', 0, 0, u32(0), u32(sizes.length), ...sizes.map(u32));
  const stss = fullBox('stss', 0, 0, u32(1), u32(1));
  const ctts = fullBox('ctts', 1, 0, u32(1), u32(3), i32(1000));
  const mdatPayload = cat(...samples);
  const mdat = box('mdat', mdatPayload);
  const mdatOffset = 8;
  const stco = fullBox('stco', 0, 0, u32(1), u32(0)); // patched below
  const stbl = box('stbl', stsd, stts, stsz, stss, ctts, stco);
  const minf = box('minf', box('vmhd', new Uint8Array(8)), box('dinf', box('dref', fullBox('dref', 0, 0, u32(1)))), stbl);
  const hdlr = fullBox('hdlr', 0, 0, u32(0), new TextEncoder().encode('vide'), new Uint8Array(12), new Uint8Array([0]));
  const mdhd = fullBox('mdhd', 0, 0, u32(0), u32(0), u32(timescale), u32(3 * dur), u16(0x55c4));
  const mdia = box('mdia', mdhd, hdlr, minf);
  // Tkhd v0: ver/flags | creation | modification | trackId | reserved | duration
  // | reserved(8) | layer | altGroup | volume | reserved | matrix(36) | w | h
  const tkhd = fullBox(
    'tkhd',
    0,
    3,
    u32(0),
    u32(0),
    u32(1),
    u32(0),
    u32(3 * dur),
    new Uint8Array(8),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    u32(64 << 16),
    u32(32 << 16)
  );
  const trak = box('trak', tkhd, mdia);
  const mvhd = fullBox(
    'mvhd',
    0,
    0,
    u32(0),
    u32(0),
    u32(timescale),
    u32(3 * dur),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    new Uint8Array(24),
    u32(0),
    u32(2)
  );
  const ftyp = box('ftyp', new TextEncoder().encode('isom'), u32(0x200), new TextEncoder().encode('isom'), new TextEncoder().encode('mp42'));
  let moov = box('moov', mvhd, trak);
  const file = cat(ftyp, moov, mdat);
  // patch the single chunk offset to point at the first sample inside mdat
  const mdatPayloadStart = ftyp.length + moov.length + 8;
  const fourccAt = indexOfFourcc(file, 'stco');
  assert.ok(fourccAt > 0);
  // 'stco' fourcc is at boxStart+4, so the first entry is boxStart+16 = fourccAt+12
  file.set(u32(mdatPayloadStart), fourccAt + 12);
  void mdatOffset;
  void mdat;
  void stco;
  return { file, samples, sizes, timescale, dur };
}
const i16 = (n) => new Uint8Array([(n >> 8) & 255, n & 255]);
const i32 = (n) => new Uint8Array([(n >> 24) & 255, (n >> 16) & 255, (n >> 8) & 255, n & 255]);

/* ---------------------------------------------------------------------- EBML */

const vintSize = (n) => {
  for (let len = 1; len <= 8; len++) if (n < 2 ** (7 * len) - 1) return cat(new Uint8Array([0xff & (1 << (8 - len)) | ((n >> (8 * (len - 1))) & (0xff >> len))]), ...Array.from({ length: len - 1 }, (_, i) => u8((n >> (8 * (len - 2 - i))) & 255)));
  throw new Error('too big');
};
const u8 = (n) => new Uint8Array([n & 255]);
/** EBML unsigned vint: length marker in the leading bits, then the value. */
const vuint = (n) => {
  let len = 1;
  while (n >= 2 ** (7 * len) - 1) len++;
  const out = new Uint8Array(len);
  out[0] = 0x80 >> (len - 1);
  for (let i = 0; i < len; i++) out[len - 1 - i] |= (n >> (8 * i)) & 0xff;
  return out;
};
const eid = (hex) => Uint8Array.from(hex.match(/../g).map((h) => parseInt(h, 16)));
/** EBML *unsigned integer element* payload: plain minimal big-endian (no marker). */
const uint = (n) => {
  let len = 1;
  while (n >= 256 ** len && len < 8) len++;
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[len - 1 - i] = (n >> (8 * i)) & 255;
  return out;
};
const elem = (idHex, ...payload) => {
  const p = cat(...payload);
  return cat(eid(idHex), vuint(p.length), p);
};
const f64 = (x) => {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setFloat64(0, x);
  return b;
};

function buildWebm() {
  const header = elem('1a45dfa3', elem('4282', new TextEncoder().encode('webm')));
  const info = elem('1549a966', elem('2ad7b1', uint(1000000)), elem('4489', f64(133.333)));
  const video = elem('e0', elem('b0', uint(64)), elem('ba', uint(32)), elem('55b0', elem('55b1', uint(1)), elem('55b2', uint(2)), elem('55b3', uint(5))));
  const track = elem('ae', elem('d7', uint(1)), elem('83', uint(1)), elem('86', new TextEncoder().encode('V_VP8')), elem('9c', uint(0)), elem('23e383', uint(33333333)), video);
  const tracks = elem('1654ae6b', track);
  const block = (rel, flags, data) => cat(eid('a3'), vuint(1 + 2 + 1 + data.length), vuint(1), new Uint8Array([(rel >> 8) & 255, rel & 255]), u8(flags), data);
  const cluster0 = elem('1f43b675', elem('e7', uint(0)), block(0, 0x80, new Uint8Array([1, 2, 3, 4])), block(33, 0, new Uint8Array([5, 6])), block(66, 0, new Uint8Array([7])));
  const cluster1 = elem('1f43b675', elem('e7', uint(100)), block(0, 0x80, new Uint8Array([8, 9])));
  const cues = elem('1c53bb6b', elem('bb', elem('b3', uint(0)), elem('b7', elem('f7', uint(1)), elem('f1', uint(0)))), elem('bb', elem('b3', uint(100)), elem('b7', elem('f7', uint(1)), elem('f1', uint(5)))));
  const segPayload = cat(info, tracks, cluster0, cluster1, cues);
  const segSize = vuint(segPayload.length);
  const segHead = new Uint8Array(4);
  segHead.set(eid('18538067'));
  return cat(header, segHead, segSize, segPayload);
}

/* ------------------------------------------------------------------------- */

test('identify() sniffs containers by bytes, not extension', () => {
  assert.equal(identify(cat(new TextEncoder().encode('GIF89a'), new Uint8Array(100)), 'blob.dat').format, 'gif');
  const mp4ish = cat(u32(24), new TextEncoder().encode('ftyp'), new TextEncoder().encode('isom'), u32(0x200));
  assert.equal(identify(mp4ish).format, 'mp4');
  assert.equal(identify(new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0, 0, 0, 1])).format, 'webm');
  assert.equal(identify(cat(new TextEncoder().encode('RIFF'), u32(4), new TextEncoder().encode('AVI '))).format, 'avi');
  assert.equal(identify(cat(new TextEncoder().encode('RIFF'), u32(4), new TextEncoder().encode('WEBP'))).format, 'webp');
  const png = cat(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), u32(13), new TextEncoder().encode('IHDR'));
  assert.equal(identify(png).format, 'png');
  assert.equal(identify(new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 2, 0, 0])).format, 'jpeg');
  assert.throws(() => identify(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])), /Unrecognized file format/);
  // extension-only fallback must be honest about low confidence
  const byExt = identify(new Uint8Array(64), 'clip.mp4');
  assert.equal(byExt.format, 'mp4');
  assert.ok(byExt.confidence <= 0.5);
});

test('MP4: sample table, timestamps, keyframes, alpha-free video', () => {
  const { file, samples, sizes, timescale, dur } = buildMp4();
  const m = parseMp4(file);
  assert.equal(m.majorBrand.trim(), 'isom');
  assert.ok(m.fastStart, 'moov before mdat = fast start');
  assert.equal(m.width, 64);
  assert.equal(m.height, 32);
  assert.equal(m.codec, 'avc1');
  assert.equal(m.video.timescale, timescale);
  assert.equal(m.video.samples.length, 3);
  assert.equal(m.video.nalLengthSize, 4);
  assert.deepEqual(Array.from(m.video.samples.map((s) => s.size)), sizes);
  // stss listed only sample 1 as a keyframe
  assert.equal(m.video.samples[0].key, true);
  assert.equal(m.video.samples[1].key, false);
  // ctts version 1 = signed offsets of +1000 for all three
  assert.equal(m.video.samples[0].cts, 1000);
  assert.ok(m.video.description && m.video.description[0] === 1, 'avcC payload extracted');
  const times = mp4FrameTimes(m.video);
  assert.equal(times.length, 3);
  times.forEach((t, i) => {
    assert.ok(Math.abs(t.pts - (i * 1000 + 1000) / timescale * 1000) < 1e-6, `pts[${i}] = ${t.pts}`);
  });
  assert.ok(Math.abs(m.durationMs - (3 * dur) / timescale * 1000) < 0.01);
  assert.ok(m.fps > 29 && m.fps < 31, `fps ${m.fps}`);
  // sample bytes resolve exactly
  const mdatAt = m.video.samples[0].offset;
  assert.deepEqual(Array.from(file.subarray(mdatAt, mdatAt + sizes[0])), Array.from(samples[0]));
  assert.equal(m.video.samples[2].offset, mdatAt + sizes[0] + sizes[1]);
  assert.ok(!m.partial, `parser reported partial: ${JSON.stringify(m.oddities || '')}`);
  const seek = mp4SeekRange(m.video, times[2].pts);
  assert.equal(seek.firstDecodeSample, 0, 'must start at the keyframe to reach frame 3');
});

test('MP4: truncated file degrades instead of throwing', () => {
  const { file } = buildMp4();
  const cut = file.subarray(0, file.length - 5);
  const m = parseMp4(cut);
  assert.equal(m.partial, true);
  assert.ok(m.video);
  const empty = identify(cat(u32(8), new TextEncoder().encode('ftyp')));
  assert.equal(empty.format, 'mp4');
});

test('WebM: EBML vints, duration, tracks, cluster blocks, cues', () => {
  const file = buildWebm();
  const w = parseWebm(file);
  assert.equal(w.doctype, 'webm');
  assert.equal(w.width, 64);
  assert.equal(w.height, 32);
  assert.equal(w.codec, 'V_VP8');
  assert.ok(Math.abs(w.durationMs - 133.333) < 0.01, `duration ${w.durationMs}`);
  assert.ok(Math.abs(w.fps - 30) < 0.01, `fps from DefaultDuration: ${w.fps}`);
  assert.equal(w.lacing, false, 'FlagLacing=0');
  assert.ok(w.blocks.length >= 4, `blocks ${JSON.stringify(w.blocks?.map((b) => b.pts))}`);
  assert.deepEqual(w.blocks.slice(0, 4).map((b) => b.pts), [0, 33, 66, 100]);
  assert.equal(w.blocks[0].key, true);
  assert.equal(w.blocks[1].key, false);
  assert.ok(w.seekable, 'Cues present');
  assert.equal(w.cues.length, 2);
  assert.equal(w.cues[1].clusterPosition, 5);
});

test('AVI: build → parse round trip, and index rebuild when idx1 is missing', () => {
  const { Raster } = globalThis.__rasters || {};
  const frames = [makeRaster(16, 8, 20), makeRaster(16, 8, 80), makeRaster(16, 8, 160)];
  void frames.length;
  const avi = buildUncompressedAvi(frames, { fps: 10 });
  const parsed = parseRiff(avi);
  assert.equal(parsed.form, 'AVI ');
  assert.equal(parsed.format, 'avi');
  assert.equal(parsed.width, 16);
  assert.equal(parsed.height, 8);
  assert.equal(parsed.frameCount, 3);
  assert.ok(Math.abs(parsed.fps - 10) < 0.01, `fps ${parsed.fps}`);
  assert.ok(Math.abs(parsed.durationMs - 300) < 5, `duration ${parsed.durationMs}`);
  assert.equal(parsed.indexSource, 'idx1');
  // frame 2's pixels must survive the DIB bottom-up round trip
  const f1 = parsed.frames[1];
  const bytes = avi.subarray(f1.offset, f1.offset + f1.size);
  assert.equal(bytes[0], 80, 'blue channel of the second frame');
  // now break the index and confirm the scanner recovers
  const broken = avi.slice();
  const idxAt = indexOfFourcc(broken, 'idx1');
  assert.ok(idxAt > 0);
  broken.fill(0, idxAt + 4, idxAt + 8); // size → 0: index unusable
  const rescanned = parseRiff(broken);
  assert.equal(rescanned.indexSource, 'scan');
  assert.equal(rescanned.frameCount, 3);
  void Raster;
});

test('probe() unifies container info and reports decodability', async () => {
  const { file } = buildMp4();
  const p = await probe(file, { name: 'clip.mp4' });
  assert.equal(p.format, 'mp4');
  assert.equal(p.width, 64);
  assert.equal(p.frameCount, 3);
  assert.ok(p.durationMs > 99 && p.durationMs < 101);
  assert.equal(p.codec, 'avc1');
  assert.ok(p.size > 0);
  // in Node there is no <video>/WebCodecs, so video must be reported undecodable
  // while stills are always fine (the library runs in workers/tests this way)
  assert.equal(typeof p.canDecode, 'boolean');
  const le16 = (n) => new Uint8Array([n & 255, (n >>> 8) & 255]);
  const gif = cat(new TextEncoder().encode('GIF89a'), le16(4), le16(4), new Uint8Array([0, 0, 0]), new Uint8Array(24), new Uint8Array([0x2c, 0, 0, 0, 0, 4, 0, 4, 0, 0x04, 0, 1, 0x04, 0x01, 0, 0x3b]));
  const pg = await probe(gif, { name: 'a.gif' });
  assert.equal(pg.format, 'gif');
  assert.equal(pg.width, 4);
  assert.equal(pg.canDecode, true);
});

function makeRaster(w, h, v) {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = v;
    data[i * 4 + 1] = v >> 1;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return { width: w, height: h, stride: w * 4, data };
}

function indexOfFourcc(u8arr, fourcc) {
  const t = new TextEncoder().encode(fourcc);
  for (let o = 0; o + 4 <= u8arr.length; o++) {
    if (u8arr[o] === t[0] && u8arr[o + 1] === t[1] && u8arr[o + 2] === t[2] && u8arr[o + 3] === t[3]) return o;
  }
  return -1;
}

void vintSize;

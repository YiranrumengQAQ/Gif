/**
 * Decode-layer tests: the WebCodecs driver against a fake decoder (ordering,
 * backpressure, keyframe warmup, corruption recovery, abort), the frame plan
 * maths, and the GIF/element sources.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Raster } from '../src/core/buffers.js';
import { createDecodeDriver, pickDecodeRange, frameToRaster, splitAnnexB, nalUnitsToSamples, buildAvcC, readH264Sps, annexBTrack, webcodecsVideoSupport } from '../src/media/webcodecs.js';
import { createFrameSource, planFrames, chooseCapturePolicy } from '../src/media/frames.js';
import { GifWriter } from '../src/enc/gif.js';
import { lzwEncode } from '../src/enc/lzw.js';

const W = 8;
const H = 8;

/** A "frame" of our fake codec: 256 raw RGBA bytes, plus pts metadata. */
function fakeSample(color) {
  const body = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    body[i * 4] = color;
    body[i * 4 + 1] = color >> 1;
    body[i * 4 + 2] = 255 - color;
    body[i * 4 + 3] = 255;
  }
  return body;
}

/**
 * Build a synthetic track whose samples are length-prefixed raw RGBA frames, and
 * a decoder factory that "decodes" them. B-frames are simulated: pts order is
 * scrambled relative to file order, so any implementation that assumes decode
 * order == display order fails here (deliberately).
 */
function makeFakeVideo({ frames = 6, bFrames = true, corruptAt = -1, keyframeEvery = 3 } = {}) {
  const payload = [];
  const samples = [];
  const timescale = 30000;
  const dur = 3000; // 100 ms
  // decode order = file order
  for (let i = 0; i < frames; i++) {
    const body = i === corruptAt ? new Uint8Array([0, 1, 2]) : fakeSample(i * 40 + 5);
    const start = payload.reduce((a, b) => a + b.length, 0);
    payload.push(body);
    samples.push({
      offset: start,
      size: body.length,
      dts: i * dur,
      // presentation order: swap each pair, so display order != file order
      pts: !bFrames ? i * dur : i % 2 === 1 ? (i - 1) * dur : (i + 1 < frames ? (i + 1) * dur : i * dur),
      duration: dur,
      key: i % keyframeEvery === 0,
      _body: body,
    });
  }
  const bytes = new Uint8Array(payload.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const p of payload) {
    bytes.set(p, o);
    o += p.length;
  }
  const track = { samples, timescale, width: W, height: H, codecString: 'raw-rgba-test', description: null };
  const decoderFactory = (handlers) => makeFakeDecoder(bytes, track, handlers, { corruptAt });
  return { bytes, track, decoderFactory, timescale, dur };
}

function makeFakeDecoder(bytes, track, handlers, opts = {}) {
  // Mirrors the parts of the VideoDecoder contract the driver depends on:
  // async delivery, presentation-order output (with a real reorder window),
  // decodeQueueSize for backpressure, reset() dropping queued work, error
  // reporting through the handler, and configure() before reuse.
  let state = 'configured';
  const pending = [];
  const reorder = [];
  const byPts = (ts) => track.samples.find((s) => Math.abs((s.pts / track.timescale) * 1e6 - ts) < 1);
  let inFlightTimers = 0;
  const emitOldest = (delay) => {
    let m = 0;
    for (let i = 1; i < reorder.length; i++) if (reorder[i].timestamp < reorder[m].timestamp) m = i;
    const frame = reorder.splice(m, 1)[0];
    if (delay) {
      inFlightTimers++;
      setTimeout(() => {
        inFlightTimers--;
        handlers.output(frame);
      }, 0);
    } else handlers.output(frame);
  };
  const dec = {
    get state() {
      return state;
    },
    get decodeQueueSize() {
      return pending.length;
    },
    configure(config) {
      dec.config = config;
      state = 'configured';
    },
    decode(chunk) {
      if (state === 'closed') throw new Error('decoder is closed');
      if (state !== 'configured') throw new Error('not configured');
      pending.push(chunk);
      setTimeout(() => {
        const c = pending.shift();
        if (!c) return; // reset() dropped it
        const sample = byPts(c.timestamp);
        if (!sample) {
          handlers.error(new Error('unknown timestamp'));
          return;
        }
        if (opts.corruptAt === track.samples.indexOf(sample)) {
          handlers.error(new Error(`bad NAL at sample ${track.samples.indexOf(sample)}`));
          return;
        }
        reorder.push({
          timestamp: c.timestamp,
          duration: c.duration,
          type: c.type === 'key' ? 'key' : 'delta',
          visibleRect: { x: 0, y: 0, width: W, height: H },
          displayWidth: W,
          displayHeight: H,
          codedWidth: W,
          codedHeight: H,
          format: 'RGBA',
          closed: false,
          __gifxIndex: track.samples.indexOf(sample),
          async copyTo(dest) {
            if (this.closed) throw new Error('frame already closed');
            dest.set(sample._body.subarray(0, Math.min(dest.length, sample._body.length)));
          },
          close() {
            this.closed = true;
            dec.delivered++;
          },
        });
        // hold a 2-frame reorder window, then emit presentation-ordered frames
        while (reorder.length > 2) emitOldest(true);
      }, 0);
    },
    async flush() {
      // VideoDecoder.flush() resolves only once every output has been delivered.
      while (pending.length || reorder.length || inFlightTimers) {
        while (reorder.length) emitOldest(false);
        await new Promise((r) => setTimeout(r, 0));
      }
    },
    reset() {
      state = 'unconfigured';
      pending.length = 0;
      reorder.length = 0;
    },
    close() {
      state = 'closed';
    },
    delivered: 0,
  };
  return dec;
}

test('WebCodecs driver: presentation order, warmup, backpressure, all frames out', async () => {
  const { bytes, track, decoderFactory, timescale, dur } = makeFakeVideo({ frames: 6 });
  const times = track.samples
    .map((s, i) => ({ index: i, pts: (s.pts / timescale) * 1000, key: s.key }))
    .sort((a, b) => a.pts - b.pts); // mp4FrameTimes hands us presentation order
  const range = pickDecodeRange(times, track, {});
  assert.equal(range.samples.length, 6, 'no warmup needed when starting at sample 0');
  assert.equal(range.frameCount, 6);
  const driver = createDecodeDriver({ bytes, track, frameTimes: times }, range, { decoderFactory, maxInFlight: 2 });
  const out = [];
  for await (const f of driver.frames()) out.push(f);
  await driver.close();
  assert.equal(out.length, 6, `got ${out.length} frames`);
  const ptsList = out.map((f) => f.ptsMs);
  for (let i = 1; i < ptsList.length; i++) assert.ok(ptsList[i] >= ptsList[i - 1], `pts must be monotonic: ${ptsList}`);
  // presentation order must be 0..5 by pts, and each frame carries the pixels of
  // the sample whose pts it has (the pair-swapped B-frame model above)
  assert.deepEqual(out.map((f) => f.index), [0, 1, 2, 3, 4, 5], 'presentation ranks are dense and ordered');
  out.forEach((f, i) => {
    assert.ok(Math.abs(f.ptsMs - i * 100) < 1e-6, `pts[${i}] = ${f.ptsMs}`);
    assert.equal(f.raster.data[0], f.sampleIndex * 40 + 5, `frame at rank ${i} (sample ${f.sampleIndex}) pixel data`);
    assert.equal(f.raster.width, W);
    assert.equal(f.raster.opaque, true, 'opaque video frames');
  });
  assert.ok(driver.stats.fed === 6);
  void dur;
});

test('WebCodecs driver: trimming starts at the previous keyframe and still yields exact frames', async () => {
  const { bytes, track, decoderFactory, timescale } = makeFakeVideo({ frames: 9, keyframeEvery: 3 });
  const times = track.samples
    .map((s, i) => ({ index: i, pts: (s.pts / timescale) * 1000, key: s.key }))
    .sort((a, b) => a.pts - b.pts);
  // want only presentation frames 4..5
  const wantedIdx = [4, 5].map((r) => times[r].index);
  const range = pickDecodeRange(times, track, { frameIndices: wantedIdx });
  assert.ok(range.startSample % 3 === 0, `decode must start on a keyframe (got ${range.startSample})`);
  assert.ok(range.warmup >= 0);
  const driver = createDecodeDriver({ bytes, track, frameTimes: times }, range, { decoderFactory });
  const out = [];
  for await (const f of driver.frames()) out.push(f);
  await driver.close();
  assert.equal(out.length, wantedIdx.length, `emitted ${out.length} of ${wantedIdx.length}`);
  assert.ok(range.warmup > 0, 'warm-up frames were actually decoded');
});

test('WebCodecs driver: a corrupt sample is reported and the stream continues from the next keyframe', async () => {
  const { bytes, track, decoderFactory, timescale } = makeFakeVideo({ frames: 6, corruptAt: 1, keyframeEvery: 3 });
  const times = track.samples
    .map((s, i) => ({ index: i, pts: (s.pts / timescale) * 1000, key: s.key }))
    .sort((a, b) => a.pts - b.pts);
  const range = pickDecodeRange(times, track, {});
  const seen = [];
  const driver = createDecodeDriver({ bytes, track, frameTimes: times }, range, { decoderFactory, onWarning: (e, at) => seen.push([at, e.message]) });
  const out = [];
  const samples = [];
  for await (const f of driver.frames()) {
    out.push(f.index);
    samples.push(f.sampleIndex);
  }
  await driver.close();
  assert.equal(driver.warnings.length, 1, 'one corruption warning');
  assert.equal(driver.warnings[0].code, 'DECODE_CORRUPT_FRAME');
  // The error is reported asynchronously, so the driver replays from the last
  // keyframe it fed: everything from there on survives, the bad sample does not.
  assert.ok(out.length >= 3, `tail recovered after replay (got ${out.length})`);
  // ranks stay hole-y on purpose: they refer to the requested presentation
  // position, so the engine can still place each recovered frame on the timeline
  assert.ok(out.includes(2) && out.includes(4) && out.includes(5), `ranks: ${out}`);
  assert.ok(samples.includes(3) && samples.includes(4) && samples.includes(5), `recovered samples: ${samples}`);
  assert.ok(!samples.includes(1), 'the corrupt sample never appears');
  assert.ok(driver.stats.replays >= 1, 'a replay actually happened');
  assert.ok(driver.stats.fed > 6, 'replayed samples were re-fed');
  assert.equal(seen.length, 1);
  // strict mode must surface it instead
  const strict = createDecodeDriver({ bytes, track, frameTimes: times }, range, { decoderFactory, strict: true });
  let threw = null;
  try {
    for await (const f of strict.frames()) void f;
  } catch (e) {
    threw = e;
  }
  await strict.close();
  assert.equal(threw.code, 'INPUT_DECODE_FAILED');
});

test('WebCodecs driver: AbortSignal stops the decode promptly', async () => {
  const { bytes, track, decoderFactory, timescale } = makeFakeVideo({ frames: 12 });
  const times = track.samples
    .map((s, i) => ({ index: i, pts: (s.pts / timescale) * 1000, key: s.key }))
    .sort((a, b) => a.pts - b.pts);
  const range = pickDecodeRange(times, track, {});
  const ac = new AbortController();
  const driver = createDecodeDriver({ bytes, track, frameTimes: times }, range, { decoderFactory, signal: ac.signal });
  let n = 0;
  let thrown = null;
  try {
    for await (const f of driver.frames()) {
      n++;
      if (n === 2) ac.abort();
      void f;
    }
  } catch (e) {
    thrown = e;
  }
  await driver.close();
  assert.ok(n < 12, `stopped after ${n} frames`);
  assert.equal(thrown.code, 'ABORTED', 'cancellation surfaces as a typed ABORTED error');
  assert.equal(driver.error.code, 'ABORTED');
});

test('frameToRaster: copyTo path, close-once safety, opaque fill', async () => {
  const src = new Uint8Array(W * H * 4).fill(7);
  for (let i = 3; i < src.length; i += 4) src[i] = 0;
  const frame = {
    timestamp: 1234,
    duration: 40000,
    visibleRect: { x: 0, y: 0, width: W, height: H },
    format: 'BGRA',
    async copyTo(dest) {
      dest.set(src);
    },
    close() {
      this.closed = true;
    },
  };
  const r = await frameToRaster(frame, { alpha: 'keep' });
  assert.equal(r.data[0], 7);
  assert.equal(r.data[3], 0, 'alpha preserved on request');
  assert.equal(r.hasAlpha, true);
  assert.equal(r.pts, 1.234);
  const r2 = await frameToRaster(frame, { opaque: true });
  assert.equal(r2.data[3], 255);
  assert.equal(r2.opaque, true);
  r.release();
  r2.release();
});

test('webcodecsVideoSupport degrades honestly without WebCodecs', async () => {
  const s = await webcodecsVideoSupport();
  assert.equal(typeof s.ok, 'boolean');
  if (typeof VideoDecoder !== 'function') assert.equal(s.reason, 'no-VideoDecoder');
});

test('planFrames: fps target, trim, thinning and duplicate skipping', () => {
  const meta = { width: 320, height: 240, durationMs: 10000, fps: 30, frameCount: 300 };
  const p = planFrames(meta, { fps: 10 });
  assert.equal(p.count, 100, `10 fps over 10 s → 100 frames (got ${p.count})`);
  assert.equal(p.frameMs, 100);
  const t = planFrames(meta, { fps: 5, start: 2000, end: 4000 });
  assert.equal(t.count, 10, `2 s at 5 fps → 10 frames (got ${t.count})`);
  assert.equal(t.startMs, 2000);
  const thin = planFrames(meta, { fps: 30, maxFrames: 10 });
  assert.equal(thin.count, 10);
  assert.ok(thin.indices[0] === 0 && thin.indices[9] > 250, 'thinning spans the whole clip instead of truncating');
  const unique = new Set(thin.indices);
  assert.equal(unique.size, thin.count, 'no duplicated source frames');
  const noFps = planFrames({ width: 8, height: 8, durationMs: 1000, fps: 25, frameCount: 25 }, {});
  assert.equal(noFps.fps, 25, 'keeps the source rate when fps is unspecified');
  const zero = planFrames({ width: 8, height: 8, durationMs: 0, fps: 0, frameCount: 0 }, { fps: 0 });
  assert.ok(zero.count >= 1, 'a 0-second source still yields one frame');
});

test('chooseCapturePolicy: only uses play-through where it is actually cheaper', () => {
  const base = { durationMs: 20000, coveredMs: 20000, seekCostMs: 45, playCostMs: 6, rVFC: true };
  assert.equal(chooseCapturePolicy({ ...base, wantFrames: 2, sourceFrames: 600, hasTrim: false }).mode, 'seek');
  // 300 seeks @45ms = 13.5 s vs 20 s of real-time playback → seeking wins at 1x
  assert.equal(chooseCapturePolicy({ ...base, wantFrames: 300, sourceFrames: 600, hasTrim: false }).mode, 'seek');
  assert.equal(chooseCapturePolicy({ ...base, wantFrames: 6, sourceFrames: 600, hasTrim: true }).mode, 'seek', 'a trim needs a seek anyway');
  assert.equal(chooseCapturePolicy({ ...base, wantFrames: 200, sourceFrames: 600, hasTrim: false, coveredMs: 20000, playbackRate: 4 }).mode, 'play', '4x playback makes a dense scan worth it');
  assert.equal(chooseCapturePolicy({ ...base, wantFrames: 300, sourceFrames: 600, hasTrim: false, rVFC: false }).mode, 'seek');
});

test('H.264 elementary stream: start codes, SPS size, avcC builder', () => {
  // 4-byte start codes with one SPS (7), one PPS (8) and two slices (5, 1)
  const sps = new Uint8Array([0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40]);
  const pps = new Uint8Array([0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xb0]);
  const slice = new Uint8Array([0x65, 1, 2, 3]);
  const stream = new Uint8Array([...[0, 0, 0, 1], ...sps, ...[0, 0, 0, 1], ...pps, ...[0, 0, 0, 1], ...slice, ...[0, 0, 0, 1], ...[0x41, 9, 9]]);
  const nals = splitAnnexB(stream);
  assert.equal(nals.length, 4, 'start codes split correctly');
  const { samples } = nalUnitsToSamples(stream, { fps: 25 });
  assert.ok(samples.length >= 1);
  const avcC = buildAvcC(sps, pps);
  assert.equal(avcC[0], 1);
  assert.equal(avcC[1], 0x64, 'profile from SPS');
  assert.equal(avcC[3], 0x1f, 'level from SPS');
  assert.equal(avcC[4], 0xff, '4-byte NAL lengths');
  assert.equal(readH264Sps(sps), null, 'a truncated SPS must not throw — it is only a size hint');
  const t = annexBTrack(stream, { fps: 25, width: 16, height: 16 });
  assert.equal(t.track.width, 16);
  assert.ok(t.track.samples[0].key, 'IDR slice marks a keyframe');
});

test('createFrameSource: GIF round trip keeps timing, indices and palettes', async () => {
  // build a real 3-frame GIF with our own writer, then read it back
  const palette = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
  const gw = new GifWriter({ width: 4, height: 4, palette, colors: 4 });
  for (let i = 0; i < 3; i++) {
    const px = new Uint8Array(16).fill(i);
    gw.addFrame({ indices: px, x: 0, y: 0, width: 4, height: 4, delay: 7 + i * 3, palette, colors: 4 });
  }
  const bytes = gw.finish();
  const src = await createFrameSource(bytes, { name: 'x.gif' });
  assert.equal(src.kind, 'gif');
  assert.equal(src.meta.width, 4);
  assert.equal(src.meta.frameCount, 3);
  assert.equal(src.meta.durationMs, (7 + 10 + 13) * 10, 'delays are 70/100/130 ms');
  const frames = [];
  for await (const f of src.frames()) frames.push(f);
  assert.equal(frames.length, 3);
  assert.equal(frames[0].ptsMs, 0);
  assert.equal(frames[1].ptsMs, 70, 'pts is the accumulated display time');
  assert.equal(frames[0].raster.data[0], 255, 'frame 0 uses palette entry 0 = red');
  assert.equal(frames[2].raster.data[2], 255, 'frame 2 uses palette entry 3 = blue');
  const idx = [];
  for await (const f of src.indices()) idx.push(Array.from(f.indices).join(','));
  assert.equal(idx[1], new Array(16).fill(1).join(','));
  await src.close();
});

test('createFrameSource: raster passthrough and error taxonomy', async () => {
  const r = new Raster(4, 4).fill(1, 2, 3, 255);
  const src = await createFrameSource([r], { duration: 500 });
  assert.equal(src.meta.frameCount, 1);
  assert.equal(src.meta.durationMs, 500);
  let n = 0;
  for await (const f of src.frames()) {
    n++;
    assert.equal(f.raster.data[0], 1);
  }
  assert.equal(n, 1);
  await src.close();
  await assert.rejects(() => createFrameSource(new Uint8Array(0)), (e) => e.code === 'INPUT_EMPTY');
  await assert.rejects(() => createFrameSource(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), (e) => e.code === 'FORMAT_UNSUPPORTED');
  await assert.rejects(() => createFrameSource('nope.gif'), (e) => e.code === 'INPUT_INVALID');
  void lzwEncode;
});

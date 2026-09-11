# GIFX — a pure-frontend video → GIF kernel

Everything a browser needs to turn a video (or a GIF, or a pile of images) into a
small, well-built animated image — **no server, no ffmpeg, no WebAssembly, no
runtime dependencies**. Import the modules you want, or wait for the bundled
`dist/gifx.js` build; either way the whole pipeline runs on the main thread or in
a worker pool sized to `hardwareConcurrency`.

```js
import { probe, parseGif, optimizeGif, GifWriter, quantize } from '@gifx/kernel';

const info = await probe(file);                  // container, tracks, duration, codec
const gif  = parseGif(bytes);                     // frames, palettes, timing, disposal
const best = optimizeGif(gif, { check: true });   // lossless remux, pixel-verified
await write(best.bytes, 'animation.gif');
```

## Layers

| layer | what it does |
| --- | --- |
| `src/core/` | typed errors with repair hints, pooled arenas + rasters, color math (sRGB/OKLab/HSL/555), emitters, progress + capability probes |
| `src/media/` | MP4 `moov`/`stsd`/`stss`/`ctts` and WebM EBML/Cues and AVI `idx1` demuxers, format sniffing, WebCodecs decode driver, frame sources (rasters / GIF / stills / `<video>`) with capture planning |
| `src/image/` | scale (nearest/bilinear/area/bicubic/lanczos), crop, black-bar detection, rotate/flip/pad/vignette/rounded corners, a CSS-like filtergraph, dither (Floyd–Steinberg, Atkinson, Bayer, blue-noise, …), temporal dither, overlays, sprite-sheet layout |
| `src/quant/` | histogram-driven quantization: median-cut, Wu, octree, k-means (OKLab), pngquant-style variance reduction, fixed/websafe/grayscale/mono — plus the palette mapper and an auto method chooser |
| `src/enc/` | GIF LZW (spec-exact, verified against `omggif`) and `GifWriter`: global/local palettes, sub-rectangles, transparency, disposal, interlacing, comments/XMP, loop count |
| `src/dec/` | tolerant GIF reader (truncated streams, out-of-range indices, broken palettes), compositor that honours disposal, structural analysis |
| `src/optimize/` | lossless GIF remuxer: duplicate-frame merging, dirty-rect sub-framing, unchanged-pixel transparency, palette compaction, clear-interval tuning, delay quantization with drift compensation — plus an LZW size model used for target-filesize search |

## Size model

`exactLzwBytes()` is the encoder itself (byte-exact, no file assembly);
`predictLzwBytes()` is a fitted screener used to rank thousands of candidates
before any of them is measured:

```
bpp = −0.344 + 0.348·h0 + 1.183·h1      (mean |error| ≈ 40 % uncalibrated,
                                   ≈ 15 % after `observe()` sees six frames)
```

`predictCandidate({ measure: true })` agrees with the real encoder to within a few
percent, which is what the auto-search relies on. `tools/oracle-lzw.mjs`
re-derives those numbers.

## Status

Implemented and covered by `test/`: **48 tests** across containers, decode
ordering, LZW/GIF format bytes, quantization-driven optimization and the size
model.

Still on the roadmap in this repo: the `convert()` orchestration layer
(`src/engine/`, `src/workers/`, `src/io/`), the APNG / animated-WebP /
`MediaRecorder` encoders, the optional DOM editor UI, `demo/`, `docs/` and the
`dist/` build (`tools/build.mjs`). Until then, import from `src/` directly — every
module is plain ESM with no bundler requirements.

## Development

```bash
npm install          # devDeps only: esbuild + omggif/gifenc as format oracles
npm test             # node --test, no browser needed
npm run check        # parse + import + public-surface + dependency policy + tests
npm run oracle:lzw   # LZW rules, demonstrated against omggif
```

The library imports nothing but relative paths: `npm run check` fails if a file in
`src/` gains a bare or `node:` specifier, so "pure frontend" is enforced, not
just claimed.

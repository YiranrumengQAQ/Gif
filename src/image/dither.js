/**
 * GIFX Kernel — dithering.
 *
 * Two families:
 *  - **error diffusion** (Floyd–Steinberg, Atkinson, Jarvis, Stucki, Burkes,
 *    Sierra, …): best for gradients, costs one pass + error rows.
 *  - **ordered / blue-noise** (Bayer 2/4/8/16, clustered-dot, spalto,
 *    precomputed blue-noise): same visual quality on GIFs at a *fraction* of
 *    the cost and — crucially for animation — no temporal crawling between
 *    frames, which is why `auto` picks Bayer for high-color-count sources and
 *    error diffusion only for gradients.
 *
 * Every error-diffusion kernel is expressed as a sparse weight list so the
 * mapper can run them from one hot loop.
 *
 * @module image/dither
 */

/** @typedef {{dx:number, dy:number, w:number, name:string, rowSpan:number}} DiffuseKernel */

const K = (name, rowSpan, ...triples) => ({
  name,
  rowSpan,
  taps: triples.map(([dx, dy, w]) => ({ dx, dy, w })),
});

/** Error-diffusion kernels. Weights are normalized by the mapper. */
export const DIFFUSION = Object.freeze({
  none: null,
  'floyd-steinberg': K('floyd-steinberg', 1, [1, 0, 7 / 16], [-1, 1, 3 / 16], [0, 1, 5 / 16], [1, 1, 1 / 16]),
  'floyd-steinberg-lite': K('floyd-steinberg-lite', 1, [1, 0, 1 / 2], [0, 1, 1 / 4], [1, 1, 1 / 4]),
  atkinson: K('atkinson', 2, [1, 0, 1 / 8], [2, 0, 1 / 8], [-1, 1, 1 / 8], [0, 1, 1 / 8], [1, 1, 1 / 8], [0, 2, 1 / 8]),
  'jarvis-justice-ninke': K('jarvis', 2, [1, 0, 7 / 48], [2, 0, 5 / 48], [-2, 1, 3 / 48], [-1, 1, 7 / 48], [0, 1, 5 / 48], [1, 1, 3 / 48], [2, 1, 1 / 48], [-2, 2, 2 / 48], [-1, 2, 4 / 48], [0, 2, 2 / 48]),
  'stucki': K('stucki', 2, [1, 0, 8 / 42], [2, 0, 4 / 42], [-2, 1, 2 / 42], [-1, 1, 4 / 42], [0, 1, 8 / 42], [1, 1, 4 / 42], [2, 1, 2 / 42], [-2, 2, 1 / 42], [-1, 2, 2 / 42], [0, 2, 4 / 42], [1, 2, 2 / 42], [2, 2, 1 / 42]),
  'burkes': K('burkes', 1, [1, 0, 8 / 32], [2, 0, 4 / 32], [-2, 1, 2 / 32], [-1, 1, 4 / 32], [0, 1, 8 / 32], [1, 1, 4 / 32], [2, 1, 2 / 32]),
  'sierra': K('sierra', 2, [1, 0, 5 / 32], [2, 0, 3 / 32], [-2, 1, 2 / 32], [-1, 1, 4 / 32], [0, 1, 5 / 32], [1, 1, 4 / 32], [2, 1, 2 / 32], [-1, 2, 2 / 32], [0, 2, 3 / 32], [1, 2, 2 / 32]),
  'sierra-2row': K('sierra-2row', 1, [1, 0, 4 / 16], [2, 0, 1 / 16], [-2, 1, 1 / 16], [-1, 1, 2 / 16], [0, 1, 4 / 16], [1, 1, 2 / 16], [2, 1, 1 / 16]),
  'sierra-lite': K('sierra-lite', 1, [1, 0, 2 / 4], [-1, 1, 1 / 4], [0, 1, 1 / 4]),
  'one-dimensional': K('one-d', 0, [1, 0, 1]),
  'two-dimensional': K('two-d', 1, [1, 0, 1 / 2], [0, 1, 1 / 2]),
  'filter-lite': K('filter-lite', 1, [1, 0, 1 / 2], [-1, 1, 1 / 4], [0, 1, 1 / 4]),
  'slerr': K('slerr', 1, [1, 0, 1 / 2], [0, 1, 1 / 2]),
  'false-floyd': K('false-floyd', 1, [1, 0, 1 / 2], [0, 1, 1 / 2]),
  'favg': K('favg', 1, [1, 0, 15 / 65], [2, 0, 12 / 65], [-1, 1, 15 / 65], [0, 1, 8 / 65]),
});

export const DIFFUSION_NAMES = Object.keys(DIFFUSION).filter((k) => DIFFUSION[k]);

/** Bayer / ordered matrices. 4×4 and 8×8 are exact; 16×16 is derived. */
export const BAYER2 = new Uint8Array([0, 2, 3, 1]);
export const BAYER4 = new Uint8Array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]);
export const BAYER8 = (() => {
  const out = new Uint8Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const q = ((y & 1) << 1) | (x & 1);
      const i0 = (((y >> 1) & 1) << 1) | ((x >> 1) & 1);
      const i1 = (((y >> 2) & 1) << 1) | ((x >> 2) & 1);
      out[y * 8 + x] = 4 * BAYER4[i1 * 4 + i0] + q;
    }
  }
  return out;
})();
export const BAYER16 = (() => {
  const out = new Uint8Array(256);
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const q = ((y & 1) << 1) | (x & 1);
      const i0 = (((y >> 1) & 3) << 2) | ((x >> 1) & 3);
      const i1 = (((y >> 3) & 1) << 1) | ((x >> 3) & 1);
      out[y * 16 + x] = 4 * BAYER8[i1 * 16 + i0] + q;
    }
  }
  return out;
})();
/** Clustered-dot (4×4 "void-and-cluster"-ish) — softer than Bayer on faces. */
export const CLUSTERED4 = new Uint8Array([12, 16, 11, 13, 14, 8, 9, 10, 6, 2, 1, 7, 4, 0, 3, 5]);
/**
 * 8×8 blue-noise-ish matrix (void-and-cluster result, hardcoded to stay
 * deterministic and allocation-free). Good low-cost alternative to Bayer with
 * less visible diagonal texture.
 */
export const BLUENOISE8 = new Uint8Array([
  26, 16, 31, 36, 21, 2, 5, 24,
  41, 48, 12, 56, 46, 33, 50, 14,
  6, 29, 0, 20, 10, 38, 35, 18,
  54, 44, 62, 49, 58, 1, 45, 3,
  19, 34, 8, 27, 4, 30, 13, 55,
  63, 52, 60, 23, 51, 47, 61, 40,
  11, 37, 28, 15, 39, 7, 25, 57,
  32, 43, 22, 42, 17, 53, 26, 9,
]);

export const ORDERED = Object.freeze({
  bayer2: { name: 'bayer2', m: BAYER2, size: 2 },
  bayer4: { name: 'bayer4', m: BAYER4, size: 4 },
  bayer8: { name: 'bayer8', m: BAYER8, size: 8 },
  bayer16: { name: 'bayer16', m: BAYER16, size: 16 },
  clustered4: { name: 'clustered4', m: CLUSTERED4, size: 4 },
  bluenoise8: { name: 'bluenoise8', m: BLUENOISE8, size: 8 },
  void: { name: 'void', m: BLUENOISE8, size: 8 },
});

/**
 * Ordered thresholds are compared against a per-pixel value in [0,1); the
 * normalized threshold array is `m[i] / (size*size)`.
 */
const normCache = new Map();
export function normalizedThresholds(name) {
  if (normCache.has(name)) return normCache.get(name);
  const o = ORDERED[name];
  if (!o) return null;
  const inv = 1 / (o.size * o.size);
  const t = new Float32Array(o.m.length);
  for (let i = 0; i < t.length; i++) t[i] = (o.m[i] + 0.5) * inv - 0.5;
  normCache.set(name, t);
  return t;
}

export function resolveDiffusion(name) {
  if (!name || name === 'none') return null;
  const direct = DIFFUSION[name];
  if (direct !== undefined) return direct;
  const alias = {
    fs: 'floyd-steinberg',
    floyd: 'floyd-steinberg',
    'floyd-steinberg-lite': 'floyd-steinberg-lite',
    otto: 'atkinson',
    'mac-atkinson': 'atkinson',
    jjn: 'jarvis-justice-ninke',
    'jarvis': 'jarvis-justice-ninke',
    'pennerton': 'burkes',
    'five-line': 'sierra',
    'two-row': 'sierra-2row',
    'sierra3': 'sierra',
    'sierra2': 'sierra-2row',
    'sierra-lite': 'sierra-lite',
    'stucki': 'stucki',
  }[name];
  return alias ? DIFFUSION[alias] || null : null;
}

export function isOrdered(name) {
  return !!name && !!ORDERED[name];
}

/**
 * `auto` policy: pick the dither that costs least for this content.
 *
 * @param {{colors:number, motion?:number, grain?:number, hasAlpha?:boolean, isPhoto?:boolean}} ctx
 * @returns {string|null}
 */
export function autoDither(ctx = {}) {
  const colors = ctx.colors || 64;
  const motion = ctx.motion ?? 0.5;
  if (colors >= 128 && motion > 0.65) return null; // too expensive temporally, and rarely needed
  if (colors >= 64 && ctx.isPhoto) return 'bluenoise8';
  if (colors <= 16) return 'floyd-steinberg';
  if (colors <= 32) return 'atkinson';
  return 'bayer8';
}

/**
 * GIFX Kernel — memory arena & typed-array pool.
 *
 * Rasterizing 1080p frames at 60 fps allocates ~8 MB/frame. Without pooling,
 * V8's GC pauses show up as multi-hundred-millisecond hitches during long
 * conversions and the tab can hit the browser's tab memory limit on 4K sources.
 *
 * The arena hands out `Uint8Array`/`Uint32Array`/`Uint16Array`/`Float32Array`
 * views backed by slabs, recycles them on `release()`, trims oversized slabs,
 * and reports a memory ledger so the scheduler can apply backpressure.
 *
 * @module core/buffers
 */
import { GifxError, ErrorCode } from './errors.js';

/** @typedef {Uint8Array|Uint16Array|Uint32Array|Int32Array|Float32Array} TypedView */

const CLASS = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, i32: Int32Array, f32: Float32Array };
const BYTES = { u8: 1, u16: 2, u32: 4, i32: 4, f32: 4 };

export class Arena {
  /**
   * @param {object} [opts]
   * @param {number} [opts.slabBytes=4194304] size of each backing slab
   * @param {number} [opts.maxBytes=0] hard cap, 0 = auto (see {@link Arena.recommendedBudget})
   * @param {number} [opts.maxRecycledSlabs=8] how many freed buffers to keep warm
   */
  constructor(opts = {}) {
    this.slabBytes = nextPow2(opts.slabBytes ?? 4 << 20);
    this.maxBytes = opts.maxBytes || Arena.recommendedBudget();
    this.maxRecycledSlabs = opts.maxRecycledSlabs ?? 8;
    /** @type {Map<string, {buffers: Float64Array[], bytes: number}>} */
    this._recycled = new Map();
    this._live = new Set();
    this.allocatedSlabBytes = 0;
    this.peakBytes = 0;
    this.stats = { requested: 0, reused: 0, slabAllocs: 0, slabTrims: 0, releases: 0, doubleFree: 0 };
    this._listeners = new Set();
    this.disposed = false;
  }

  /**
   * Budget heuristic: browsers cap a tab around 2–4 GB but the practical
   * ceiling for a JS heap is much lower. We take deviceMemory (in GiB, when
   * exposed) minus a safety margin, else assume 4 GiB of usable tab space.
   */
  static recommendedBudget() {
    let gb = 4;
    if (typeof navigator !== 'undefined' && typeof navigator.deviceMemory === 'number') {
      gb = Math.max(1.5, navigator.deviceMemory * 0.5);
    }
    if (typeof performance !== 'undefined' && performance.memory && performance.memory.jsHeapSizeLimit) {
      gb = Math.min(gb, (performance.memory.jsHeapSizeLimit / (1 << 30)) * 0.6);
    }
    return Math.max(256 << 20, Math.floor(gb * (1 << 30)));
  }

  get inUseBytes() {
    let n = 0;
    for (const b of this._live) n += b.byteLength;
    return n;
  }
  get recycledBytes() {
    let n = 0;
    for (const v of this._recycled.values()) n += v.bytes;
    return n;
  }
  get committedBytes() {
    return this.allocatedSlabBytes;
  }
  get utilization() {
    return this.maxBytes ? this.committedBytes / this.maxBytes : 0;
  }

  onChange(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }
  _notify(kind, info) {
    if (!this._listeners.size) return;
    const snap = { kind, ...info, inUse: this.inUseBytes, committed: this.committedBytes, peak: this.peakBytes };
    for (const fn of this._listeners) {
      try {
        fn(snap);
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * @param {'u8'|'u16'|'u32'|'i32'|'f32'} kind
   * @param {number} count element count (not bytes)
   * @param {object} [opts]
   * @param {boolean} [opts.zero=true] zero-fill
   * @param {string} [opts.tag] label for diagnostics
   * @returns {TypedView & { __arena: object }}
   */
  acquire(kind, count, opts = {}) {
    if (this.disposed) throw new GifxError(ErrorCode.STATE_ERROR, 'arena disposed');
    if (!(count > 0)) count = 1;
    this.stats.requested++;
    const need = count * BYTES[kind];
    const key = `${kind}:${pow2Class(need)}`;
    const bucket = this._recycled.get(key);
    if (bucket && bucket.buffers.length) {
      const buf = bucket.buffers.pop();
      bucket.bytes -= buf.byteLength;
      this.stats.reused++;
      if (opts.zero !== false && buf.fill) buf.fill(0);
      this._live.add(buf);
      this._peak(buf.byteLength);
      return buf;
    }
    if (this.committedBytes + need > this.maxBytes) {
      this._collect();
      if (this.committedBytes + need > this.maxBytes) {
        throw new GifxError(ErrorCode.OUT_OF_MEMORY, `arena budget exceeded: need ${need} bytes, committed ${this.committedBytes}, cap ${this.maxBytes}`, {
          hint: `Reduce maxFramesInFlight / output size, or raise arena.maxBytes. Tag: ${opts.tag || '-'}`,
          data: { need, committed: this.committedBytes, cap: this.maxBytes, tag: opts.tag },
        });
      }
    }
    const Ctor = CLASS[kind];
    const buf = new Ctor(count);
    buf.__arena = { kind, count, key, tag: opts.tag };
    this.allocatedSlabBytes += buf.byteLength;
    this.stats.slabAllocs++;
    this._live.add(buf);
    this._peak(buf.byteLength);
    return buf;
  }

  _peak(n) {
    const now = this.inUseBytes;
    if (now > this.peakBytes) this.peakBytes = now;
    if (n && this.stats.requested % 64 === 0) this._notify('alloc', { bytes: n });
  }

  /** @param {TypedView} buf */
  release(buf) {
    if (!buf || !buf.__arena) return false;
    if (!this._live.delete(buf)) {
      this.stats.doubleFree++;
      return false;
    }
    const meta = buf.__arena;
    let bucket = this._recycled.get(meta.key);
    if (!bucket) this._recycled.set(meta.key, (bucket = { buffers: [], bytes: 0 }));
    // Trim pathologically large slabs so a single 4K job doesn't pin memory.
    if (buf.byteLength > this.slabBytes * 2 && this._live.size === 0) {
      this.allocatedSlabBytes -= buf.byteLength;
      this.stats.slabTrims++;
      delete buf.__arena;
      this._notify('trim', { bytes: buf.byteLength });
      return true;
    }
    if (bucket.buffers.length < this.maxRecycledSlabs) {
      bucket.buffers.push(buf);
      bucket.bytes += buf.byteLength;
    } else {
      this.allocatedSlabBytes -= buf.byteLength;
    }
    this.stats.releases++;
    delete buf.__arena;
    return true;
  }

  /** Give back every recycled slab (call between jobs). */
  trim() {
    let freed = 0;
    for (const [key, bucket] of this._recycled) {
      for (const b of bucket.buffers) freed += b.byteLength;
      this._recycled.delete(key);
    }
    this.allocatedSlabBytes -= freed;
    if (freed) this._notify('trimAll', { bytes: freed });
    return freed;
  }

  /** Best-effort reclaim before a big allocation fails. */
  _collect() {
    this.trim();
    if (typeof globalThis !== 'undefined' && globalThis.gc) {
      try {
        globalThis.gc();
      } catch {
        /* --expose-gc only */
      }
    }
  }

  snapshot() {
    return {
      inUse: this.inUseBytes,
      recycled: this.recycledBytes,
      committed: this.committedBytes,
      peak: this.peakBytes,
      cap: this.maxBytes,
      live: this._live.size,
      stats: { ...this.stats },
      buckets: [...this._recycled.entries()]
        .map(([k, v]) => ({ key: k, count: v.buffers.length, bytes: v.bytes }))
        .sort((a, b) => b.bytes - a.bytes)
        .slice(0, 12),
    };
  }

  dispose() {
    this._live.clear();
    this._recycled.clear();
    this.allocatedSlabBytes = 0;
    this.disposed = true;
    this._listeners.clear();
  }
}

function pow2Class(bytes) {
  return nextPow2(bytes);
}
export function nextPow2(x) {
  let p = 64;
  while (p < x) p *= 2;
  return p;
}

/** Shared default arena (per-realm). Libraries users can override. */
let defaultArena = null;
export function getArena(opts) {
  if (!defaultArena || opts) defaultArena = new Arena(opts);
  return defaultArena;
}
export function setArena(a) {
  defaultArena = a;
}

/**
 * RGBA frame buffer with stride support and cheap clone/move helpers.
 *
 * Layout is tightly packed RGBA (stride = width*4) by default; stride is kept
 * as a field so dirty-rect work and padded intermediates never need copies.
 */
export class Raster {
  /**
   * @param {number} width
   * @param {number} height
   * @param {Uint8Array} [data]
   * @param {Arena} [arena]
   */
  constructor(width, height, data, arena) {
    this.width = width;
    this.height = height;
    this.stride = width * 4;
    this.arena = arena || null;
    if (data) this.data = data;
    else {
      const buf = (arena || getArena()).acquire('u8', width * height * 4, { tag: `raster:${width}x${height}` });
      this.data = buf;
      this._owned = true;
    }
    /** alpha channel present? set by callers who scanned it */
    this.hasAlpha = false;
    /**
     * Opposite of hasAlpha: the producer guarantees alpha === 255 everywhere.
     * Filters use it to take a straight-blend path instead of premultiplied
     * (≈15% faster scaling, and it stops opaque video from acquiring a fake
     * alpha edge). Never guess it — only set it from a real scan or from
     * `alpha:'discard'` decoding.
     */
    this.opaque = false;
    this.pts = 0;
    this.duration = 0;
    this.index = -1;
  }

  static alloc(w, h, arena, opts) {
    return new Raster(w, h, undefined, arena).configure(opts);
  }
  configure(opts = {}) {
    if (opts.pts != null) this.pts = opts.pts;
    if (opts.duration != null) this.duration = opts.duration;
    if (opts.index != null) this.index = opts.index;
    if (opts.hasAlpha != null) this.hasAlpha = opts.hasAlpha;
    if (opts.opaque != null) this.opaque = opts.opaque;
    return this;
  }

  /**
   * Hand the backing buffer back to the arena. Idempotent, so pipeline code can
   * `finally`-release without tracking who owns what.
   */
  release() {
    if (!this.data) return false;
    const owned = this._owned;
    const arena = this.arena;
    const buf = this.data;
    this.data = null;
    if (owned && arena) arena.release(buf);
    return true;
  }

  get bytes() {
    return this.data.byteLength;
  }
  get pixels() {
    return this.width * this.height;
  }

  /** Fill with a solid color (used by pad/background/letterbox). */
  fill(r, g, b, a = 255) {
    const d = this.data;
    for (let i = 0; i < this.stride; i += 4) {
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = a;
    }
    for (let y = 1; y < this.height; y++) d.copyWithin(y * this.stride, 0, this.stride);
    return this;
  }

  /** Alpha-preserving transparent clear. */
  clearAlpha() {
    const d = this.data;
    for (let i = 0; i < this.stride; i += 4) d[i + 3] = 0;
    for (let y = 1; y < this.height; y++) d.copyWithin(y * this.stride, 0, this.stride);
    return this;
  }

  clone(arena) {
    const out = new Raster(this.width, this.height, undefined, arena || this.arena);
    out.data.set(this.data);
    out.pts = this.pts;
    out.duration = this.duration;
    out.index = this.index;
    out.hasAlpha = this.hasAlpha;
    return out;
  }

  /** Crop region in place (zero-copy view via a new Raster over a subarray). */
  subview(x, y, w, h) {
    const off = y * this.stride + x * 4;
    const need = w * 4;
    if (need === this.stride) {
      const r = new Raster(w, h, this.data.subarray(off, off + need * h));
      r.stride = this.stride;
      return r;
    }
    const out = new Raster(w, h);
    for (let row = 0; row < h; row++) out.data.set(this.data.subarray(off + row * this.stride, off + row * this.stride + need), row * need);
    return out;
  }

  /** Composite `src` at (x,y). Supports partial out-of-bounds clipping. */
  composite(src, x = 0, y = 0, opacity = 1, mode = 'normal') {
    const dst = this.data;
    const sx0 = Math.max(0, -x);
    const sy0 = Math.max(0, -y);
    const w = Math.min(src.width - sx0, this.width - x - (x < 0 ? 0 : 0) - Math.max(0, -x));
    const h = Math.min(src.height - sy0, this.height - y);
    const cw = Math.max(0, w);
    const ch = Math.max(0, h);
    const sStride = src.stride;
    for (let row = 0; row < ch; row++) {
      const dy = (y + sy0 + row) * this.stride + (x + sx0) * 4;
      const sy = (sy0 + row) * sStride + sx0 * 4;
      for (let col = 0; col < cw; col++) {
        const di = dy + col * 4;
        const si = sy + col * 4;
        const sa = src.data[si + 3];
        if (sa === 0) continue;
        if (mode === 'copy') {
          dst[di] = src.data[si];
          dst[di + 1] = src.data[si + 1];
          dst[di + 2] = src.data[si + 2];
          dst[di + 3] = sa;
          continue;
        }
        const a = (sa / 255) * opacity;
        if (a >= 0.996) {
          dst[di] = src.data[si];
          dst[di + 1] = src.data[si + 1];
          dst[di + 2] = src.data[si + 2];
          dst[di + 3] = Math.max(dst[di + 3], sa);
          continue;
        }
        const ia = 1 - a;
        dst[di] = src.data[si] * a + dst[di] * ia;
        dst[di + 1] = src.data[si + 1] * a + dst[di + 1] * ia;
        dst[di + 2] = src.data[si + 2] * a + dst[di + 2] * ia;
        dst[di + 3] = sa + dst[di + 3] * ia;
      }
    }
    return this;
  }

  /** True if any pixel has alpha < 255 (strided scan, early exit). */
  detectAlpha(sampleStep = 1) {
    const d = this.data;
    for (let i = 3; i < d.length; i += 4 * sampleStep) if (d[i] < 255) return true;
    return false;
  }

  toImageData() {
    if (typeof ImageData !== 'undefined') return new ImageData(new Uint8ClampedArray(this.data.buffer, this.data.byteOffset, this.data.length), this.width, this.height);
    return { data: this.data, width: this.width, height: this.height };
  }

  /** @param {Transferable[]} [transfers] */
  transfer() {
    return { buffer: this.data.buffer, byteOffset: this.data.byteOffset, byteLength: this.data.byteLength, width: this.width, height: this.height, stride: this.stride, pts: this.pts, duration: this.duration, index: this.index, hasAlpha: this.hasAlpha };
  }
  static fromTransfer(t, dataOverride) {
    const r = new Raster(t.width, t.height, dataOverride || new Uint8Array(t.buffer, t.byteOffset, t.byteLength));
    r.stride = t.stride;
    r.pts = t.pts;
    r.duration = t.duration;
    r.index = t.index;
    r.hasAlpha = !!t.hasAlpha;
    return r;
  }

  dispose() {
    this.release();
  }

  /** Rough 0..1 perceptual difference; used by dedup + adaptive decisions. */
  diff(other, step = 1) {
    const a = this.data;
    const b = other.data;
    if (a.length !== b.length) return 1;
    let sum = 0;
    let n = 0;
    for (let i = 0; i < a.length; i += 4 * step) {
      sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
      n += 3;
    }
    return sum / (n * 255);
  }
}

/**
 * 3-2-1 stack for the quantizers/encoders: index maps and scratch buffers
 * that would otherwise be reallocated per frame.
 */
export class Scratch {
  constructor(width, height) {
    this.resize(width, height);
  }
  resize(w, h) {
    const n = w * h;
    if (!this.index || this.index.length < n) this.index = new Uint8Array(n);
    if (!this.index16 || this.index16.length < n) this.index16 = new Uint16Array(n);
    if (!this.err) this.err = new Int16Array(n * 3);
    if (!this.hist) this.hist = new Int32Array(65536);
    this.width = w;
    this.height = h;
    return this;
  }
  get pixels() {
    return this.width * this.height;
  }
}

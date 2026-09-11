/**
 * GIFX Kernel — event emitter, deferreds, cancellation, throttling.
 *
 * @module core/events
 */
import { GifxError, ErrorCode } from './errors.js';

/** Minimal, allocation-light, re-entrant-safe event emitter. */
export class Emitter {
  constructor() {
    /** @type {Map<string, Set<Function>>} */
    this._listeners = new Map();
    /** @type {Map<string, Set<Function>>} */
    this._once = new Map();
    this._maxListeners = 0; // 0 = unlimited
    this._silent = false;
  }

  on(type, fn) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    let set = this._listeners.get(type);
    if (!set) this._listeners.set(type, (set = new Set()));
    set.add(fn);
    this._checkMax(type, set);
    return this;
  }

  once(type, fn) {
    let set = this._once.get(type);
    if (!set) this._once.set(type, (set = new Set()));
    set.add(fn);
    return this;
  }

  off(type, fn) {
    if (type === undefined) {
      this._listeners.clear();
      this._once.clear();
      return this;
    }
    if (fn === undefined) {
      this._listeners.delete(type);
      this._once.delete(type);
      return this;
    }
    this._listeners.get(type)?.delete(fn);
    this._once.get(type)?.delete(fn);
    return this;
  }

  /** @returns {number} number of listeners invoked */
  emit(type, payload) {
    if (this._silent) return 0;
    let n = 0;
    const set = this._listeners.get(type);
    if (set && set.size) {
      // snapshot so mutation during iteration is safe
      for (const fn of [...set]) {
        try {
          fn(payload, type);
          n++;
        } catch (err) {
          this._report(type, err);
        }
      }
    }
    const onceSet = this._once.get(type);
    if (onceSet && onceSet.size) {
      this._once.delete(type);
      for (const fn of [...onceSet]) {
        try {
          fn(payload, type);
          n++;
        } catch (err) {
          this._report(type, err);
        }
      }
    }
    return n;
  }

  /** Emit but coalesce multiple rapid events into one microtask-delivered call. */
  emitCoalesced(type, payloadGetter) {
    if (this._coalesce && this._coalesce.has(type)) {
      this._coalesce.get(type).payload = payloadGetter;
      return;
    }
    if (!this._coalesce) this._coalesce = new Map();
    const slot = { payload: payloadGetter };
    this._coalesce.set(type, slot);
    queueMicrotask(() => {
      this._coalesce.delete(type);
      this.emit(type, typeof slot.payload === 'function' ? slot.payload() : slot.payload);
    });
  }

  listenerCount(type) {
    return (this._listeners.get(type)?.size || 0) + (this._once.get(type)?.size || 0);
  }

  _checkMax(type, set) {
    if (this._maxListeners && set.size > this._maxListeners) {
      // eslint-disable-next-line no-console
      console.warn(`[gifx] possible listener leak on "${type}": ${set.size} listeners`);
    }
  }

  _report(type, err) {
    if (type !== 'error' && this.listenerCount('error')) this.emit('error', { type, error: err });
    // eslint-disable-next-line no-console
    else console.error(`[gifx] listener for "${type}" threw`, err);
  }

  removeAllListeners() {
    return this.off();
  }

  dispose() {
    this._listeners.clear();
    this._once.clear();
    this._silent = true;
  }
}

/**
 * Token combining AbortController with a "graceful stop" request, so in-flight
 * work can drain cleanly instead of being hard-killed.
 */
export class CancelToken {
  /** @param {AbortSignal} [external] */
  constructor(external) {
    this._ctrl = new AbortController();
    this._graceful = false;
    this.reason = null;
    if (external) {
      if (external.aborted) this.abort(external.reason);
      else
        external.addEventListener(
          'abort',
          () => this.abort(external.reason),
          { once: true },
        );
    }
  }

  get signal() {
    return this._ctrl.signal;
  }

  get cancelled() {
    return this._ctrl.signal.aborted;
  }

  /** True once `stop()` was requested — producers stop queueing new work. */
  get stopping() {
    return this._graceful || this._ctrl.signal.aborted;
  }

  /** Ask for a graceful drain (finish what is in flight, then stop). */
  stop(reason = 'stop requested') {
    this._graceful = true;
    this.reason = this.reason || reason;
  }

  abort(reason = 'aborted') {
    this._graceful = true;
    this.reason = this.reason || reason;
    if (!this._ctrl.signal.aborted) this._ctrl.abort(typeof reason === 'string' ? new Error(reason) : reason);
  }

  throwIfCancelled() {
    if (this._ctrl.signal.aborted) {
      throw new GifxError(ErrorCode.ABORTED, `Operation aborted${this.reason ? ` (${this.reason})` : ''}`);
    }
  }
}



/** Promise + resolve/reject handles. */
export function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  promise.catch(() => {}); // avoid unhandled rejection noise for fire-and-forget awaits
  return { promise, resolve, reject };
}

/** Sleep; resolves immediately when ms<=0. */
export function sleep(ms, token) {
  if (!(ms > 0)) return Promise.resolve();
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    token?.signal?.addEventListener?.(
      'abort',
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}

/** `await yieldToUI()` between tight loops so the page stays responsive. */
export function yieldToUI() {
  return new Promise((r) => (typeof queueMicrotask === 'function' ? queueMicrotask(r) : setTimeout(r, 0)));
}

/**
 * Rate-limited progress reporter.
 *
 * Progress callbacks are the #1 cause of jank in naive converters: a callback
 * per frame × 3000 frames × React re-render = frozen tab. This collapses calls
 * to at most one per `interval` ms plus always the terminal one.
 */
export class ProgressReporter {
  /**
   * @param {(p:object)=>void} [onProgress]
   * @param {{interval?:number, key?:string}} [opts]
   */
  constructor(onProgress, opts = {}) {
    this.cb = typeof onProgress === 'function' ? onProgress : null;
    this.interval = opts.interval ?? 66;
    this.last = -Infinity;
    this._pending = null;
    this._timer = 0;
    this.value = { stage: 'init', pct: 0, done: 0, total: 0, stagePct: 0, bytes: 0, fps: 0, eta: null };
    this.startedAt = Date.now();
    this.stages = new Map();
  }

  /** Begin a stage; nested stages are replaced, timings recorded. */
  stage(name, total = 0, meta) {
    this.stages.set(name, { name, startedAt: Date.now(), endedAt: 0, total, done: 0, meta });
    return this._publish(0, true);
  }

  tick(done, total, extra) {
    return this._publish(total ? done / total : 0, false, { done, total, ...extra });
  }

  inc(delta = 1, total, extra) {
    const s = [...this.stages.values()].pop();
    if (s) s.done += delta;
    return this._publish(total ? (s ? s.done / total : 0) : 0, false, {
      done: s ? s.done : 0,
      total: total ?? (s ? s.total : 0),
      ...extra,
    });
  }

  bytes(n) {
    return this._publish(undefined, false, { bytes: n });
  }

  finish() {
    this._flush();
    this._publish(1, true, { finished: true });
    return this.value;
  }

  _publish(stagePct, force, extra) {
    const v = this.value;
    if (stagePct !== undefined) {
      v.stagePct = Math.max(0, Math.min(1, stagePct));
      const s = [...this.stages.values()].pop();
      if (s) s.done = s.total ? s.total * v.stagePct : s.done;
    }
    if (extra) Object.assign(v, extra);
    const now = Date.now();
    v.elapsed = now - this.startedAt;
    if (v.total > 0 && v.done > 0) {
      const per = v.elapsed / v.done;
      v.eta = Math.max(0, Math.round(((v.total - v.done) * per) / 100) * 100);
      v.fps = (v.done / Math.max(1, v.elapsed)) * 1000;
    }
    v.pct = this._overallPct();
    this._pending = v;
    if (force || now - this.last >= this.interval) this._flush();
    return v;
  }

  /**
   * Weighted overall percentage across the standard 5-stage pipeline.
   * Stages already completed count as 1, the currently active stage
   * proportionally, later stages as 0.
   */
  _overallPct() {
    const order = PROGRESS_WEIGHTS.map((w) => w[0]);
    const started = [...this.stages.keys()];
    const lastIdx = started.length ? order.indexOf(started[started.length - 1]) : -1;
    let acc = 0;
    for (const [stage, weight] of PROGRESS_WEIGHTS) {
      const i = order.indexOf(stage);
      const s = this.stages.get(stage);
      if (s) acc += weight * (s.total ? clamp01(s.done / s.total) : 1);
      else if (lastIdx >= 0 && i < lastIdx) acc += weight;
    }
    return Math.max(0, Math.min(1, acc));
  }

  _flush() {
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = 0;
    }
    if (!this.cb || !this._pending) return;
    this.last = Date.now();
    const snapshot = { ...this._pending, stages: this.stageSummary() };
    this._pending = null;
    try {
      this.cb(snapshot);
    } catch {
      /* user callback must not kill the pipeline */
    }
  }

  stageSummary() {
    const out = [];
    for (const s of this.stages.values())
      out.push({
        name: s.name,
        total: s.total,
        done: Math.round(s.done),
        ms: (s.endedAt || Date.now()) - s.startedAt,
        meta: s.meta,
      });
    return out;
  }

  endStage(name) {
    const s = this.stages.get(name) || [...this.stages.values()].pop();
    if (s) s.endedAt = Date.now();
  }
}

const PROGRESS_WEIGHTS = [
  ['probe', 0.06],
  ['decode', 0.44],
  ['filter', 0.2],
  ['quantize', 0.12],
  ['encode', 0.18],
];

export function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Backpressure channel: bounded async queue with transferable support.
 *
 * @template T
 */
export class Channel {
  /**
   * @param {number} [capacity=8]
   * @param {(msg:object)=>void} [onCancel]
   */
  constructor(capacity = 8, onCancel) {
    this.capacity = Math.max(1, capacity | 0);
    /** @type {{value:T}|{error:*}[] */
    this._q = [];
    this._pushers = [];
    this._takers = [];
    this._closed = false;
    this._err = null;
    this.onCancel = onCancel;
    this.highWater = 0;
    this.produced = 0;
    this.consumed = 0;
  }

  get length() {
    return this._q.length;
  }
  get backpressured() {
    return this._q.length >= this.capacity;
  }

  async push(value, meta) {
    if (this._closed) throw new Error('channel closed');
    this.produced++;
    if (this._q.length < this.capacity) {
      this._q.push({ value, meta });
      this._wakeTaker();
      return;
    }
    this.highWater = Math.max(this.highWater, this._q.length);
    await new Promise((resolve, reject) => this._pushers.push({ value, meta, resolve, reject }));
  }

  async *iterate() {
    while (true) {
      const item = await this.take();
      if (item === Channel.DONE) return;
      yield item;
    }
  }

  /** @returns {Promise<T|typeof Channel.DONE>} */
  take() {
    if (this._q.length) {
      const it = this._q.shift();
      this.consumed++;
      this._wakePusher();
      return Promise.resolve(it.value);
    }
    if (this._err) return Promise.reject(this._err);
    if (this._closed) return Promise.resolve(Channel.DONE);
    return new Promise((resolve, reject) => this._takers.push({ resolve, reject }));
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    while (this._takers.length) this._takers.shift().resolve(Channel.DONE);
    while (this._pushers.length) this._pushers.shift().resolve();
  }

  fail(err) {
    this._err = err;
    while (this._takers.length) this._takers.shift().reject(err);
    while (this._pushers.length) this._pushers.shift().reject(err);
  }

  _wakeTaker() {
    if (!this._takers.length) return;
    const it = this._q.shift();
    if (!it) return;
    this.consumed++;
    this._takers.shift().resolve(it.value);
  }

  _wakePusher() {
    if (!this._pushers.length) return;
    const p = this._pushers.shift();
    this._q.push({ value: p.value, meta: p.meta });
    p.resolve();
  }
}
Channel.DONE = Symbol('channel.done');

/** Retry helper with exponential backoff and jitter. */
export async function withRetry(fn, { retries = 2, baseDelay = 60, maxDelay = 2000, onRetry, isRetryable, signal } = {}) {
  let attempt = 0;
  for (;;) {
    try {
      return await fn(attempt);
    } catch (err) {
      attempt++;
      const ok = (isRetryable ? isRetryable(err) : err && err.retryable) && attempt <= retries;
      if (!ok || signal?.aborted) throw err;
      const delay = Math.min(maxDelay, baseDelay * 2 ** (attempt - 1)) * (0.7 + Math.random() * 0.6);
      onRetry?.(err, attempt, delay);
      await sleep(delay);
    }
  }
}

/**
 * GIFX Kernel — logging.
 *
 * Levels: `silent < error < warn < info < debug < trace`. A ring buffer keeps
 * the last N records so the demo's "diagnostics" panel and bug reports can
 * dump them (`logger.dump()` / `logger.records`).
 *
 * @module core/log
 */
import { GifxError, ErrorSeverity } from './errors.js';

export const Level = { silent: 0, error: 1, warn: 2, info: 3, debug: 4, trace: 5 };

export class Logger {
  /**
   * @param {object} [opts]
   * @param {keyof typeof Level} [opts.level='warn']
   * @param {string} [opts.namespace='gifx']
   * @param {(rec:object)=>void} [opts.sink] custom transport (also used for ring buffer)
   * @param {number} [opts.bufferSize=500]
   */
  constructor(opts = {}) {
    this.namespace = opts.namespace || 'gifx';
    this.level = Level[opts.level] ?? Level.warn;
    this.sink = opts.sink || null;
    this.bufferSize = opts.bufferSize ?? 500;
    /** @type {{t:number, level:string, ns:string, msg:string, data:any}[]} */
    this.records = [];
    this._dropCount = 0;
    this.groupDepth = 0;
    this.table = opts.table !== false;
  }

  child(ns) {
    const l = new Logger({ namespace: `${this.namespace}:${ns}`, level: reverseLevel(this.level), sink: this.sink, bufferSize: this.bufferSize });
    l.parent = this;
    return l;
  }
  setLevel(l) {
    this.level = typeof l === 'number' ? l : Level[l] ?? this.level;
    return this;
  }
  isEnabled(l) {
    return this.level >= (typeof l === 'string' ? Level[l] : l);
  }

  _log(level, msg, data) {
    if (this.level < Level[level]) return;
    const rec = { t: Date.now(), level, ns: this.namespace, msg, data: data === undefined ? undefined : data };
    this.records.push(rec);
    while (this.records.length > this.bufferSize) {
      this.records.shift();
      this._dropCount++;
    }
    if (this.sink) {
      try {
        this.sink(rec);
        return;
      } catch {
        /* never let a transport break the pipeline */
      }
    }
    const prefix = `%c${level.toUpperCase()}%c ${this.namespace}`;
    const style = { error: 'color:#ff5560', warn: 'color:#ffb347', info: 'color:#59c1f5', debug: 'color:#9aa4b2', trace: 'color:#6c7684', silent: '' }[level];
    const consoleFn = level === 'error' ? console.error : level === 'warn' ? console.warn : level === 'trace' ? console.debug : console.log;
    if (typeof consoleFn === 'function') {
      try {
        consoleFn(prefix, `color:${style}`, 'font-style:italic;color:#8b95a3', ...(data === undefined ? [] : [data]));
      } catch {
        consoleFn(`${level.toUpperCase()} ${this.namespace}: ${msg}`, data);
      }
    }
  }

  error(msg, data) {
    this._log('error', msg, data);
  }
  warn(msg, data) {
    this._log('warn', msg, data);
  }
  info(msg, data) {
    this._log('info', msg, data);
  }
  debug(msg, data) {
    this._log('debug', msg, data);
  }
  trace(msg, data) {
    this._log('trace', msg, data);
  }

  /** Log a GifxError with its hint, in a single readable line. */
  failure(err, context = '') {
    if (GifxError.isGifxError(err)) {
      const sev = err.severity === ErrorSeverity.FATAL || err.severity === ErrorSeverity.ERROR ? 'error' : 'warn';
      this._log(sev, `${context ? context + ': ' : ''}${err.code}: ${err.message}${err.path ? ` (at ${err.path})` : ''}${err.hint ? `\n↳ ${err.hint}` : ''}`, err.data);
    } else {
      this._log('error', `${context ? context + ': ' : ''}${err?.message || String(err)}`, err);
    }
  }

  dump(filter) {
    const recs = filter ? this.records.filter((r) => r.level === filter || (filter.test && filter.test(r.msg))) : this.records;
    return recs.map((r) => `${new Date(r.t).toISOString()} ${r.level.toUpperCase().padEnd(5)} ${r.ns} — ${r.msg}${r.data !== undefined ? ` ${safeJson(r.data)}` : ''}`).join('\n');
  }
  table_() {
    return { dropped: this._dropCount, kept: this.records.length, level: reverseLevel(this.level) };
  }
  clear() {
    this.records.length = 0;
    this._dropCount = 0;
  }
}
function reverseLevel(n) {
  return Object.keys(Level).find((k) => Level[k] === n) || 'warn';
}
function safeJson(v) {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/** Process/module default logger. `GIFX.logger.setLevel('debug')`. */
export const logger = new Logger({ level: readInitialLevel() });

function readInitialLevel() {
  try {
    const g = globalThis;
    if (g.__GIFX_LOG_LEVEL__) return g.__GIFX_LOG_LEVEL__;
    const qs = g.location?.search;
    if (qs) {
      const m = /[?&]gifxlog=(\w+)/.exec(qs);
      if (m) return m[1];
    }
  } catch {
    /* ignore */
  }
  return 'warn';
}

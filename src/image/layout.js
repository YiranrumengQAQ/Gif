/**
 * GIFX Kernel — multi-raster layout ops.
 *
 * These turn N frames/images into one canvas: before/after comparisons, contact
 * sheets (filmstrips), tiled sprite sheets, vertical/horizontal stacks, mirror
 * kaleidoscopes and the "scrolling" effect people use for tall screenshots.
 *
 * Everything returns rasters of the *requested* size and never assumes equal
 * inputs — mismatched sizes are normalized by area-scaling into the cell.
 *
 * @module image/layout
 */
import { Raster } from '../core/buffers.js';
import { clamp255, parseColor } from '../core/color.js';
import { ErrorCode } from '../core/errors.js';
import { blit, drawBitmapText } from './overlay.js';
import { scale, pad, crop } from './ops.js';

/**
 * Stack rasters in a line.
 * @param {Raster[]} images
 * @param {object} [opts] `direction:'horizontal'|'vertical'`, `gap`, `color`,
 *   `align:'start'|'center'|'end'`, `equalize:true` (scale all to the same
 *   cross-axis size), `labels:string[]`
 */
export function stack(images, opts = {}) {
  const list = (images || []).filter(Boolean);
  if (!list.length) throw mkErr('stack() needs at least one image', ErrorCode.LAYOUT_EMPTY);
  const dir = opts.direction === 'vertical' || opts.dir === 'v' ? 'vertical' : 'horizontal';
  const gap = Math.max(0, opts.gap | 0);
  const cross = opts.equalize === false ? null : Math.min(...list.map((i) => (dir === 'horizontal' ? i.height : i.width)));
  const cells = list.map((img) => {
    if (!cross) return img;
    const w = dir === 'horizontal' ? Math.max(1, Math.round(img.width * (cross / img.height))) : cross;
    const h = dir === 'horizontal' ? cross : Math.max(1, Math.round(img.height * (cross / img.width)));
    if (w === img.width && h === img.height) return img;
    return scale(img, w, h, { kernel: opts.kernel || 'area' });
  });
  let width = 0;
  let height = 0;
  if (dir === 'horizontal') {
    width = cells.reduce((a, c) => a + c.width, 0) + gap * (cells.length - 1);
    height = Math.max(...cells.map((c) => c.height));
  } else {
    height = cells.reduce((a, c) => a + c.height, 0) + gap * (cells.length - 1);
    width = Math.max(...cells.map((c) => c.width));
  }
  const out = new Raster(width, height);
  if (opts.color) out.fill(...normalizeColor(opts.color));
  else if (opts.transparent) out.fill(0, 0, 0, 0);
  let cursor = 0;
  for (const c of cells) {
    const align = opts.align || 'center';
    let off = 0;
    if (align === 'center') off = Math.round(((dir === 'horizontal' ? height : width) - c[dir === 'horizontal' ? 'height' : 'width']) / 2);
    else if (align === 'end') off = (dir === 'horizontal' ? height : width) - c[dir === 'horizontal' ? 'height' : 'width'];
    if (dir === 'horizontal') blit(out, c, cursor, off, 1);
    else blit(out, c, off, cursor, 1);
    cursor += (dir === 'horizontal' ? c.width : c.height) + gap;
  }
  for (let i = 0; i < cells.length; i++) if (cells[i] !== list[i]) cells[i].release?.();
  return out;
}

const normalizeColor = (c) => {
  const p = parseColor(c);
  return [p[0], p[1], p[2], p[3]];
};

/**
 * Tile frames into a grid — the contact-sheet/filmstrip layout.
 * @param {Raster[]} frames
 * @param {object} opts `columns` (0 = sqrt), `rows`, `gap`, `color`,
 *   `order:'row'|'column'|'diagonal'`, `cell:{width,height}`, `fit:'contain'|'cover'|'fill'`
 * @returns {{raster:Raster, columns:number, rows:number, cellWidth:number, cellHeight:number, rects:Array}}
 */
export function tile(frames, opts = {}) {
  const list = (frames || []).filter(Boolean);
  if (!list.length) throw mkErr('tile() needs at least one image', ErrorCode.LAYOUT_EMPTY);
  const n = list.length;
  let columns = opts.columns | 0;
  let rows = opts.rows | 0;
  if (!columns && !rows) {
    columns = Math.max(1, Math.ceil(Math.sqrt(n)));
    rows = Math.ceil(n / columns);
  } else if (!columns) columns = Math.ceil(n / rows);
  else if (!rows) rows = Math.ceil(n / columns);
  while (columns * rows < n) rows++;
  const gap = Math.max(0, opts.gap | 0);
  const fit = opts.fit || 'contain';
  let cellW = opts.cell?.width | 0;
  let cellH = opts.cell?.height | 0;
  if (!cellW || !cellH) {
    cellW = cellW || Math.max(...list.map((i) => i.width));
    cellH = cellH || Math.max(...list.map((i) => i.height));
  }
  if (opts.maxCellWidth && cellW > opts.maxCellWidth) {
    const k = opts.maxCellWidth / cellW;
    cellW = Math.round(cellW * k);
    cellH = Math.max(1, Math.round(cellH * k));
  }
  const width = columns * cellW + (columns + 1) * gap;
  const height = rows * cellH + (rows + 1) * gap;
  const out = new Raster(width, height);
  if (opts.color) out.fill(...normalizeColor(opts.color));
  const rects = [];
  const order = opts.order || 'row';
  for (let i = 0; i < n; i++) {
    let c;
    let r;
    if (order === 'column') {
      c = Math.floor(i / rows);
      r = i % rows;
    } else if (order === 'diagonal') {
      const diag = Math.floor(Math.sqrt(i * 2));
      c = i - (diag * (diag + 1)) / 2;
      r = diag - c;
      if (r < 0 || c >= columns || r >= rows) {
        c = i % columns;
        r = Math.floor(i / columns);
      }
    } else {
      c = i % columns;
      r = Math.floor(i / columns);
    }
    if (c >= columns) c = columns - 1;
    if (r >= rows) r = rows - 1;
    const x = gap + c * (cellW + gap);
    const y = gap + r * (cellH + gap);
    rects.push({ index: i, x, y, width: cellW, height: cellH });
    const img = list[i];
    let prepared = img;
    if (fit === 'fill') {
      if (img.width !== cellW || img.height !== cellH) prepared = scale(img, cellW, cellH, { kernel: 'area' });
    } else if (fit === 'cover') {
      const s = Math.max(cellW / img.width, cellH / img.height);
      const w = Math.round(img.width * s);
      const h = Math.round(img.height * s);
      const tmp = scale(img, w, h, { kernel: 'area' });
      prepared = centerCrop(tmp, cellW, cellH);
      if (tmp !== prepared) tmp.release?.();
    } else {
      const s = Math.min(cellW / img.width, cellH / img.height);
      const w = Math.max(1, Math.round(img.width * s));
      const h = Math.max(1, Math.round(img.height * s));
      prepared = w === img.width && h === img.height ? img : scale(img, w, h, { kernel: 'area' });
    }
    const ox = x + Math.round((cellW - prepared.width) / 2);
    const oy = y + Math.round((cellH - prepared.height) / 2);
    blit(out, prepared, ox, oy, 1);
    if (opts.cellBorder) strokeCell(out, x, y, cellW, cellH, opts.cellBorder, opts.borderColor);
    if (opts.numbered) numberCell(out, i, x, y, cellW, opts);
    if (prepared !== img) prepared.release?.();
  }
  return { raster: out, columns, rows, cellWidth: cellW, cellHeight: cellH, rects, gap };
}

function centerCrop(img, w, h) {
  return crop(img, Math.max(0, Math.round((img.width - w) / 2)), Math.max(0, Math.round((img.height - h) / 2)), w, h);
}

function strokeCell(out, x, y, w, h, thickness, color) {
  const t = Math.max(1, thickness | 0);
  const c = color ? normalizeColor(color) : [255, 255, 255, 255];
  const d = out.data;
  const put = (px, py) => {
    if (px < 0 || py < 0 || px >= out.width || py >= out.height) return;
    const i = py * out.stride + px * 4;
    d[i] = c[0];
    d[i + 1] = c[1];
    d[i + 2] = c[2];
    d[i + 3] = 255;
  };
  for (let yy = y; yy < y + h; yy++) for (let tt = 0; tt < t; tt++) {
    put(x + tt, yy);
    put(x + w - 1 - tt, yy);
  }
  for (let xx = x; xx < x + w; xx++) for (let tt = 0; tt < t; tt++) {
    put(xx, y + tt);
    put(xx, y + h - 1 - tt);
  }
}

function numberCell(out, i, x, y, cellW, opts) {
  drawBitmapText(out, String(i + 1), { x: x + 3, y: y + 3, scale: Math.max(1, Math.round(cellW / 90)), color: opts.numberColor || '#ffffff', background: '#000000', backgroundAlpha: 0.5 });
}

/**
 * Side-by-side / over-under A-B comparison of two clips (before/after demos).
 * `mode`: 'sbs' | 'stack' | 'slider' (progressive reveal along x) | 'diff'
 * (amplified pixel difference — great for showing what a filter changed).
 */
export function compare(a, b, mode = 'sbs', opts = {}) {
  const w = Math.min(a.width, b.width);
  const h = Math.min(a.height, b.height);
  const A = a.width === w && a.height === h ? a : scale(a, w, h, { kernel: 'area' });
  const B = b.width === w && b.height === h ? b : scale(b, w, h, { kernel: 'area' });
  if (mode === 'diff') {
    const out = new Raster(w, h);
    const d = out.data;
    const gain = opts.gain || 4;
    for (let i = 0, n = w * h * 4; i < n; i += 4) {
      const dr = Math.abs(A.data[i] - B.data[i]) * gain;
      const dg = Math.abs(A.data[i + 1] - B.data[i + 1]) * gain;
      const db = Math.abs(A.data[i + 2] - B.data[i + 2]) * gain;
      d[i] = clamp255(dr > 8 ? 255 : 0);
      d[i + 1] = clamp255(dg > 8 ? 255 : 0);
      d[i + 2] = clamp255(db > 8 ? 255 : 0);
      d[i + 3] = 255;
    }
    return out;
  }
  if (mode === 'stack') return stack([A, B], { direction: 'vertical', gap: opts.gap || 0, equalize: true });
  if (mode === 'slider') {
    const out = new Raster(w, h);
    const split = Math.round(w * (opts.position ?? 0.5));
    out.data.set(A.data.subarray(0, A.height * A.stride), 0);
    for (let y = 0; y < h; y++) {
      const dy = y * out.stride;
      const by = y * B.stride;
      for (let x = split; x < w; x++) {
        const si = by + x * 4;
        const di = dy + x * 4;
        out.data[di] = B.data[si];
        out.data[di + 1] = B.data[si + 1];
        out.data[di + 2] = B.data[si + 2];
        out.data[di + 3] = B.data[si + 3];
      }
    }
    const t = Math.max(1, opts.divider || 2);
    for (let y = 0; y < h; y++) for (let k = 0; k < t; k++) {
      const i = (y * out.stride + Math.min(w - 1, split + k) * 4);
      out.data[i] = 255;
      out.data[i + 1] = 255;
      out.data[i + 2] = 0;
      out.data[i + 3] = 255;
    }
    return out;
  }
  return stack([A, B], { direction: 'horizontal', gap: opts.gap || 0, equalize: true });
}

/**
 * Mirror into 2 (flip) or 4 (kaleidoscope) halves.
 * `seam` rotates the mirror axis by N columns so a seamless tile doesn't show a
 * hard center line.
 */
export function mirror(img, { axes = 4, seam = 0 } = {}) {
  const flipH = (src) => {
    const out = new Raster(src.width, src.height);
    for (let y = 0; y < src.height; y++) {
      const srow = y * src.stride;
      for (let x = 0; x < src.width; x++) {
        const si = srow + (src.width - 1 - x) * 4;
        const di = y * out.stride + x * 4;
        out.data[di] = src.data[si];
        out.data[di + 1] = src.data[si + 1];
        out.data[di + 2] = src.data[si + 2];
        out.data[di + 3] = src.data[si + 3];
      }
    }
    return shiftSeam(out, seam);
  };
  const flipV = (src) => {
    const out = new Raster(src.width, src.height);
    for (let y = 0; y < src.height; y++) {
      const srow = (src.height - 1 - y) * src.stride;
      out.data.set(src.data.subarray(srow, srow + src.width * 4), y * out.stride);
    }
    return out;
  };
  const right = flipH(img);
  if (axes === 2) {
    const out = new Raster(img.width * 2, img.height);
    blit(out, img, 0, 0, 1);
    blit(out, right, img.width, 0, 1);
    right.release?.();
    return out;
  }
  const bottom = flipV(img);
  const bottomRight = flipH(bottom);
  const out = new Raster(img.width * 2, img.height * 2);
  blit(out, img, 0, 0, 1);
  blit(out, right, img.width, 0, 1);
  blit(out, bottom, 0, img.height, 1);
  blit(out, bottomRight, img.width, img.height, 1);
  right.release?.();
  bottom.release?.();
  bottomRight.release?.();
  return out;
}

function shiftSeam(r, seam) {
  if (!seam) return r;
  const out = new Raster(r.width, r.height);
  for (let y = 0; y < r.height; y++) {
    const row = y * r.stride;
    const orow = y * out.stride;
    for (let x = 0; x < r.width; x++) {
      const sx = (((x + seam) % r.width) + r.width) % r.width;
      out.data[orow + x * 4] = r.data[row + sx * 4];
      out.data[orow + x * 4 + 1] = r.data[row + sx * 4 + 1];
      out.data[orow + x * 4 + 2] = r.data[row + sx * 4 + 2];
      out.data[orow + x * 4 + 3] = r.data[row + sx * 4 + 3];
    }
  }
  r.release?.();
  return out;
}

/**
 * Infinite scroll of a tall image (screenshots → animated GIF). Returns the
 * frame schedule; the engine blits at each offset.
 */
export function scrollPlan(image, opts = {}) {
  const viewH = opts.viewHeight || opts.height || Math.min(image.height, 480);
  const step = Math.max(1, opts.step || Math.max(1, Math.round(viewH / (opts.frames || 24))));
  const frames = [];
  const mode = opts.mode || 'loop';
  if (mode === 'pingpong') {
    for (let y = 0; y + viewH <= image.height; y += step) frames.push({ y });
    for (let y = image.height - viewH - step; y > 0; y -= step) frames.push({ y });
  } else {
    const total = image.height + viewH;
    for (let y = 0; y < total; y += step) frames.push({ y: Math.min(y, image.height - viewH) });
  }
  return { viewWidth: image.width, viewHeight: viewH, frames, mode };
}

/** Render one scroll frame from a tall image. */
export function scrollFrame(image, plan, index) {
  const p = plan.frames[Math.max(0, Math.min(plan.frames.length - 1, index))];
  const out = new Raster(plan.viewWidth, plan.viewHeight);
  const y0 = p.y | 0;
  if (y0 + plan.viewHeight <= image.height) {
    for (let y = 0; y < plan.viewHeight; y++) {
      out.data.set(image.data.subarray((y0 + y) * image.stride, (y0 + y) * image.stride + plan.viewWidth * 4), y * out.stride);
    }
  } else {
    const avail = image.height - y0;
    for (let y = 0; y < avail; y++) out.data.set(image.data.subarray((y0 + y) * image.stride, (y0 + y) * image.stride + plan.viewWidth * 4), y * out.stride);
    for (let y = avail; y < plan.viewHeight; y++) out.data.fill(0, y * out.stride, y * out.stride + plan.viewWidth * 4);
  }
  return out;
}

/** Fit many images into one fixed-size canvas as a mosaic of equal cells. */
export function mosaic(images, width, height, opts = {}) {
  const t = tile(images, { columns: opts.columns, rows: opts.rows, gap: opts.gap, color: opts.color, fit: 'cover', cell: { width: Math.round(width / (opts.columns || 1)), height: Math.round(height / (opts.rows || 1)) } });
  if (t.raster.width === width && t.raster.height === height) return t;
  const s = pad(scale(t.raster, width, height, { kernel: 'area' }), width, height, { color: opts.color });
  t.raster.release?.();
  return { ...t, raster: s, cellWidth: width / t.columns, cellHeight: height / t.rows };
}

export function listLayouts() {
  return ['stack', 'tile', 'mosaic', 'compare', 'mirror', 'scrollPlan', 'scrollFrame'];
}

function mkErr(msg, code) {
  const e = new Error(msg);
  e.code = code;
  return e;
}

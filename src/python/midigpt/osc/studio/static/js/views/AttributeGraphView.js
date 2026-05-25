// AttributeGraphView.js — fills #attr-wrap, three line series with independent normalization

import { View } from '../core/mvc.js';
import { DRUM_ONLY_ATTRS } from '../models/SessionModel.js';

// Each series pairs a per-bar realized feature with the matching track-level
// token the server lifts off the prompt (sampledKey). Same units inside a
// row — bar-level min/max bracket the track-level token. Sampled levels are
// quantized 0..N-1; realized is in the source feature's raw units. The graph
// normalizes per series using the realized range, so the dashed "target" and
// the sampled step line share that visual scale.
const SERIES = [
  { key: 'note_density',      sampledKey: 'note_density',      label: 'Density',     color: '#4e9eff' },
  { key: 'min_polyphony',     sampledKey: 'min_polyphony',     label: 'Min Poly',    color: '#7bd389' },
  { key: 'max_polyphony',     sampledKey: 'max_polyphony',     label: 'Max Poly',    color: '#4ecca3' },
  { key: 'min_note_duration', sampledKey: 'min_note_duration', label: 'Min Dur',    color: '#ffb38a' },
  { key: 'max_note_duration', sampledKey: 'max_note_duration', label: 'Max Dur',    color: '#ffd166' },
];

const PAD = { top: 28, right: 16, bottom: 20, left: 40 };

export class AttributeGraphView extends View {
  constructor(canvas, sessionModel) {
    // canvas is the <canvas id="attrCanvas"> element
    super(canvas.parentElement);
    this._canvas  = canvas;
    this._ctx     = canvas.getContext('2d');
    this._session = sessionModel;
    this._w = 0;
    this._h = 0;
    this._frameScheduled = false;

    this._resizeObs = new ResizeObserver(() => this._onResize());
    this._resizeObs.observe(this.el);
    this._onResize();

    sessionModel.on('change', (patch) => {
      if ('realizedAttrs' in patch || 'attrs' in patch
          || 'sampledAttrs' in patch
          || 'agentIsDrum' in patch || 'attrCaps' in patch) {
        this._scheduleRender();
      }
    });
  }

  _onResize() {
    const r = this.el.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width));
    const h = Math.max(1, Math.floor(r.height));
    this._canvas.width  = w;
    this._canvas.height = h;
    this._canvas.style.width  = w + 'px';
    this._canvas.style.height = h + 'px';
    this._w = w;
    this._h = h;
    this._scheduleRender();
  }

  _scheduleRender() {
    if (this._frameScheduled) return;
    this._frameScheduled = true;
    requestAnimationFrame(() => {
      this._frameScheduled = false;
      this._render();
    });
  }

  _render() {
    const { _ctx: ctx, _w: W, _h: H } = this;
    if (!W || !H) return;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    const realized  = this._session.get('realizedAttrs') ?? [];
    const requested = this._session.get('attrs') ?? {};
    const sampled   = this._session.get('sampledAttrs') ?? [];

    this._drawGrid(ctx, W, H);

    const caps    = this._session.get('attrCaps');
    const isDrum  = !!this._session.get('agentIsDrum');
    const visible = SERIES.filter(s => {
      if (caps && caps[s.key] === false) return false;
      if (DRUM_ONLY_ATTRS.has(s.key) && !isDrum) return false;
      return true;
    });
    this._visibleSeries = visible;

    for (const series of visible) {
      const pts = realized
        .filter(r => r[series.key] != null)
        .map(r => ({ bar: r.bar, val: r[series.key] }));

      this._drawSeries(ctx, W, H, series, pts);

      const reqVal = requested[series.key];
      if (reqVal != null && reqVal >= 0) {
        this._drawRequestedLine(ctx, W, H, series, pts, reqVal);
      }

      if (series.sampledKey) {
        const steps = sampled
          .map(s => ({ start: s.startBar, end: s.endBar, val: s.attrs?.[series.sampledKey] }))
          .filter(s => s.val != null);
        if (steps.length) this._drawSampledSteps(ctx, W, H, series, pts, steps);
      }
    }

    this._drawLegend(ctx, W, H);
  }

  _drawGrid(ctx, W, H) {
    const usableH = H - PAD.top - PAD.bottom;
    ctx.strokeStyle = 'rgba(48,54,61,0.5)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([2, 4]);
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (i / 4) * usableH;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(W - PAD.right, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, H - PAD.bottom);
    ctx.lineTo(W - PAD.right, H - PAD.bottom);
    ctx.stroke();
  }

  _normalizePoints(pts) {
    if (pts.length === 0) return { vals: [], minV: 0, maxV: 1 };
    const vals = pts.map(p => p.val);
    let minV = Math.min(...vals);
    let maxV = Math.max(...vals);
    if (minV === maxV) { minV -= 1; maxV += 1; }
    return { vals, minV, maxV };
  }

  _toCanvas(bar, val, minV, maxV, W, H, totalBars) {
    const usableW = W - PAD.left - PAD.right;
    const usableH = H - PAD.top  - PAD.bottom;
    const x = PAD.left + (bar / Math.max(1, totalBars - 1)) * usableW;
    const y = PAD.top  + usableH - ((val - minV) / (maxV - minV)) * usableH;
    return { x, y };
  }

  _drawSeries(ctx, W, H, series, pts) {
    if (pts.length < 1) return;
    const { minV, maxV } = this._normalizePoints(pts);
    const totalBars = pts[pts.length - 1].bar + 1;

    ctx.strokeStyle = series.color;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const { x, y } = this._toCanvas(p.bar, p.val, minV, maxV, W, H, totalBars);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    ctx.fillStyle = series.color;
    for (const p of pts) {
      const { x, y } = this._toCanvas(p.bar, p.val, minV, maxV, W, H, totalBars);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawSampledSteps(ctx, W, H, series, pts, steps) {
    // Step function over the bar axis: each [start, end) window holds val.
    // Uses the realized series' y-normalization so visible position lines up
    // with the dashed "requested" line — match means the model honored it.
    if (pts.length < 1) return;
    const { minV, maxV } = this._normalizePoints(pts);
    const totalBars = pts[pts.length - 1].bar + 1;
    const usableW = W - PAD.left - PAD.right;
    const usableH = H - PAD.top  - PAD.bottom;
    const xOf = (bar) => PAD.left + (bar / Math.max(1, totalBars - 1)) * usableW;
    const yOf = (v)   => PAD.top + usableH - ((v - minV) / (maxV - minV)) * usableH;

    ctx.strokeStyle = series.color;
    ctx.lineWidth   = 2.5;
    ctx.globalAlpha = 0.45;
    ctx.setLineDash([]);
    for (const s of steps) {
      const y = yOf(Math.max(minV, Math.min(maxV, s.val)));
      const x0 = xOf(s.start);
      const x1 = xOf(Math.max(s.start, s.end - 1));
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  _drawRequestedLine(ctx, W, H, series, pts, reqVal) {
    if (pts.length < 1) return;
    const { minV, maxV } = this._normalizePoints(pts);
    // Clamp reqVal to existing range for display
    const clampedReq = Math.max(minV, Math.min(maxV, reqVal));
    const usableH = H - PAD.top - PAD.bottom;
    const y = PAD.top + usableH - ((clampedReq - minV) / (maxV - minV)) * usableH;

    ctx.strokeStyle = series.color;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  _drawLegend(ctx, W, H) {
    ctx.font         = '9px JetBrains Mono, monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign    = 'left';
    let x = PAD.left;
    const legendSeries = this._visibleSeries ?? SERIES;
    for (const s of legendSeries) {
      ctx.fillStyle = s.color;
      ctx.fillRect(x, 10, 12, 2);
      ctx.fillText(s.label, x + 14, 10);
      x += 70;
      if (x > W - PAD.right - 60) break;
    }
    // Style key
    ctx.fillStyle = '#8b949e';
    const keyX = W - PAD.right - 120;
    if (keyX > x) ctx.fillText('— realized  ·· target  ▬ sampled', keyX, 10);
    ctx.textBaseline = 'alphabetic';
  }
}

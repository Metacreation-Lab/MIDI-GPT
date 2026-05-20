import { View } from '../core/mvc.js';

const ATTRS = [
  { key: 'note_density',  label: 'Density',   color: '#4ecca3' },
  { key: 'max_polyphony', label: 'Polyphony',  color: '#c77dff' },
  { key: 'tension',       label: 'Tension',    color: '#ff6b6b' },
  { key: 'mean_pitch',    label: 'Pitch',      color: '#ffd166' },
];

const PAD   = { top: 28, right: 16, bottom: 28, left: 40 };
const MAX_BARS = 32;

export class AttributeGraphView extends View {
  constructor(el, sessionModel) {
    super(el);
    this._session = sessionModel;
    this._canvas  = document.createElement('canvas');
    this._canvas.style.cssText = 'display:block;width:100%;height:100%';
    this.el.appendChild(this._canvas);
    this._ctx = this._canvas.getContext('2d');

    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(this.el);
    this._resize();

    sessionModel.on('change', (patch) => {
      if ('realizedAttrs' in patch || 'attrs' in patch) this._scheduleRender();
    });
  }

  _resize() {
    const r = this.el.getBoundingClientRect();
    const dpr = devicePixelRatio;
    this._canvas.width  = r.width  * dpr;
    this._canvas.height = r.height * dpr;
    this._canvas.style.width  = r.width  + 'px';
    this._canvas.style.height = r.height + 'px';
    this._w = r.width;
    this._h = r.height;
    this._scheduleRender();
  }

  _scheduleRender() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => { this._frame = null; this.render(); });
  }

  render() {
    const { _ctx: ctx, _w: W, _h: H } = this;
    if (!W || !H) return;
    const dpr = devicePixelRatio;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    this._drawBackground(ctx, W, H);
    this._drawAxes(ctx, W, H);

    const realized = this._session.get('realizedAttrs') ?? [];
    const requested = this._session.get('attrs') ?? {};

    for (const attr of ATTRS) {
      this._drawRealized(ctx, W, H, attr, realized);
      if (requested[attr.key] >= 0) {
        this._drawRequested(ctx, W, H, attr, requested[attr.key]);
      }
    }

    this._drawLegend(ctx, W);
    ctx.restore();
  }

  _drawBackground(ctx, W, H) {
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);
    // Horizontal guide lines
    for (let v = 0; v <= 9; v += 3) {
      const y = this._vy(v, H);
      ctx.strokeStyle = '#1e2d40';
      ctx.lineWidth   = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle   = '#4a5568';
      ctx.font        = '9px JetBrains Mono, monospace';
      ctx.textBaseline = 'middle';
      ctx.fillText(v, 4, y);
    }
  }

  _drawAxes(ctx, W, H) {
    ctx.strokeStyle = '#2a3a6a';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, H - PAD.bottom);
    ctx.lineTo(W - PAD.right, H - PAD.bottom);
    ctx.stroke();
  }

  _vx(bar, W) {
    const usableW = W - PAD.left - PAD.right;
    return PAD.left + (bar / MAX_BARS) * usableW;
  }
  _vy(val, H) {
    const usableH = H - PAD.top - PAD.bottom;
    return PAD.top + usableH - (val / 9) * usableH;
  }

  _drawRealized(ctx, W, H, attr, realized) {
    const pts = realized
      .filter(r => r[attr.key] != null)
      .map(r => ({ x: this._vx(r.bar, W), y: this._vy(r[attr.key], H) }));
    if (pts.length < 1) return;

    ctx.strokeStyle = attr.color;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();

    // Dots
    ctx.fillStyle = attr.color;
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  _drawRequested(ctx, W, H, attr, value) {
    const y = this._vy(value, H);
    ctx.strokeStyle = attr.color;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  _drawLegend(ctx, W) {
    ctx.font      = '9px JetBrains Mono, monospace';
    ctx.textBaseline = 'middle';
    let x = PAD.left;
    for (const attr of ATTRS) {
      ctx.fillStyle   = attr.color;
      ctx.fillRect(x, 8, 12, 2);
      ctx.fillText(attr.label, x + 16, 9);
      x += 60;
    }
    // Legend key for line styles
    ctx.fillStyle = '#4a5568';
    ctx.fillText('— realized  ·· requested', W - 120, 9);
  }
}

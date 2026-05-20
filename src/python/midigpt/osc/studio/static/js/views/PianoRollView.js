import { View } from '../core/mvc.js';

const PITCH_MIN  = 21;   // A0
const PITCH_MAX  = 108;  // C8
const NUM_KEYS   = PITCH_MAX - PITCH_MIN + 1;
const KEY_HEIGHT = 6;    // px per semitone
const BAR_WIDTH  = 120;  // px per bar
const HEADER_H   = 24;   // ruler height

const BLACK_KEYS = new Set([1,3,6,8,10]); // semitone offsets within octave

function isBlack(pitch) { return BLACK_KEYS.has((pitch - PITCH_MIN) % 12); }

export class PianoRollView extends View {
  constructor(el, sessionModel) {
    super(el);
    this._session   = sessionModel;
    this._tracks    = [];
    this._scrollX   = 0;
    this._scrollY   = (NUM_KEYS / 2 - 12) * KEY_HEIGHT;
    this._animFrame = null;

    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'display:block;width:100%;height:100%;cursor:default';
    this.el.style.overflow = 'hidden';
    this.el.appendChild(this._canvas);
    this._ctx = this._canvas.getContext('2d');

    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(this.el);
    this._resize();

    this._canvas.addEventListener('wheel', e => this._onWheel(e), { passive: false });

    sessionModel.on('change', () => this._scheduleRender());
  }

  setTracks(tracks) {
    this._tracks = tracks;
    // Listen to new bars
    for (const t of tracks) {
      t.on('bar:generated', () => this._scheduleRender());
    }
    this._scheduleRender();
  }

  _resize() {
    const r = this.el.getBoundingClientRect();
    this._canvas.width  = r.width  * devicePixelRatio;
    this._canvas.height = r.height * devicePixelRatio;
    this._canvas.style.width  = r.width  + 'px';
    this._canvas.style.height = r.height + 'px';
    this._w = r.width;
    this._h = r.height;
    this._scheduleRender();
  }

  _scheduleRender() {
    if (this._animFrame) return;
    this._animFrame = requestAnimationFrame(() => { this._animFrame = null; this.render(); });
  }

  render() {
    const { _ctx: ctx, _canvas: cv, _w: W, _h: H } = this;
    if (!W || !H) return;
    const dpr = devicePixelRatio;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    this._drawGrid(ctx, W, H);
    this._drawRuler(ctx, W);
    this._drawNotes(ctx, W, H);
    this._drawPlayhead(ctx, H);

    ctx.restore();
  }

  _drawGrid(ctx, W, H) {
    const startPitch = Math.floor(PITCH_MAX - (this._scrollY + H) / KEY_HEIGHT);
    const endPitch   = Math.ceil(PITCH_MAX  -  this._scrollY      / KEY_HEIGHT);

    for (let p = Math.max(PITCH_MIN, startPitch); p <= Math.min(PITCH_MAX, endPitch); p++) {
      const y = HEADER_H + (PITCH_MAX - p) * KEY_HEIGHT - this._scrollY;
      ctx.fillStyle = isBlack(p) ? '#1a1a2e' : '#16213e';
      ctx.fillRect(0, y, W, KEY_HEIGHT);
      // C lines
      if (p % 12 === 0) {
        ctx.strokeStyle = '#2a3a6a';
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }
    }
    // Bar grid lines
    const barStart = Math.floor(this._scrollX / BAR_WIDTH);
    const barEnd   = barStart + Math.ceil(W / BAR_WIDTH) + 1;
    ctx.strokeStyle = '#2a3a6a';
    ctx.lineWidth = 1;
    for (let b = barStart; b <= barEnd; b++) {
      const x = b * BAR_WIDTH - this._scrollX;
      ctx.beginPath(); ctx.moveTo(x, HEADER_H); ctx.lineTo(x, H); ctx.stroke();
    }
  }

  _drawRuler(ctx, W) {
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, HEADER_H);
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textBaseline = 'middle';
    const barStart = Math.floor(this._scrollX / BAR_WIDTH);
    const barEnd   = barStart + Math.ceil(W / BAR_WIDTH) + 1;
    for (let b = barStart; b <= barEnd; b++) {
      const x = b * BAR_WIDTH - this._scrollX;
      ctx.fillText(b + 1, x + 3, HEADER_H / 2);
    }
    // Current bar highlight
    const cur = this._session.get('currentBar') ?? 0;
    const cx  = cur * BAR_WIDTH - this._scrollX;
    ctx.fillStyle = 'rgba(78,158,255,0.15)';
    ctx.fillRect(cx, 0, BAR_WIDTH, HEADER_H);
  }

  _drawNotes(ctx, W, H) {
    for (const track of this._tracks) {
      const color = track.color;
      const bars  = track.isAgent ? track.get('generatedBars') : this._getLoopBars(track);
      if (!bars) continue;

      for (const [barIdxStr, notes] of Object.entries(bars)) {
        const barIdx = +barIdxStr;
        const barX   = barIdx * BAR_WIDTH - this._scrollX;
        if (barX + BAR_WIDTH < 0 || barX > W) continue;

        for (const n of notes) {
          const x  = barX + n.onset    * BAR_WIDTH;
          const w  = Math.max(2, n.duration * BAR_WIDTH);
          const y  = HEADER_H + (PITCH_MAX - n.pitch) * KEY_HEIGHT - this._scrollY;
          const h  = KEY_HEIGHT;
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(x, y, w, h);
          ctx.globalAlpha = 1;
        }
      }
    }
  }

  _getLoopBars(track) {
    const loopBars = track.get('loopBars');
    if (!loopBars?.length) return {};
    // Repeat loop bars out to current bar
    const cur = this._session.get('currentBar') ?? 0;
    const out = {};
    for (let b = 0; b <= cur + 8; b++) {
      out[b] = loopBars[b % loopBars.length]?.notes ?? [];
    }
    return out;
  }

  _drawPlayhead(ctx, H) {
    const cur = this._session.get('currentBar') ?? 0;
    const x   = cur * BAR_WIDTH - this._scrollX;
    ctx.strokeStyle = '#4e9eff';
    ctx.lineWidth   = 2;
    ctx.beginPath(); ctx.moveTo(x, HEADER_H); ctx.lineTo(x, H); ctx.stroke();
  }

  _onWheel(e) {
    e.preventDefault();
    if (e.shiftKey) {
      this._scrollX = Math.max(0, this._scrollX + e.deltaY);
    } else {
      this._scrollY = Math.max(0, Math.min(
        NUM_KEYS * KEY_HEIGHT - this._h + HEADER_H,
        this._scrollY + e.deltaY,
      ));
    }
    this._scheduleRender();
  }
}

// TimingView — generation latency diagnostics panel

const MAX_HISTORY = 8;

export class TimingView {
  constructor(el, session) {
    this._el      = el;
    this._session = session;
    this._records = [];    // [{bar, latencyMs}]
    this._open    = null;  // {bar, startMs}
    this._status  = 'idle';
    this._elapsed = 0;
    this._rafId   = null;

    this._build();

    // Generation latency measurement:
    //   start  = the moment we sent bar:end to the server (a candidate gen trigger)
    //   commit = the server confirms it's actually generating (status="generating")
    //            — during the buffer phase, this never arrives and we drop the candidate
    //   close  = generated/close received → record latency
    this._pendingStart = null;  // {bar, startMs} — candidate, not yet committed

    session.on('bar:sent', (bar) => {
      // Replace any prior uncommitted candidate (it was a buffer-phase no-op).
      this._pendingStart = { bar, startMs: performance.now() };
    });

    session.on('change', (patch) => {
      if ('serverStatus' in patch) {
        if (patch.serverStatus === 'generating' && this._pendingStart && !this._open) {
          this._open = { ...this._pendingStart };
          this._pendingStart = null;
          this._status = 'generating';
          this._startTimer();
          this._render();
        }
      }
      if ('state' in patch && patch.state === 'STOPPED') {
        this._status = 'idle';
        this._open = null;
        this._pendingStart = null;
        this._stopTimer();
        this._render();
      }
    });

    session.on('generation:close', ({ barIndex }) => {
      if (this._open) {
        const latencyMs = performance.now() - this._open.startMs;
        this._records.push({ bar: barIndex, latencyMs });
        if (this._records.length > MAX_HISTORY) this._records.shift();
        this._open = null;
      }
      this._status = 'idle';
      this._stopTimer();
      this._render();
    });
  }

  _build() {
    this._el.className = 'timing-panel';
    this._el.innerHTML = `
      <div class="timing-header">
        <span class="timing-title">GEN TIMING</span>
        <span class="timing-status-badge" id="tStatus">idle</span>
      </div>
      <div class="timing-stats" id="tStats">
        <span class="timing-stat-label">last</span><span class="timing-stat-val" id="tLast">—</span>
        <span class="timing-stat-label">avg(5)</span><span class="timing-stat-val" id="tAvg">—</span>
        <span class="timing-stat-label">bar</span><span class="timing-stat-val" id="tBar">—</span>
        <span class="timing-stat-label">fit</span><span class="timing-stat-val" id="tFit">—</span>
      </div>
      <div class="timing-bars" id="tBars"></div>`;
  }

  _startTimer() {
    if (this._rafId) return;
    const tick = () => {
      if (this._open) {
        this._elapsed = performance.now() - this._open.startMs;
        this._render();
        this._rafId = requestAnimationFrame(tick);
      } else {
        this._rafId = null;
      }
    };
    this._rafId = requestAnimationFrame(tick);
  }

  _stopTimer() {
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
  }

  _render() {
    const params  = this._session.get('params') || {};
    const bpm     = this._session.get('bpm')    || 120;
    const tsNum   = this._session.get('tsNum')  || 4;
    const tsDen   = this._session.get('tsDen')  || 4;
    const barMs   = (60000 / bpm) * tsNum * (4 / tsDen);
    const lookahead = +(params.lookahead_bars ?? 2);

    const last5    = this._records.slice(-5);
    const avgMs    = last5.length
      ? last5.reduce((s, r) => s + r.latencyMs, 0) / last5.length
      : null;
    const lastMs   = this._records.length ? this._records.at(-1).latencyMs : null;
    const budget   = barMs * lookahead;
    const keeping  = avgMs != null ? avgMs < budget : null;

    // Status badge
    const statusEl = this._el.querySelector('#tStatus');
    if (this._status === 'generating') {
      statusEl.textContent = `GEN ${(this._elapsed / 1000).toFixed(1)}s`;
      statusEl.className   = 'timing-status-badge generating';
    } else {
      statusEl.textContent = 'idle';
      statusEl.className   = 'timing-status-badge';
    }

    this._el.querySelector('#tLast').textContent = lastMs != null ? `${(lastMs/1000).toFixed(2)}s` : '—';
    this._el.querySelector('#tAvg').textContent  = avgMs  != null ? `${(avgMs /1000).toFixed(2)}s` : '—';
    this._el.querySelector('#tBar').textContent  = `${(barMs/1000).toFixed(2)}s`;

    const fitEl = this._el.querySelector('#tFit');
    if (keeping === null) {
      fitEl.textContent = '—';
      fitEl.className   = 'timing-stat-val';
    } else if (keeping) {
      fitEl.textContent = '✓';
      fitEl.className   = 'timing-stat-val fit-ok';
    } else {
      fitEl.textContent = '⚠ behind';
      fitEl.className   = 'timing-stat-val fit-warn';
    }

    // Bar history sparklines
    const barsEl = this._el.querySelector('#tBars');
    barsEl.innerHTML = '';
    const maxLat = Math.max(...this._records.map(r => r.latencyMs), budget, 1);
    for (const rec of this._records.slice(-MAX_HISTORY)) {
      const pct  = Math.min(1, rec.latencyMs / maxLat);
      const over = rec.latencyMs > budget;
      const row  = document.createElement('div');
      row.className = 'timing-bar-row';
      row.innerHTML = `
        <span class="timing-bar-label">#${rec.bar}</span>
        <div class="timing-bar-track">
          <div class="timing-bar-fill ${over ? 'over-budget' : ''}"
               style="width:${(pct*100).toFixed(1)}%"></div>
          <div class="timing-bar-budget" style="left:${Math.min(100,(budget/maxLat)*100).toFixed(1)}%"></div>
        </div>
        <span class="timing-bar-val">${(rec.latencyMs/1000).toFixed(2)}s</span>`;
      barsEl.appendChild(row);
    }

    // Animate the in-progress bar if generating
    if (this._status === 'generating' && this._open) {
      const pct  = Math.min(1, this._elapsed / maxLat);
      const over = this._elapsed > budget;
      const row  = document.createElement('div');
      row.className = 'timing-bar-row active';
      row.innerHTML = `
        <span class="timing-bar-label">#${this._open.bar}</span>
        <div class="timing-bar-track">
          <div class="timing-bar-fill ${over ? 'over-budget' : ''} in-progress"
               style="width:${(pct*100).toFixed(1)}%"></div>
          <div class="timing-bar-budget" style="left:${Math.min(100,(budget/maxLat)*100).toFixed(1)}%"></div>
        </div>
        <span class="timing-bar-val">${(this._elapsed/1000).toFixed(2)}s…</span>`;
      barsEl.appendChild(row);
    }
  }
}

// ContextView.js — combined piano roll + prompt viz canvas

export const TRACK_ROW_H  = 80;  // px per track in piano roll section
export const PROMPT_ROW_H = 28;  // px per track in prompt viz section

// All MASKED_* variants share this base; they're disambiguated by the diagonal
// stripe overlay color (see MASK_STRIPE_COLORS + _drawDiagonalStripes).
const MASKED_BASE = '#8957e5';

const BAR_STATE_COLORS = {
  CONTEXT:       '#238636',
  GENERATED:     '#1f6feb',
  TO_GENERATE:   '#d29922',
  WASTED:        '#da3633',
  PENDING:       '#6e7681',
  MASKED_FUTURE: MASKED_BASE,
  MASKED_PAST:   MASKED_BASE,
  MASKED_FAILED: MASKED_BASE,
  OUTSIDE:       '#0d1117',

  // Simplified model-side view — only what the model's prompt actually sees.
  M_CONTEXT:     '#238636',
  M_MASKED:      MASKED_BASE,
  M_TO_GEN:      '#d29922',
};

// Detail viz states → the three states the model's tokenizer emits per bar.
// PENDING stays PENDING (no prompt built yet); OUTSIDE stays OUTSIDE.
export const MODEL_STATE = {
  CONTEXT:       'M_CONTEXT',
  GENERATED:     'M_CONTEXT',
  TO_GENERATE:   'M_TO_GEN',
  WASTED:        'M_TO_GEN',
  PENDING:       'PENDING',
  MASKED_FUTURE: 'M_MASKED',
  MASKED_PAST:   'M_MASKED',
  MASKED_FAILED: 'M_MASKED',
  OUTSIDE:       'OUTSIDE',
};

const MASK_STRIPE_COLORS = {
  MASKED_FUTURE: 'rgba(255,255,255,0.35)', // bright stripes = future
  MASKED_PAST:   'rgba(0,0,0,0.45)',       // dark stripes   = past/persistent
  MASKED_FAILED: 'rgba(218,54,51,0.75)',   // red stripes    = failed gen
  MASKED:        'rgba(255,255,255,0.25)', // legacy fallback
};

const BAR_STATE_LABELS = {
  CONTEXT:       'CTX',
  GENERATED:     'GEN',
  TO_GENERATE:   '→T←',
  WASTED:        'WST',
  PENDING:       '···',
  MASKED_FUTURE: 'M·F',
  MASKED_PAST:   'M·P',
  MASKED_FAILED: 'M·X',
  OUTSIDE:       '',

  M_CONTEXT:     'CTX',
  M_MASKED:      'MSK',
  M_TO_GEN:      '→T←',
};

const LOOKAHEAD_UNDERLINE = '#d29922';

export class ContextView {
  constructor(canvas, sessionModel) {
    this._canvas  = canvas;
    this._ctx     = canvas.getContext('2d');
    this._session = sessionModel;
    this._tracks  = [];
    this._simpleMode = false;

    // Pitch ranges per track (never shrink)
    this._pitchRanges = new Map(); // trackId → {min, max}

    // Live notes: trackId → Map(barIdx → notes[])
    this._liveNotes = new Map();

    // Generating bars: key `${trackId}:${barIdx}` → playheadAtOpen
    this._generatingBars = new Map();

    // Wasted bars: Set of barIdx
    this._wastedBars = new Set();

    // Set of agent barIdx that completed inference (even if 0 notes).
    this._completedAgentBars = new Set();

    // rAF handle for generating pulse animation
    this._rafHandle = null;
    this._frameScheduled = false;

    // ResizeObserver
    this._resizeObs = new ResizeObserver(() => this._onResize());
    this._resizeObs.observe(canvas.parentElement);
    this._onResize();

    sessionModel.on('change', (patch) => {
      if ('currentBar' in patch || 'params' in patch || 'serverStatus' in patch) {
        this._scheduleRedraw();
      }
      if (patch.serverStatus === 'generating') {
        this._startGeneratingLoop();
      }
      // A new bar means a fresh target — start animating until it transitions
      // out of the to-gen zone or completes (TARGET_DONE blink).
      if ('currentBar' in patch) {
        this._startGeneratingLoop();
      }
    });
  }

  // ── Public API ────────────────────────────────────────────────────────────

  setSimpleMode(on) {
    this._simpleMode = !!on;
    if (this._serverCanvas) {
      this._serverCanvas.hidden = !on;
      const lbl = document.getElementById('server-canvas-label');
      if (lbl) lbl.hidden = !on;
    }
    this._scheduleRedraw();
    if (on) this._drawServerView();
  }

  setServerCanvas(canvas) {
    this._serverCanvas = canvas;
    this._serverCtx    = canvas?.getContext('2d') ?? null;
    this._serverSnap   = null;
    this._mismatchSet  = new Set();   // `${trackId}:${barIdx}` keys
  }

  /**
   * Compare a server-emitted prompt state against what this view predicted for
   * the same target_bar / start_bar / end_bar. Returns a diff summary the
   * studio's status pill can display. Each disagreement is logged for the
   * console so we know which bar / track diverged.
   */
  compareWithServerState(snap) {
    const { target_bar, start_bar, end_bar, tracks: serverTracks } = snap;
    const params      = this._session.get('params') ?? {};
    const modelDim    = params.model_dim ?? 4;
    const lookahead   = params.lookahead_bars ?? 2;
    const numAnticip  = Math.max(1, params.num_anticipated_bars ?? 1);
    const bufferBars  = params.buffer_bars ?? 8;
    const adaptBuffer = params.adapt_buffer ?? true;
    const policy      = params.warmup_policy ?? 'a_empty';
    const firstGenPlayhead = adaptBuffer
      ? Math.max(0, bufferBars - lookahead)
      : bufferBars;
    const firstTargetBar = firstGenPlayhead + lookahead;
    // Reconstruct the playhead that produced this gen from target_bar.
    const playhead = Math.max(0, target_bar - lookahead);
    const genActive = playhead >= firstGenPlayhead;
    const numCompletedTicks = genActive
      ? Math.floor((playhead - firstGenPlayhead + numAnticip - 1) / numAnticip)
      : 0;
    const toGenFirstBar = firstTargetBar + numCompletedTicks * numAnticip;
    const toGenLastBar  = toGenFirstBar + numAnticip - 1;
    const inWarmup      = playhead < (modelDim - numAnticip - lookahead);

    let matches = 0, mismatches = 0;
    const diffs = [];
    for (const tinfo of serverTracks) {
      const track = this._tracks.find(t => t.piece_idx === tinfo.id)
                 ?? this._tracks[tinfo.id]
                 ?? { isAgent: tinfo.is_agent, id: tinfo.id, get: () => ({}) };
      for (let i = 0; i < tinfo.states.length; i++) {
        const bar = start_bar + i;
        const detail = this._getBarState(
          track, bar, playhead, firstGenPlayhead, firstTargetBar,
          toGenFirstBar, toGenLastBar, inWarmup, policy,
        );
        const predicted = MODEL_STATE[detail] ?? detail;
        const pCode = predicted === 'M_CONTEXT' ? 'C'
                    : predicted === 'M_TO_GEN'  ? 'T'
                    : predicted === 'M_MASKED'  ? 'M'
                    : '?';
        const actual = tinfo.states[i];
        if (pCode === actual) matches++;
        else {
          mismatches++;
          diffs.push({ track: tinfo.id, bar, predicted: pCode, actual, detail });
        }
      }
    }
    const status = mismatches === 0
      ? { ok: true,  text: `✓ viz matches server (${matches} bars)` }
      : { ok: false, text: `✗ ${mismatches}/${matches + mismatches} mismatch`, diffs };
    if (!status.ok) console.warn('[viz-compare]', status.text, diffs);
    this._serverSnap = snap;
    this._mismatchSet = new Set((diffs ?? []).map(d => `${d.track}:${d.bar}`));
    this._drawServerView();
    return status;
  }

  _drawServerView() {
    const canvas = this._serverCanvas;
    const ctx    = this._serverCtx;
    if (!canvas || !ctx || !this._simpleMode || !this._serverSnap) return;
    const snap = this._serverSnap;
    const W = this._canvas.width;
    const n = this._tracks.length;
    const params    = this._session.get('params') ?? {};
    const modelDim  = params.model_dim ?? 4;
    const playhead  = this._session.get('currentBar') ?? 0;
    const numAnticip = Math.max(1, params.num_anticipated_bars ?? 1);
    const lookahead  = params.lookahead_bars ?? 2;
    const steadyPlayheadCol = modelDim - numAnticip - lookahead;
    const playheadCol = Math.min(playhead, steadyPlayheadCol);
    const colToBar  = (col) => playhead + (col - playheadCol);
    const colW = W / modelDim;

    canvas.width  = W;
    canvas.height = n * PROMPT_ROW_H;
    canvas.style.width  = W + 'px';
    canvas.style.height = (n * PROMPT_ROW_H) + 'px';

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const codeToState = { C: 'M_CONTEXT', T: 'M_TO_GEN', M: 'M_MASKED' };

    for (let ti = 0; ti < n; ti++) {
      const track  = this._tracks[ti];
      const tinfo  = snap.tracks.find(t => t.id === (track.piece_idx ?? track.id) ?? ti)
                  ?? snap.tracks[ti];
      const rowY   = ti * PROMPT_ROW_H;
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, rowY, W, PROMPT_ROW_H);
      for (let col = 0; col < modelDim; col++) {
        const bar = colToBar(col);
        const cellX = Math.round(col * colW);
        const cellW = Math.round((col + 1) * colW) - cellX;
        if (bar < 0) {
          ctx.fillStyle = BAR_STATE_COLORS.OUTSIDE;
          ctx.fillRect(cellX, rowY, cellW, PROMPT_ROW_H);
          continue;
        }
        let code = '?';
        if (tinfo && bar >= snap.start_bar && bar < snap.end_bar) {
          code = tinfo.states[bar - snap.start_bar];
        }
        const state = codeToState[code] ?? 'OUTSIDE';
        ctx.fillStyle = BAR_STATE_COLORS[state] ?? '#30363d';
        ctx.fillRect(cellX, rowY, cellW, PROMPT_ROW_H);

        const label = BAR_STATE_LABELS[state] ?? code;
        if (label) {
          ctx.fillStyle    = 'rgba(230,237,243,0.85)';
          ctx.font         = '9px JetBrains Mono, monospace';
          ctx.textBaseline = 'middle';
          ctx.textAlign    = 'center';
          ctx.fillText(label, cellX + cellW / 2, rowY + PROMPT_ROW_H / 2);
        }
        const tid = tinfo?.id ?? ti;
        if (this._mismatchSet.has(`${tid}:${bar}`)) {
          ctx.strokeStyle = '#f85149';
          ctx.lineWidth   = 2;
          ctx.strokeRect(cellX + 1, rowY + 1, cellW - 2, PROMPT_ROW_H - 2);
        }
      }
    }
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  setTracks(tracks) {
    this._tracks = tracks;
    // Scan loop notes for initial pitch ranges
    for (const track of tracks) {
      const id = track.id;
      if (!this._pitchRanges.has(id)) {
        this._pitchRanges.set(id, { min: 60, max: 72 });
      }
      const loopBars = track.get('loopBars') ?? [];
      for (const bar of loopBars) {
        for (const note of (bar.notes ?? [])) {
          this._expandPitchRange(id, note.pitch);
        }
      }
    }
    this._onResize();
    this._scheduleRedraw();
  }

  onGeneratedOpen(trackId, barIndex) {
    const playhead = this._session.get('currentBar');
    this._generatingBars.set(`${trackId}:${barIndex}`, playhead);
    this._startGeneratingLoop();
  }

  onGeneratedClose(trackId, barIndex, notes) {
    this._generatingBars.delete(`${trackId}:${barIndex}`);
    this._completedAgentBars.add(barIndex);
    const playhead = this._session.get('currentBar');
    if (barIndex < playhead) {
      this._wastedBars.add(barIndex);
    }
    // Expand pitch ranges for generated notes
    for (const note of (notes ?? [])) {
      this._expandPitchRange(trackId, note.pitch);
    }
    if (this._generatingBars.size === 0) {
      // Loop will self-terminate
    }
    this._scheduleRedraw();
  }

  onLiveNote(track, pitch, velocity) {
    const trackId  = track.id;
    const playhead = this._session.get('currentBar');
    if (!this._liveNotes.has(trackId)) {
      this._liveNotes.set(trackId, new Map());
    }
    const barMap = this._liveNotes.get(trackId);
    if (!barMap.has(playhead)) barMap.set(playhead, []);
    barMap.get(playhead).push({ pitch, velocity, onset: 0, duration: 0.5 });
    this._expandPitchRange(trackId, pitch);
    this._scheduleRedraw();
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _expandPitchRange(trackId, pitch) {
    const r = this._pitchRanges.get(trackId) ?? { min: 60, max: 72 };
    const newMin = Math.min(r.min, pitch - 3);
    const newMax = Math.max(r.max, pitch + 3);
    if (newMin !== r.min || newMax !== r.max) {
      this._pitchRanges.set(trackId, { min: newMin, max: newMax });
    }
  }

  _onResize() {
    const n = this._tracks.length;
    const canvasH = n * (TRACK_ROW_H + PROMPT_ROW_H);
    const wrap    = this._canvas.parentElement;
    const w       = wrap.clientWidth || 800;

    this._canvas.width  = w;
    this._canvas.height = canvasH;
    this._canvas.style.width  = w + 'px';
    this._canvas.style.height = canvasH + 'px';

    this._scheduleRedraw();
  }

  _scheduleRedraw() {
    if (this._frameScheduled) return;
    this._frameScheduled = true;
    requestAnimationFrame(() => {
      this._frameScheduled = false;
      this._draw();
    });
  }

  _startGeneratingLoop() {
    if (this._rafHandle !== null) return; // already running
    const loop = () => {
      this._needsAnim = false;
      this._draw();
      // Continue while: per-bar gen open, server inflight, or any TARGET_DONE
      // bar is blinking in the to-gen zone (flagged during _draw).
      const active = this._needsAnim
        || this._generatingBars.size > 0
        || this._session.get('serverStatus') === 'generating';
      if (active) {
        this._rafHandle = requestAnimationFrame(loop);
      } else {
        this._rafHandle = null;
      }
    };
    this._rafHandle = requestAnimationFrame(loop);
  }

  _draw() {
    const { _canvas: canvas, _ctx: ctx, _tracks: tracks } = this;
    const W = canvas.width;
    const H = canvas.height;
    if (!W || !H || tracks.length === 0) {
      ctx.clearRect(0, 0, W, H);
      return;
    }

    const params    = this._session.get('params') ?? {};
    const modelDim  = params.model_dim  ?? 4;
    const playhead  = this._session.get('currentBar') ?? 0;
    const bufferBars   = params.buffer_bars   ?? 8;
    const lookahead    = params.lookahead_bars ?? 2;
    const numAnticip   = Math.max(1, params.num_anticipated_bars ?? 1);
    const adaptBuffer  = params.adapt_buffer  ?? true;

    // Grid geometry:
    //   warmup:      window pinned at [0, modelDim). playheadCol == playhead.
    //                target sits at col `playhead+lookahead` INSIDE the window;
    //                cols after target+numAnticip are WASTED (AR-sampled then
    //                discarded — model has to fill the window to model_dim).
    //   established: window slides each tick so playheadCol == steady and the
    //                to-gen tail sits at the rightmost numAnticip cols. No waste.
    const steadyPlayheadCol = modelDim - numAnticip - lookahead;
    const inWarmup    = playhead < steadyPlayheadCol;
    const playheadCol = Math.min(playhead, steadyPlayheadCol);
    const colToBar    = (col) => playhead + (col - playheadCol);
    const barToCol    = (bar) => playheadCol + (bar - playhead);

    // firstGenPlayhead = bar index where the FIRST generation can fire
    const firstGenPlayhead = adaptBuffer
      ? Math.max(0, bufferBars - lookahead)
      : bufferBars;
    const firstTargetBar = firstGenPlayhead + lookahead;

    // Ticks fire every `numAnticip` bars. A tick at the boundary is in-flight
    // (TO_GENERATE), prior ticks are completed (GENERATED). Between boundaries
    // the next-upcoming tick is shown as TO_GENERATE.
    const genActive       = playhead >= firstGenPlayhead;
    const numCompletedTicks = genActive
      ? Math.floor((playhead - firstGenPlayhead + numAnticip - 1) / numAnticip)
      : 0;
    const toGenFirstBar   = firstTargetBar + numCompletedTicks * numAnticip;
    const toGenLastBar    = toGenFirstBar + numAnticip - 1;
    const toGenFirstCol   = barToCol(toGenFirstBar);
    const toGenLastCol    = barToCol(toGenLastBar);
    const lookaheadFirstCol = playheadCol + 1;
    const lookaheadLastCol  = toGenFirstCol - 1;
    const warmupPolicy = params.warmup_policy ?? 'a_empty';
    const serverStatus = this._session.get('serverStatus') ?? 'idle';

    // One-shot per render: log the geometry so we can verify warmup detection
    // and policy propagation when the visualization looks wrong.
    if (this._lastGeomKey !== `${playhead}/${modelDim}/${warmupPolicy}/${inWarmup}`) {
      this._lastGeomKey = `${playhead}/${modelDim}/${warmupPolicy}/${inWarmup}`;
      console.log('[ctxview]',
        'playhead=', playhead,
        'modelDim=', modelDim,
        'lookahead=', lookahead,
        'numAnticip=', numAnticip,
        'steadyCol=', steadyPlayheadCol,
        'inWarmup=', inWarmup,
        'warmupPolicy=', JSON.stringify(warmupPolicy));
    }
    const isInflight   = serverStatus === 'generating';

    const colW = W / modelDim;
    const n    = tracks.length;

    ctx.clearRect(0, 0, W, H);

    // ── Piano roll section ────────────────────────────────────────────────
    for (let ti = 0; ti < n; ti++) {
      const track  = tracks[ti];
      const rowY   = ti * TRACK_ROW_H;
      const bg     = ti % 2 === 0 ? '#161b22' : '#0d1117';

      ctx.fillStyle = bg;
      ctx.fillRect(0, rowY, W, TRACK_ROW_H);

      const pitchRange = this._pitchRanges.get(track.id) ?? { min: 60, max: 72 };
      const pitchSpan  = Math.max(1, pitchRange.max - pitchRange.min);
      const noteH      = Math.max(2, TRACK_ROW_H / pitchSpan);

      for (let col = 0; col < modelDim; col++) {
        const barIdx = colToBar(col);
        const cellX  = Math.round(col * colW);
        const cellW  = Math.round((col + 1) * colW) - cellX;

        // Currently-playing bar highlight
        if (col === playheadCol) {
          ctx.fillStyle = 'rgba(78,158,255,0.06)';
          ctx.fillRect(cellX, rowY, cellW, TRACK_ROW_H);
        }

        if (barIdx < 0) continue;

        const loopBars  = track.get('loopBars') ?? [];
        const genBars   = track.get('generatedBars') ?? {};

        let notes = null;

        if (track.isAgent) {
          notes = genBars[barIdx] ?? null;
        } else if (track.mode === 'loop') {
          const loopBar = loopBars.length > 0 ? loopBars[barIdx % loopBars.length] : null;
          notes = loopBar?.notes ?? null;
        } else if (track.mode === 'live') {
          const barMap = this._liveNotes.get(track.id);
          notes = barMap?.get(barIdx) ?? null;
        }

        if (!notes || notes.length === 0) continue;

        // Alpha: dim for loop past bars, bright for current (playhead)
        const isLoop   = !track.isAgent && track.mode === 'loop';
        const alpha    = isLoop && col !== playheadCol ? 0.3 : 1.0;
        const color    = track.color ?? '#4e9eff';

        ctx.globalAlpha = alpha;
        ctx.fillStyle   = color;

        for (const note of notes) {
          const noteY = rowY + TRACK_ROW_H
            - ((note.pitch - pitchRange.min) / pitchSpan) * TRACK_ROW_H
            - noteH;
          const noteX = cellX + (note.onset ?? 0) * cellW;
          const noteW = Math.max(3, (note.duration ?? 0.5) * cellW * 0.9);
          ctx.fillRect(Math.round(noteX), Math.round(noteY), Math.round(noteW), noteH);
        }

        ctx.globalAlpha = 1.0;
      }

      // Bar number labels
      for (let col = 0; col < modelDim; col++) {
        const barIdx = colToBar(col);
        if (barIdx < 0) continue;
        const cellX = Math.round(col * colW);
        ctx.fillStyle = '#8b949e';
        ctx.font      = '10px JetBrains Mono, monospace';
        ctx.textBaseline = 'top';
        ctx.fillText(String(barIdx), cellX + 3, rowY + 2);
      }
    }

    // Piano roll column dividers
    ctx.strokeStyle = 'rgba(48,54,61,0.6)';
    ctx.lineWidth   = 1;
    for (let col = 1; col < modelDim; col++) {
      const x = Math.round(col * colW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, n * TRACK_ROW_H);
      ctx.stroke();
    }

    // ── Section divider ───────────────────────────────────────────────────
    const dividerY = n * TRACK_ROW_H;
    ctx.fillStyle = '#30363d';
    ctx.fillRect(0, dividerY, W, 2);

    // ── Prompt viz section ────────────────────────────────────────────────
    const promptBase = dividerY + 2;
    const now = performance.now();

    for (let ti = 0; ti < n; ti++) {
      const track  = tracks[ti];
      const rowY   = promptBase + ti * PROMPT_ROW_H;

      // Row background
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, rowY, W, PROMPT_ROW_H);

      for (let col = 0; col < modelDim; col++) {
        const barIdx = colToBar(col);
        const cellX  = Math.round(col * colW);
        const cellW  = Math.round((col + 1) * colW) - cellX;

        const isLookahead  = col >= lookaheadFirstCol && col <= lookaheadLastCol;
        const detail = this._getBarState(
          track, barIdx, playhead, firstGenPlayhead, firstTargetBar,
          toGenFirstBar, toGenLastBar, inWarmup, warmupPolicy,
        );
        const state = this._simpleMode ? (MODEL_STATE[detail] ?? detail) : detail;
        let   alpha  = 1.0;

        // Pulse the in-flight TO_GENERATE block while the server is generating
        // (or this specific bar is between /open and /close). Same state name
        // the review uses — only the animation is studio-only.
        const liveGen = isInflight
          || this._generatingBars.has(`${track.id}:${barIdx}`);
        if (state === 'TO_GENERATE' && liveGen) {
          alpha = 0.55 + 0.45 * Math.sin(now / 250);
          this._needsAnim = true;
        }
        if (state === 'OUTSIDE') {
          ctx.fillStyle = BAR_STATE_COLORS.OUTSIDE;
          ctx.fillRect(cellX, rowY, cellW, PROMPT_ROW_H);
          continue;
        }

        ctx.globalAlpha = alpha;
        ctx.fillStyle   = BAR_STATE_COLORS[state] ?? '#30363d';
        ctx.fillRect(cellX, rowY, cellW, PROMPT_ROW_H);
        ctx.globalAlpha = 1.0;

        // Mask variants share the same fill — the stripe overlay distinguishes
        // future / past / failed.
        const stripeColor = MASK_STRIPE_COLORS[state];
        if (stripeColor) {
          this._drawDiagonalStripes(
            cellX, rowY, cellW, PROMPT_ROW_H, stripeColor,
          );
        }

        // Lookahead underline (drawn instead of using a distinct fill color).
        // The bar's color reflects its actual state; the underline just marks
        // it as inside the lookahead window.
        if (isLookahead) {
          ctx.fillStyle = LOOKAHEAD_UNDERLINE;
          ctx.fillRect(cellX + 1, rowY + PROMPT_ROW_H - 3, cellW - 2, 2);
        }

        // Label
        const label = BAR_STATE_LABELS[state] ?? '';
        if (label) {
          ctx.fillStyle    = 'rgba(230,237,243,0.85)';
          ctx.font         = '9px JetBrains Mono, monospace';
          ctx.textBaseline = 'middle';
          ctx.textAlign    = 'center';
          ctx.fillText(label, cellX + cellW / 2, rowY + PROMPT_ROW_H / 2);
        }
      }
    }

    // Prompt viz column dividers
    ctx.strokeStyle = 'rgba(48,54,61,0.6)';
    ctx.lineWidth   = 1;
    for (let col = 1; col < modelDim; col++) {
      const x = Math.round(col * colW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, promptBase);
      ctx.lineTo(x, promptBase + n * PROMPT_ROW_H);
      ctx.stroke();
    }

    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  _drawDiagonalStripes(x, y, w, h, color) {
    const ctx = this._ctx;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    const step = 6;
    // Draw lines going from bottom-left to top-right across a region wide
    // enough to fully cover the cell at this angle.
    for (let off = -h; off < w + h; off += step) {
      ctx.beginPath();
      ctx.moveTo(x + off,     y + h);
      ctx.lineTo(x + off + h, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  _getBarState(track, barIdx, playhead, firstGenPlayhead, firstTargetBar,
               toGenFirstBar, toGenLastBar, inWarmup, warmupPolicy) {
    // Mirrors `bar_state` in scripts/gen_context_scenario.py exactly.
    // Live runtime signals (generating/completed/inflight) drive *animation*
    // in _draw, not the state name itself — so the studio and review viz
    // always agree on what the bar IS, just the studio may pulse it.
    if (barIdx < 0) return 'OUTSIDE';

    const genActive = playhead >= firstGenPlayhead;

    if (!track.isAgent) {
      if (barIdx < playhead) return 'CONTEXT';
      return genActive ? 'MASKED_FUTURE' : 'PENDING';
    }

    const trackId   = track.id;
    const completed = this._completedAgentBars.has(barIdx);
    const genBars   = track.get('generatedBars') ?? {};
    const failed    = completed && (genBars[barIdx]?.length ?? 0) === 0;

    if (failed) return 'MASKED_FAILED';

    if (barIdx < firstTargetBar) {
      if (!genActive) return 'PENDING';
      if (warmupPolicy === 'a_empty')  return 'CONTEXT';
      if (warmupPolicy === 'a_masked') return 'MASKED_PAST';
      if (playhead === firstGenPlayhead) return 'TO_GENERATE';
      return 'CONTEXT';
    }

    if (barIdx < toGenFirstBar) return 'GENERATED';
    if (barIdx <= toGenLastBar) return genActive ? 'TO_GENERATE' : 'PENDING';
    if (inWarmup) return genActive ? 'WASTED' : 'PENDING';
    return 'MASKED_FUTURE';
  }
}

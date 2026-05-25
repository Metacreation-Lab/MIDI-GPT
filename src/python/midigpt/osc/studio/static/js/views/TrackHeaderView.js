// TrackHeaderView.js — left column track headers (piano rows + prompt rows)

import { View } from '../core/mvc.js';
import { TRACK_ROW_H, PROMPT_ROW_H } from './ContextView.js';
import { MODE_LOOP, MODE_LIVE, MODE_AGENT } from '../models/TrackModel.js';

const GM_PROGRAMS = [
  [0,  'Acoustic Grand Piano'],
  [1,  'Bright Acoustic Piano'],
  [4,  'Electric Piano 1'],
  [24, 'Acoustic Guitar (nylon)'],
  [25, 'Acoustic Guitar (steel)'],
  [26, 'Electric Guitar (jazz)'],
  [32, 'Acoustic Bass'],
  [33, 'Electric Bass (finger)'],
  [40, 'Violin'],
  [41, 'Viola'],
  [56, 'Trumpet'],
  [57, 'Trombone'],
  [60, 'French Horn'],
  [68, 'Oboe'],
  [71, 'Clarinet'],
  [73, 'Flute'],
  [80, 'Square Wave'],
  [88, 'Pad 1 (new age)'],
  [105,'Banjo'],
  [116,'Woodblock'],
];

// Mask variants share the same base fill and are distinguished by diagonal
// stripe overlays (matches ContextView). `stripe` is an rgba overlay color.
const MASK_BASE = '#8957e5';
const LEGEND = [
  { state:'CONTEXT',       color:'#238636', label:'context'                       },
  { state:'MASKED_FUTURE', color:MASK_BASE, stripe:'rgba(255,255,255,0.55)',
                                           label:'masked · future (>= playhead)' },
  { state:'MASKED_PAST',   color:MASK_BASE, stripe:'rgba(0,0,0,0.55)',
                                           label:'masked · pre-history (policy)' },
  { state:'MASKED_FAILED', color:MASK_BASE, stripe:'rgba(218,54,51,0.85)',
                                           label:'masked · failed gen'           },
  { state:'GENERATING',    color:'#d29922', label:'generating (inflight)'         },
  { state:'TARGET_DONE',   color:'#1f6feb', label:'current target (done)'         },
  { state:'GENERATED',     color:'#1f6feb', label:'generated (past)'              },
  { state:'WASTED',        color:'#da3633', label:'wasted'                        },
  { state:'PENDING',       color:'#6e7681', label:'pending'                       },
  { state:'LOOKAHEAD',     color:'#d29922', label:'lookahead (underline)'         },
];

export class TrackHeaderView extends View {
  constructor(el, appCtrl, midiCtrl, audioCtrl, sessionModel) {
    super(el);
    this._app     = appCtrl;
    this._midi    = midiCtrl;
    this._audio   = audioCtrl;
    this._session = sessionModel;
    this._tracks  = [];
    this._build();
    sessionModel.on('change', (p) => {
      if ('state' in p) this._updateLockState(p.state);
    });
    this._updateLockState(sessionModel.get('state'));
  }

  _updateLockState(state) {
    const locked = state === 'RUNNING';
    for (const btn of this.el.querySelectorAll('#btnAddCond, #btnAddAgent, .btn-remove')) {
      btn.disabled = locked;
      btn.classList.toggle('locked', locked);
    }
  }

  _build() {
    this.el.innerHTML = '';

    // Add-track buttons at top
    const addBar = document.createElement('div');
    addBar.style.cssText = 'display:flex;gap:4px;padding:6px 8px;border-bottom:1px solid var(--border);flex-shrink:0;';
    addBar.innerHTML = `
      <button id="btnAddCond"  class="btn btn-xs btn-secondary">+ Cond</button>
      <button id="btnAddAgent" class="btn btn-xs btn-primary">+ Agent</button>`;
    addBar.querySelector('#btnAddCond').onclick  = () => this._app.addConditioningTrack();
    addBar.querySelector('#btnAddAgent').onclick = () => this._app.addAgentTrack();
    this.el.appendChild(addBar);

    // Piano section
    this._pianoSection = document.createElement('div');
    this._pianoSection.id = 'pianoSection';
    this.el.appendChild(this._pianoSection);

    // Prompt label divider
    this._promptDivider = document.createElement('div');
    this._promptDivider.className = 'track-section-label';
    this._promptDivider.textContent = 'PROMPT';
    this.el.appendChild(this._promptDivider);

    // Prompt section
    this._promptSection = document.createElement('div');
    this._promptSection.id = 'promptSection';
    this.el.appendChild(this._promptSection);

    // Legend
    this._legendEl = document.createElement('div');
    this._legendEl.className = 'prompt-legend';
    this._buildLegend();
    this.el.appendChild(this._legendEl);
  }

  _buildLegend() {
    this._legendEl.innerHTML = '';
    for (const item of LEGEND) {
      const div = document.createElement('div');
      div.className = 'legend-item';
      // Diagonal stripe overlay for mask variants; built with a linear-gradient
      // so it renders identically to the canvas stripes in ContextView.
      const bg = item.stripe
        ? `repeating-linear-gradient(45deg, ${item.stripe} 0 2px, transparent 2px 6px), ${item.color}`
        : item.color;
      div.innerHTML = `
        <span class="legend-swatch" style="background:${bg}"></span>
        <span>${item.label}</span>`;
      this._legendEl.appendChild(div);
    }
  }

  setTracks(tracks) {
    this._tracks = tracks;
    this._renderPianoRows(tracks);
    this._renderPromptRows(tracks);
    this._updateLockState(this._session.get('state'));
  }

  _renderPianoRows(tracks) {
    this._pianoSection.innerHTML = '';
    for (const track of tracks) {
      this._pianoSection.appendChild(this._buildPianoRow(track));
    }
  }

  _renderPromptRows(tracks) {
    this._promptSection.innerHTML = '';
    for (const track of tracks) {
      this._promptSection.appendChild(this._buildPromptRow(track));
    }
  }

  _buildPianoRow(track) {
    const row = document.createElement('div');
    row.className = 'track-piano-row';
    row.style.height = TRACK_ROW_H + 'px';
    row.dataset.trackId = track.id;

    const modeLabel = track.isAgent ? 'AGENT'
      : track.mode === MODE_LIVE ? 'LIVE' : 'LOOP';
    const modeBadgeClass = track.isAgent ? 'mode-badge-agent'
      : track.mode === MODE_LIVE ? 'mode-badge-live' : 'mode-badge-loop';

    const instrOptions = GM_PROGRAMS.map(([prog, name]) =>
      `<option value="${prog}" ${track.get('instrument') === prog ? 'selected' : ''}>${prog}: ${name}</option>`
    ).join('');

    // Top line: color pill, name, mode badge, remove button
    const topLine = document.createElement('div');
    topLine.className = 'track-piano-row-top';
    topLine.innerHTML = `
      <span class="track-color-pill" style="background:${track.color}"></span>
      <span class="track-name-label">${track.get('name')}</span>
      <span class="mode-badge ${modeBadgeClass}">${modeLabel}</span>
      ${!track.isAgent ? `<button class="btn btn-xs btn-danger btn-remove" title="Remove track">✕</button>` : ''}`;

    topLine.querySelector('.btn-remove')?.addEventListener('click', () => {
      this._app.removeTrack(track.id);
    });

    // Bottom line: controls
    const botLine = document.createElement('div');
    botLine.className = 'track-piano-row-bottom';

    // Instrument select
    const selInstr = document.createElement('select');
    selInstr.className = 'sel-instr sel-sm';
    selInstr.innerHTML = instrOptions;
    selInstr.style.maxWidth = '100%';
    selInstr.addEventListener('change', () => {
      track.set('instrument', +selInstr.value);
      this._audio.changeProgram(track);
    });
    botLine.appendChild(selInstr);

    if (!track.isAgent) {
      // Mode select
      const selMode = document.createElement('select');
      selMode.className = 'sel-mode sel-sm';
      selMode.innerHTML = `
        <option value="loop" ${track.mode === MODE_LOOP ? 'selected':''}>Loop</option>
        <option value="live" ${track.mode === MODE_LIVE ? 'selected':''}>Live</option>`;
      selMode.addEventListener('change', () => {
        track.set('mode', selMode.value);
        // Re-render to show/hide mode-specific controls
        this.setTracks(this._tracks);
      });
      botLine.appendChild(selMode);

      if (track.mode === MODE_LOOP) {
        // Upload button
        const btnUpload = document.createElement('button');
        btnUpload.className = 'btn btn-xs btn-secondary';
        btnUpload.textContent = '📂';
        btnUpload.title = 'Load MIDI loop';
        btnUpload.addEventListener('click', () => {
          const inp = document.createElement('input');
          inp.type   = 'file';
          inp.accept = '.mid,.midi';
          inp.onchange = async () => {
            const file = inp.files[0];
            if (!file) return;
            const bars = await this._parseMidiBars(file);
            track.setLoopBars(bars);
          };
          inp.click();
        });
        botLine.appendChild(btnUpload);
      }

      if (track.mode === MODE_LIVE) {
        // MIDI device select
        const selDev = document.createElement('select');
        selDev.className = 'sel-device sel-sm';
        selDev.innerHTML = `<option value="">— device —</option>` +
          this._midi.getDevices().map(d =>
            `<option value="${d.id}" ${track.get('midiDeviceId') === d.id ? 'selected':''}>${d.name}</option>`
          ).join('');
        selDev.addEventListener('change', () => {
          track.set('midiDeviceId', selDev.value || null);
          this._midi.setTracks(this._tracks);
        });
        botLine.appendChild(selDev);
      }

      // Volume slider
      const vol = document.createElement('input');
      vol.type  = 'range';
      vol.className = 'vol-slider';
      vol.min   = '0';
      vol.max   = '1';
      vol.step  = '0.05';
      vol.value = String(track.get('volume') ?? 1);
      vol.title = 'Volume';
      vol.addEventListener('input', () => {
        track.set('volume', +vol.value);
        this._audio.setVolume(track, +vol.value);
      });
      botLine.appendChild(vol);
    }

    row.appendChild(topLine);
    row.appendChild(botLine);
    return row;
  }

  _buildPromptRow(track) {
    const row = document.createElement('div');
    row.className = 'track-prompt-row';
    row.style.height = PROMPT_ROW_H + 'px';
    row.dataset.trackId = track.id;

    let label;
    if (track.isAgent) {
      label = '▶ agent';
    } else {
      const mode = track.mode === MODE_LIVE ? 'live' : 'loop';
      label = `${track.get('name').toLowerCase()} · ${mode}`;
    }
    row.textContent = label;
    return row;
  }

  async _parseMidiBars(file) {
    const buf = await file.arrayBuffer();
    try {
      if (typeof Midi !== 'undefined') {
        const midi = new Midi(buf);
        const ppq  = midi.header.ppq;
        const bars = [];
        const ticksPerBar = ppq * 4; // assume 4/4
        for (const track of midi.tracks) {
          for (const note of track.notes) {
            const barIdx = Math.floor(note.ticks / ticksPerBar);
            while (bars.length <= barIdx) bars.push({ notes: [], tsNum: 4, tsDen: 4 });
            bars[barIdx].notes.push({
              pitch:    note.midi,
              velocity: Math.round(note.velocity * 127),
              onset:    (note.ticks % ticksPerBar) / ticksPerBar,
              duration: note.durationTicks / ticksPerBar,
            });
          }
        }
        return bars;
      }
    } catch (e) {
      console.warn('MIDI parse error:', e);
    }
    return [];
  }
}

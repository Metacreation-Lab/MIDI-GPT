import { View } from '../core/mvc.js';
import { MODE_LOOP, MODE_LIVE, MODE_AGENT } from '../models/TrackModel.js';

const GM_PROGRAMS = [
  [0,'Acoustic Grand Piano'],[1,'Bright Acoustic Piano'],[4,'Electric Piano 1'],
  [24,'Acoustic Guitar (nylon)'],[25,'Acoustic Guitar (steel)'],[26,'Electric Guitar (jazz)'],
  [32,'Acoustic Bass'],[33,'Electric Bass (finger)'],[40,'Violin'],[41,'Viola'],
  [56,'Trumpet'],[57,'Trombone'],[60,'French Horn'],[68,'Oboe'],[71,'Clarinet'],
  [73,'Flute'],[80,'Square Wave'],[88,'Pad 1 (new age)'],[105,'Banjo'],
  [116,'Woodblock'],[118,'Synth Drum'],
];

export class TrackListView extends View {
  constructor(el, appCtrl, midiCtrl, audioCtrl, sessionModel) {
    super(el);
    this._app     = appCtrl;
    this._midi    = midiCtrl;
    this._audio   = audioCtrl;
    this._session = sessionModel;
    this._build();
    sessionModel.on('tracks:changed', (tracks) => this._renderTracks(tracks));
  }

  _build() {
    this.el.innerHTML = `
      <div class="track-list-header">
        <h3>Tracks</h3>
        <div class="track-add-btns">
          <button id="btnAddCond"  class="btn btn-sm btn-secondary">+ Cond</button>
          <button id="btnAddAgent" class="btn btn-sm btn-primary"   >+ Agent</button>
        </div>
      </div>
      <div id="trackRows"></div>`;

    this.el.querySelector('#btnAddCond').onclick  = () => {
      this._app.addConditioningTrack();
      this._renderTracks(this._app.getTracks());
    };
    this.el.querySelector('#btnAddAgent').onclick = () => {
      this._app.addAgentTrack();
      this._renderTracks(this._app.getTracks());
    };
  }

  _renderTracks(tracks) {
    const container = this.el.querySelector('#trackRows');
    container.innerHTML = '';
    for (const track of tracks) {
      container.appendChild(this._buildRow(track));
    }
  }

  _buildRow(track) {
    const row = document.createElement('div');
    row.className = 'track-row';
    row.dataset.trackId = track.id;

    const modeLabel = track.isAgent ? 'AGENT' : track.mode.toUpperCase();
    const instrOptions = GM_PROGRAMS.map(([prog, name]) =>
      `<option value="${prog}" ${track.get('instrument') === prog ? 'selected' : ''}>${prog}: ${name}</option>`
    ).join('');

    row.innerHTML = `
      <div class="track-color" style="background:${track.color}"></div>
      <div class="track-info">
        <span class="track-name">${track.get('name')}</span>
        <span class="track-mode-badge">${modeLabel}</span>
      </div>
      <div class="track-controls">
        ${!track.isAgent ? `
          <select class="sel-mode sel-sm">
            <option value="loop"  ${track.mode === MODE_LOOP ? 'selected':''}>Loop</option>
            <option value="live"  ${track.mode === MODE_LIVE ? 'selected':''}>Live MIDI</option>
          </select>
        ` : ''}
        <select class="sel-instr sel-sm">${instrOptions}</select>
        <input type="range" class="vol-slider" min="0" max="1" step="0.05"
               value="${track.get('volume')}" title="Volume">
        ${track.mode === MODE_LOOP && !track.isAgent ? `
          <button class="btn btn-xs btn-secondary btn-upload-loop">📂 Loop</button>
        ` : ''}
        ${track.mode === MODE_LIVE ? `
          <select class="sel-device sel-sm"><option value="">— MIDI device —</option>
            ${this._midi.getDevices().map(d =>
              `<option value="${d.id}">${d.name}</option>`
            ).join('')}
          </select>
          <input type="number" class="inp-ch inp-sm" min="1" max="16" placeholder="Ch">
        ` : ''}
        ${!track.isAgent ? `<button class="btn btn-xs btn-danger btn-remove-track">✕</button>` : ''}
      </div>`;

    // Mode switch
    const selMode = row.querySelector('.sel-mode');
    selMode?.addEventListener('change', () => {
      track.set('mode', selMode.value);
      this._renderTracks(this._app.getTracks());
    });

    // Instrument change
    const selInstr = row.querySelector('.sel-instr');
    selInstr?.addEventListener('change', () => {
      track.set('instrument', +selInstr.value);
      this._audio.changeProgram(track);
    });

    // Volume
    const volSlider = row.querySelector('.vol-slider');
    volSlider?.addEventListener('input', () => {
      track.set('volume', +volSlider.value);
      this._audio.setVolume(track, +volSlider.value);
    });

    // MIDI device / channel
    const selDev = row.querySelector('.sel-device');
    selDev?.addEventListener('change', () => {
      track.set('midiDeviceId', selDev.value || null);
      this._midi.setTracks(this._app.getTracks());
    });
    const inpCh = row.querySelector('.inp-ch');
    inpCh?.addEventListener('change', () => {
      track.set('midiChannel', inpCh.value ? +inpCh.value : null);
    });

    // Loop upload
    row.querySelector('.btn-upload-loop')?.addEventListener('click', () => {
      const inp = document.createElement('input');
      inp.type  = 'file';
      inp.accept = '.mid,.midi';
      inp.onchange = async () => {
        const file = inp.files[0];
        if (!file) return;
        const bars = await this._parseMidiBars(file);
        track.setLoopBars(bars);
      };
      inp.click();
    });

    // Remove
    row.querySelector('.btn-remove-track')?.addEventListener('click', () => {
      this._app.removeTrack(track.id);
    });

    return row;
  }

  async _parseMidiBars(file) {
    // Minimal MIDI parser: use Tone.js MIDI or a simple approach.
    // For now, fall back to Tone.Midi if available, else return empty.
    const buf = await file.arrayBuffer();
    try {
      if (typeof Midi !== 'undefined') {
        const midi = new Midi(buf);
        const bpm  = midi.header.tempos[0]?.bpm ?? 120;
        const ppq  = midi.header.ppq;
        const bars = [];
        // Group notes by bar (assuming 4/4 for now)
        const ticksPerBar = ppq * 4;
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
    } catch (e) { console.warn('MIDI parse error:', e); }
    return [];
  }
}

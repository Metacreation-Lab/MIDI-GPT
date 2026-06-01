// AppController — orchestrates session lifecycle, track registration, event routing.

import { TrackModel, MODE_LOOP, MODE_LIVE, MODE_AGENT, TRACK_TYPE_MELODIC, TRACK_TYPE_DRUM } from '../models/TrackModel.js';

export class AppController {
  constructor(sessionModel, oscBridge, midiCtrl, audioCtrl) {
    this._session = sessionModel;
    this._osc     = oscBridge;
    this._midi    = midiCtrl;
    this._audio   = audioCtrl;
    this._tracks  = [];   // [TrackModel]
    this._agentTrack = null;
    // pending generated notes per bar, assembled from open/note/close events
    this._pendingBar = null; // { trackId, barIndex, notes: [] }
    // Agent bars waiting for their playback tick. Keyed by barIndex.
    // Drained on bar:sent so playback aligns with the audible start of the
    // bar (works for both live and step modes — step pauses bar:sent until
    // the next stepBar() call, which is exactly the desired behaviour).
    this._pendingAgentBars = new Map();
  }

  init() {
    // OSC replies → session state
    this._osc.on('/midigpt/session/ready',   () => this._session.setState('INITIALIZING'));
    this._osc.on('/midigpt/capabilities', (payload) => {
      try {
        const caps = JSON.parse(payload);
        console.log('[capabilities]', caps);
        this._session.set('attrCaps', caps);
        this._session.emit('caps:update', caps);
      } catch (e) { console.warn('bad capabilities payload', e, payload); }
    });
    this._osc.on('/midigpt/session/started', () => {
      this._session.setState('RUNNING');
      this._midi.startLoop(
        this._session.get('bpm'),
        this._session.get('tsNum'),
        this._session.get('tsDen'),
      );
    });
    this._osc.on('/midigpt/session/stopped', () => {
      this._session.setState('STOPPED');
      this._midi.stopLoop();
    });

    this._osc.on('/midigpt/status', (s) => this._session.setServerStatus(s));
    this._osc.on('/midigpt/prompt/state', (payload) => {
      try {
        const snap = JSON.parse(payload);
        console.log('[prompt/state]', snap);
        this._session.emit('prompt:state', snap);
      } catch (e) { console.warn('bad prompt/state payload', e, payload); }
    });
    this._osc.on('/midigpt/error',  (code, msg) => console.error(`OSC error ${code}: ${msg}`));

    // Generated note assembly
    this._osc.on('/midigpt/generated/open',  (trackId, barIndex, _count) => {
      this._pendingBar = { trackId, barIndex, notes: [] };
      this._session.emit('generation:open', { trackId, barIndex });
    });
    this._osc.on('/midigpt/generated/note',  (trackId, barIndex, pitch, velocity, onset, duration) => {
      if (this._pendingBar?.barIndex === barIndex) {
        this._pendingBar.notes.push({ pitch, velocity, onset, duration });
      }
    });
    this._osc.on('/midigpt/generated/close', (trackId, barIndex) => {
      if (!this._pendingBar || this._pendingBar.barIndex !== barIndex) return;
      const { notes } = this._pendingBar;
      this._pendingBar = null;
      this._session.emit('generation:close', { trackId, barIndex, notes });
      if (this._agentTrack) {
        this._agentTrack.addGeneratedBar(barIndex, notes);
        // Queue for playback at the bar:sent tick for this barIndex. If the
        // bar already played (late arrival), play immediately.
        // currentBar = bar currently playing. If the generation arrived after
        // its own bar:sent already fired (barIndex <= currentBar), the queue
        // would never drain — play it immediately instead.
        const playhead = this._session.get('currentBar') ?? -1;
        if (barIndex <= playhead) {
          this._playAgentBarNow(notes);
        } else {
          this._pendingAgentBars.set(barIndex, notes);
        }
      }
    });
    this._osc.on('/midigpt/sampled/attrs', (payload) => {
      try {
        const s = JSON.parse(payload);
        this._session.recordSampled(
          s.start_bar | 0, s.end_bar | 0, s.attrs || {},
        );
      } catch (e) { console.warn('bad sampled/attrs payload', e, payload); }
    });

    this._osc.on('/midigpt/generated/features', (trackId, barIndex, density, pitch, velocity, maxPoly, meanDur, minPoly, minDur, maxDur) => {
      this._session.recordRealized(barIndex, {
        note_density:      density,
        mean_pitch:        pitch,
        mean_velocity:     velocity,
        min_polyphony:     minPoly,
        max_polyphony:     maxPoly,
        mean_duration:     meanDur,
        min_note_duration: minDur,
        max_note_duration: maxDur,
      });
    });

    // MIDI bar ticks → session bar counter
    this._midi.on('bar:tick', (bar) => this._session.set('currentBar', bar));

    // bar:end sent → timing tracker (may or may not lead to a generation;
    // during the buffer phase the server holds off until enough bars accumulate)
    this._midi.on('bar:sent', (bar) => {
      this._session.emit('bar:sent', bar);
      // Drain any queued agent bar whose audible start is this tick. Pause
      // (step mode) automatically defers — bar:sent only fires from tickBar.
      const pending = this._pendingAgentBars.get(bar);
      if (pending) {
        this._pendingAgentBars.delete(bar);
        this._playAgentBarNow(pending);
      }
    });

    // MIDI note-on → audio preview
    this._midi._onNoteOn = (track, pitch, velocity) => {
      this._audio.noteOn(track, pitch, velocity);
    };

    // Bridge connection
    this._osc.onConnect(()    => this._session.setBridgeStatus('connected'));
    this._osc.onDisconnect(() => this._session.setBridgeStatus('disconnected'));
  }

  _playAgentBarNow(notes) {
    if (!this._agentTrack) return;
    const bpm    = this._session.get('bpm');
    const tsNum  = this._session.get('tsNum');
    const tsDen  = this._session.get('tsDen');
    const barSec = (60 / bpm) * tsNum * (4 / tsDen);
    this._audio.scheduleBar(
      this._agentTrack, notes,
      Date.now() / 1000, barSec, 0,
    );
  }

  // ── Track management ─────────────────────────────────────────────────────

  addConditioningTrack(options = {}) {
    const t = new TrackModel(options);
    t.set('pieceIdx', this._tracks.length);
    this._tracks.push(t);
    this._syncControllers();
    this._session.emit('tracks:changed', this._tracks);
    return t;
  }

  addAgentTrack() {
    if (this._agentTrack) { console.warn('Agent track already exists'); return this._agentTrack; }
    const t = new TrackModel({ mode: MODE_AGENT, isAgent: true });
    t.set('pieceIdx', this._tracks.length);
    this._tracks.push(t);
    this._agentTrack = t;
    t.on('change', (patch) => {
      if ('trackType' in patch) this._updateAgentIsDrum();
    });
    this._updateAgentIsDrum();
    this._syncControllers();
    this._session.emit('tracks:changed', this._tracks);
    return t;
  }

  _updateAgentIsDrum() {
    const isDrum = this._agentTrack?.get('trackType') === TRACK_TYPE_DRUM;
    this._session.set('agentIsDrum', isDrum);
  }

  removeTrack(trackId) {
    const idx = this._tracks.findIndex(t => t.id === trackId);
    if (idx < 0) return;
    const t = this._tracks[idx];
    if (t.isAgent) {
      this._agentTrack = null;
      this._updateAgentIsDrum();
    }
    this._tracks.splice(idx, 1);
    // Re-index pieceIdx
    this._tracks.forEach((tr, i) => tr.set('pieceIdx', i));
    this._syncControllers();
    this._session.emit('tracks:changed', this._tracks);
  }

  getTracks() { return [...this._tracks]; }

  _syncControllers() {
    this._midi.setTracks(this._tracks);
    this._audio.setTracks(this._tracks);
  }

  // ── Session lifecycle ────────────────────────────────────────────────────

  connect() {
    this._osc.connect();
  }

  initSession(name = 'studio') {
    this._session.reset();
    this._osc.sessionInit(name);
  }

  startSession() {
    if (!this._agentTrack) { alert('Add an agent track first.'); return; }
    // Register all tracks with the OSC server
    for (const t of this._tracks) {
      this._osc.trackCreate(
        t.get('pieceIdx'),
        t.get('instrument'),
        t.get('trackType'),
        t.isAgent,
      );
    }
    // Push current params
    const params = this._session.get('params');
    for (const [k, v] of Object.entries(params)) {
      this._osc.paramSet(k, v);
    }
    // Push current attrs — skip any the checkpoint doesn't support.
    const attrs = this._session.get('attrs');
    const caps  = this._session.get('attrCaps');
    for (const [k, v] of Object.entries(attrs)) {
      if (caps && caps[k] === false) continue;
      this._osc.attrSet(k, v);
    }
    this._osc.sessionStart();
    this._audio.resume();
  }

  stopSession() {
    this._osc.sessionStop();
    this._midi.stopLoop();
  }
}

// MidiController — Web MIDI API, loop player, live MIDI routing, CC bindings.

import { MODE_LOOP, MODE_LIVE, MODE_AGENT } from '../models/TrackModel.js';
import { PARAM_META } from '../models/SessionModel.js';

export class MidiController {
  constructor(sessionModel, oscBridge) {
    this._session  = sessionModel;
    this._osc      = oscBridge;
    this._tracks   = [];       // [TrackModel]
    this._inputs   = new Map(); // deviceId → MIDIInput
    this._handlers = new Map(); // deviceId → listener fn

    this._loopInterval  = null;
    this._loopBarIndex  = 0;
    this._loopTickMs    = 500;  // recalculated on bpm/ts change
    this._stepMode      = false;
    this._loopTsNum     = 4;
    this._loopTsDen     = 4;
    this._captureMode   = false;
    this._captureTarget = null; // param name waiting for CC

    this._onCC     = null; // external callback: (ccNum, value, paramName?) => {}
    this._onNoteOn = null; // external callback for live playback preview
  }

  // ── MIDI access ──────────────────────────────────────────────────────────

  async requestAccess() {
    if (!navigator.requestMIDIAccess) {
      console.warn('Web MIDI API not supported in this browser');
      return;
    }
    try {
      const access = await navigator.requestMIDIAccess({ sysex: false });
      this._refreshInputs(access);
      access.onstatechange = () => this._refreshInputs(access);
    } catch (e) {
      console.error('MIDI access denied:', e);
    }
  }

  _refreshInputs(access) {
    this._inputs.clear();
    for (const [id, input] of access.inputs) {
      this._inputs.set(id, input);
    }
    this.emit('devices:changed', [...this._inputs.values()].map(i => ({ id: i.id, name: i.name })));
  }

  getDevices() {
    return [...this._inputs.values()].map(i => ({ id: i.id, name: i.name }));
  }

  // ── Track registration ───────────────────────────────────────────────────

  setTracks(tracks) {
    // Remove old live handlers
    for (const [id, fn] of this._handlers) {
      const input = this._inputs.get(id);
      if (input) input.onmidimessage = null;
    }
    this._handlers.clear();
    this._tracks = tracks;
    this._attachLiveHandlers();
  }

  _attachLiveHandlers() {
    for (const track of this._tracks) {
      if (track.mode !== MODE_LIVE || !track.get('midiDeviceId')) continue;
      const input = this._inputs.get(track.get('midiDeviceId'));
      if (!input) continue;
      const fn = (msg) => this._onMidiMessage(msg, track);
      input.onmidimessage = fn;
      this._handlers.set(track.get('midiDeviceId'), fn);
    }
  }

  _onMidiMessage(msg, track) {
    const [status, data1, data2] = msg.data;
    const type    = status & 0xF0;
    const channel = (status & 0x0F) + 1;

    const wantCh = track.get('midiChannel');
    if (wantCh !== null && channel !== wantCh) return;

    if (type === 0x90 && data2 > 0) {
      // Note On
      const pieceIdx = track.get('pieceIdx');
      if (pieceIdx === null) return;
      // onset/duration will be 0/0.5 for live — bar controller fills exact values
      this._osc.note(pieceIdx, data1, data2, 0.0, 0.5, this._loopBarIndex);
      this._onNoteOn?.(track, data1, data2);
    } else if (type === 0xB0) {
      // CC
      this._handleCC(data1, data2);
    }
  }

  // ── MIDI CC binding ──────────────────────────────────────────────────────

  startCapture(paramName) {
    this._captureMode   = true;
    this._captureTarget = paramName;
  }

  stopCapture() {
    this._captureMode   = false;
    this._captureTarget = null;
  }

  _handleCC(ccNum, value) {
    if (this._captureMode && this._captureTarget) {
      this._session.bindCC(ccNum, this._captureTarget);
      this._captureMode = false;
      this._captureTarget = null;
    }
    const bindings  = this._session.get('ccBindings');
    const paramName = bindings[ccNum];
    if (paramName) {
      const meta   = PARAM_META[paramName];
      let   mapped = value / 127;
      if (meta) {
        if (meta.type === 'bool') {
          mapped = value > 63;
        } else {
          mapped = meta.min + mapped * (meta.max - meta.min);
          if (meta.type === 'int') mapped = Math.round(mapped);
          else mapped = Math.round(mapped / meta.step) * meta.step;
        }
      }
      this._session.setParam(paramName, mapped);
      this._osc.paramSet(paramName, mapped);
    }
    this._onCC?.(ccNum, value, paramName);
  }

  // ── Loop player ──────────────────────────────────────────────────────────

  startLoop(bpm, tsNum, tsDen) {
    this.stopLoop();
    this._loopBarIndex = 0;
    this._loopTickMs   = (60000 / bpm) * tsNum * (4 / tsDen);
    this._loopTsNum    = tsNum;
    this._loopTsDen    = tsDen;
    if (this._stepMode) return; // wait for manual stepBar() calls
    // Fire bar 0 immediately. setInterval would otherwise wait a full
    // _loopTickMs before the first tick — the user-visible symptom is
    // "bars are playing one bar too late."
    this._tickBar(tsNum, tsDen);
    this._loopInterval = setInterval(() => this._tickBar(tsNum, tsDen), this._loopTickMs);
  }

  stopLoop() {
    if (this._loopInterval) { clearInterval(this._loopInterval); this._loopInterval = null; }
  }

  updateTempo(bpm, tsNum, tsDen) {
    if (this._loopInterval) this.startLoop(bpm, tsNum, tsDen);
  }

  setStepMode(on) {
    this._stepMode = !!on;
    if (on && this._loopInterval) {
      clearInterval(this._loopInterval);
      this._loopInterval = null;
    } else if (!on && this._loopBarIndex > 0 && !this._loopInterval) {
      this._loopInterval = setInterval(
        () => this._tickBar(this._loopTsNum, this._loopTsDen), this._loopTickMs);
    }
  }

  stepBar() {
    if (!this._stepMode) return;
    this._tickBar(this._loopTsNum, this._loopTsDen);
  }

  _tickBar(tsNum, tsDen) {
    const bar = this._loopBarIndex;

    // /bar/end refers to the bar whose audio JUST ended (the previous bar).
    // Sending barEnd(bar) at the START of bar would tell the server a bar
    // ended before it had played, advancing bars_completed by one too early
    // and causing generation to fire one bar ahead of the audible playhead.
    if (bar > 0) {
      this._osc.barEnd(bar - 1, tsNum, tsDen);
    }

    for (const track of this._tracks) {
      if (track.mode !== MODE_LOOP || track.isAgent) continue;
      const loopBar = track.advanceLoop();
      if (!loopBar) continue;
      const pieceIdx = track.get('pieceIdx');
      if (pieceIdx === null) continue;

      for (const note of (loopBar.notes ?? [])) {
        this._osc.note(
          pieceIdx,
          note.pitch,
          note.velocity,
          note.onset   ?? 0,
          note.duration ?? 0.5,
          bar,
        );
        // Local audio preview (loop tracks otherwise wouldn't be audible
        // until the OSC server echoed them back, which it doesn't).
        const onsetMs = (note.onset ?? 0) * this._loopTickMs;
        const durMs   = Math.max(50, (note.duration ?? 0.5) * this._loopTickMs);
        setTimeout(
          () => this._onNoteOn?.(track, note.pitch, note.velocity, durMs),
          onsetMs,
        );
      }
    }
    // bar:sent / bar:tick refer to the bar that is NOW starting to play
    // audibly. bar:sent drains queued agent notes for this bar.
    this.emit('bar:sent', bar);
    this.emit('bar:tick', bar);
    this._loopBarIndex++;
  }

  // Observable shim (simple)
  emit(event, data) { document.dispatchEvent(new CustomEvent(`midi:${event}`, { detail: data })); }
  on(event, fn)     { document.addEventListener(`midi:${event}`, e => fn(e.detail)); }
}

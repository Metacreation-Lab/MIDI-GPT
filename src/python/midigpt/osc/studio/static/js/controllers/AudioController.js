// AudioController — Spessasynth SF2 synthesis for human + agent tracks.

export class AudioController {
  constructor() {
    this._ctx      = null;
    this._synth    = null;
    this._ready    = false;
    this._tracks   = [];    // [TrackModel]
    this._channels = {};    // pieceIdx → MIDI channel (1–16)
  }

  async init(sf2Url = '/static/default.sf2') {
    if (this._ready) return;

    // Spessasynth loaded via CDN/module in index.html
    if (typeof Synthetizer === 'undefined') {
      console.error('Spessasynth not loaded — audio disabled');
      return;
    }
    try {
      this._ctx   = new AudioContext();
      const resp  = await fetch(sf2Url);
      const buf   = await resp.arrayBuffer();
      this._synth = new Synthetizer(this._ctx.destination, buf);
      this._ready = true;
      console.log('Spessasynth ready');
    } catch (e) {
      console.error('Spessasynth init failed:', e);
    }
  }

  setTracks(tracks) {
    this._tracks = tracks;
    // Assign MIDI channels 1–N (agent gets last slot)
    let ch = 1;
    this._channels = {};
    for (const t of tracks) {
      const idx = t.get('pieceIdx') ?? t.id;
      this._channels[idx] = ch <= 16 ? ch++ : 16;
      if (this._ready) {
        this._synth.programChange(this._channels[idx] - 1, t.get('instrument'));
      }
    }
  }

  // Play a note-on immediately (used for loop preview + live passthrough)
  noteOn(track, pitch, velocity, durationMs = 500) {
    if (!this._ready) return;
    const ch = this._channels[track.get('pieceIdx') ?? track.id];
    if (!ch) return;
    this._synth.noteOn(ch - 1, pitch, velocity);
    setTimeout(() => this._synth.noteOff(ch - 1, pitch), durationMs);
  }

  // Schedule a full bar of notes (agent track replay)
  scheduleBar(track, notes, barStartSec, barDurationSec) {
    if (!this._ready) return;
    const ch = this._channels[track.get('pieceIdx') ?? track.id];
    if (!ch) return;
    for (const n of notes) {
      const onsetSec  = barStartSec + n.onset    * barDurationSec;
      const durSec    = Math.max(0.05, n.duration * barDurationSec);
      const now       = this._ctx.currentTime;
      const delay     = Math.max(0, onsetSec - now);
      setTimeout(() => {
        if (!this._ready) return;
        this._synth.noteOn(ch - 1, n.pitch, n.velocity);
        setTimeout(() => this._synth.noteOff(ch - 1, n.pitch), durSec * 1000);
      }, delay * 1000);
    }
  }

  setVolume(track, gain) {
    // Spessasynth doesn't have per-channel volume API directly; use CC 7
    if (!this._ready) return;
    const ch  = this._channels[track.get('pieceIdx') ?? track.id];
    if (!ch) return;
    this._synth.controllerChange(ch - 1, 7, Math.round(gain * 127));
  }

  changeProgram(track) {
    if (!this._ready) return;
    const ch  = this._channels[track.get('pieceIdx') ?? track.id];
    if (!ch) return;
    this._synth.programChange(ch - 1, track.get('instrument'));
  }

  resume() { this._ctx?.resume(); }
}

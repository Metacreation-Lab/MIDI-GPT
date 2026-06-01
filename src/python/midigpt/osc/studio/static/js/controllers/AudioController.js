// AudioController — SoundFont-based playback via spessasynth_lib.
//
// Loads `/static/sf2/arachno.sf2` and routes each track to its own MIDI
// channel with a programChange derived from the track's GM `instrument` field.
// If the SF2 synth fails to load (e.g. offline, network error), we fall back
// to a plain oscillator synth so the UI is still usable.
//
// MIDI channel allocation:
//   - Drum tracks  → channel 9  (GM percussion)
//   - Melodic      → channels 0,1,2,3,4,5,6,7,8,10,11,... (skip 9)

const SF2_URL    = '/static/sf2/arachno.sf2';
const SPESS_VER  = '4.3.3';
// esm.sh resolves bare-specifier deps (spessasynth_core) so the ESM entry
// works in the browser without a bundler. The worklet processor on jsdelivr
// is pre-bundled and has no bare imports.
const SPESS_LIB  = `https://esm.sh/spessasynth_lib@${SPESS_VER}`;
const SPESS_WKLT = `https://cdn.jsdelivr.net/npm/spessasynth_lib@${SPESS_VER}/dist/spessasynth_processor.min.js`;

export class AudioController {
  constructor() {
    this._ctx       = null;
    this._ready     = false;
    this._tracks    = [];
    this._synth     = null;          // spessasynth Synthetizer (or null)
    this._gains     = {};            // pieceIdx → GainNode (fallback synth only)
    this._volumes   = {};            // pieceIdx → 0..1
    this._channels  = {};            // pieceIdx → MIDI channel
    this._nextCh    = 0;
  }

  async init() {
    if (this._ready) return;
    this._ctx = new AudioContext();
    this._ready = true;

    // Try to load spessasynth_lib + the Arachno SF2. If anything fails we
    // continue with the oscillator fallback — _synth stays null.
    try {
      await this._ctx.audioWorklet.addModule(SPESS_WKLT);
      const mod = await import(SPESS_LIB);
      const sfResp = await fetch(SF2_URL);
      if (!sfResp.ok) throw new Error(`SF2 fetch ${sfResp.status}`);
      const sfBuf = await sfResp.arrayBuffer();
      const synth = new mod.WorkletSynthesizer(this._ctx);
      await synth.soundBankManager.addSoundBank(sfBuf, 'main');
      await synth.isReady;
      synth.connect(this._ctx.destination);
      this._synth = synth;
      console.log('Audio ready (SF2 / spessasynth)');
      // Re-apply programs now that synth is up.
      for (const t of this._tracks) this._applyProgram(t);
    } catch (e) {
      console.warn('SF2 synth unavailable, falling back to oscillator:', e);
      this._synth = null;
      console.log('Audio ready (Web Audio oscillator synth)');
    }
  }

  setTracks(tracks) {
    this._tracks = tracks;
    for (const t of tracks) {
      const idx = t.get('pieceIdx') ?? t.id;
      if (this._channels[idx] === undefined) {
        this._channels[idx] = this._allocChannel(t);
      }
      this._applyProgram(t);
      if (this._gains[idx] === undefined && this._ctx && !this._synth) {
        const g = this._ctx.createGain();
        g.gain.value = this._volumes[idx] ?? t.get('volume') ?? 0.5;
        g.connect(this._ctx.destination);
        this._gains[idx] = g;
      }
    }
  }

  noteOn(track, pitch, velocity, durationMs = 500) {
    if (!this._ready) return;
    const idx = track.get('pieceIdx') ?? track.id;
    if (this._synth) {
      const ch = this._channels[idx] ?? 0;
      this._synth.noteOn(ch, pitch, velocity);
      setTimeout(() => this._synth?.noteOff(ch, pitch), durationMs);
    } else {
      const dest = this._gains[idx] ?? this._ctx.destination;
      this._playOscillator(pitch, velocity, durationMs / 1000, dest, track.isAgent);
    }
  }

  scheduleBar(track, notes, _barStartSec, barDurationSec, delayMs = 0) {
    if (!this._ready) return;
    const idx  = track.get('pieceIdx') ?? track.id;
    const now  = this._ctx.currentTime;
    const delaySec = delayMs / 1000;
    if (this._synth) {
      const ch = this._channels[idx] ?? 0;
      for (const n of notes) {
        const onset = (n.onset ?? 0) * barDurationSec;
        const dur   = Math.max(0.05, (n.duration ?? 0.5) * barDurationSec);
        const startMs = Math.max(0, delayMs + onset * 1000);
        setTimeout(() => this._synth?.noteOn(ch, n.pitch, n.velocity), startMs);
        setTimeout(() => this._synth?.noteOff(ch, n.pitch), startMs + dur * 1000);
      }
    } else {
      const dest = this._gains[idx] ?? this._ctx.destination;
      for (const n of notes) {
        const onset = n.onset ?? 0;
        const dur   = Math.max(0.05, (n.duration ?? 0.5) * barDurationSec);
        this._scheduleOscillator(
          pitch2freq(n.pitch), n.velocity, dur, dest,
          now + delaySec + onset * barDurationSec, track.isAgent);
      }
    }
  }

  setVolume(track, gain) {
    const idx = track.get('pieceIdx') ?? track.id;
    this._volumes[idx] = gain;
    if (this._synth) {
      // GM CC7 = channel volume; spessasynth accepts controllerChange
      const ch = this._channels[idx] ?? 0;
      this._synth.controllerChange?.(ch, 7, Math.round(gain * 127));
    } else if (this._gains[idx]) {
      this._gains[idx].gain.setTargetAtTime(gain, this._ctx.currentTime, 0.01);
    }
  }

  changeProgram(track) { this._applyProgram(track); }

  resume() { this._ctx?.resume(); }

  // ── Private ──────────────────────────────────────────────────────────────

  _allocChannel(track) {
    if (this._isDrum(track)) return 9;
    let ch = this._nextCh;
    if (ch === 9) ch = 10;
    this._nextCh = ch + 1;
    if (this._nextCh === 9) this._nextCh = 10;
    return ch % 16;
  }

  _isDrum(track) {
    // TRACK_TYPE_DRUM is typically id=2 in the encoder; tolerate either form.
    const tt = track.get('trackType');
    return tt === 2 || tt === 'drum' || track.get('isDrum') === true;
  }

  _applyProgram(track) {
    if (!this._synth) return;
    const idx  = track.get('pieceIdx') ?? track.id;
    const ch   = this._channels[idx] ?? 0;
    const prog = Math.max(0, Math.min(127, track.get('instrument') ?? 0));
    if (!this._isDrum(track)) {
      this._synth.programChange?.(ch, prog);
    }
  }

  _playOscillator(pitch, velocity, durSec, dest, isAgent) {
    const now = this._ctx.currentTime;
    this._scheduleOscillator(pitch2freq(pitch), velocity, durSec, dest, now, isAgent);
  }

  _scheduleOscillator(freq, velocity, durSec, dest, startSec, isAgent) {
    const ctx  = this._ctx;
    const amp  = (velocity / 127) * 0.25;
    const type = isAgent ? 'sawtooth' : 'triangle';

    const osc  = ctx.createOscillator();
    const env  = ctx.createGain();

    osc.type            = type;
    osc.frequency.value = freq;
    env.gain.setValueAtTime(0, startSec);
    env.gain.linearRampToValueAtTime(amp, startSec + 0.008);
    env.gain.setTargetAtTime(amp * 0.6, startSec + 0.008, 0.05);
    env.gain.setTargetAtTime(0, startSec + durSec * 0.8, 0.04);

    osc.connect(env);
    env.connect(dest);
    osc.start(startSec);
    osc.stop(startSec + durSec + 0.1);
  }
}

function pitch2freq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

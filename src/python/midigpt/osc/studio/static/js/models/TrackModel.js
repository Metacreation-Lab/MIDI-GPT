import { Model } from '../core/mvc.js';

let _nextId = 0;

// track_type values matching the OSC server (10 = melodic, 11 = drum)
export const TRACK_TYPE_MELODIC = 10;
export const TRACK_TYPE_DRUM    = 11;

// Conditioning track modes
export const MODE_LOOP  = 'loop';
export const MODE_LIVE  = 'live';
export const MODE_AGENT = 'agent';

const COLORS = ['#4e9eff','#4ecca3','#ff6b6b','#ffd166','#c77dff','#ff9f43'];

export class TrackModel extends Model {
  constructor({ mode = MODE_LOOP, isAgent = false } = {}) {
    super({
      id:         _nextId++,
      name:       isAgent ? 'Agent' : `Track ${_nextId}`,
      color:      COLORS[(_nextId - 1) % COLORS.length],
      mode:       isAgent ? MODE_AGENT : mode,
      isAgent,
      instrument: 0,       // GM program 0–127
      trackType:  TRACK_TYPE_MELODIC,
      ignore:     false,
      volume:     1.0,

      // Loop mode
      loopBars:     [],     // [{notes: [{pitch,velocity,onset,duration}], tsNum, tsDen}]
      loopPosition: 0,      // current bar index within loop

      // Live MIDI In mode
      midiDeviceId: null,
      midiChannel:  null,   // 1–16, null = all

      // Agent track — received notes per bar: { [barIndex]: [{pitch,velocity,onset,duration}] }
      generatedBars: {},

      // Server piece_idx assigned after track:create
      pieceIdx: null,
    });
  }

  get id()      { return this.get('id'); }
  get isAgent() { return this.get('isAgent'); }
  get mode()    { return this.get('mode'); }
  get color()   { return this.get('color'); }

  addGeneratedBar(barIndex, notes) {
    const bars = { ...this.get('generatedBars'), [barIndex]: notes };
    this.set('generatedBars', bars);
    this.emit('bar:generated', { barIndex, notes });
  }

  setLoopBars(bars) {
    this.set('loopBars', bars);
    this.set('loopPosition', 0);
  }

  // Called by loop player on each bar completion
  advanceLoop() {
    const bars = this.get('loopBars');
    if (!bars.length) return null;
    const pos  = this.get('loopPosition');
    const bar  = bars[pos];
    this.set('loopPosition', (pos + 1) % bars.length);
    return bar;
  }

  // Peek at a future loop bar without advancing (offset=0 → next bar after current position)
  peekLoop(offset) {
    const bars = this.get('loopBars');
    if (!bars.length) return null;
    return bars[(this.get('loopPosition') + offset) % bars.length];
  }
}

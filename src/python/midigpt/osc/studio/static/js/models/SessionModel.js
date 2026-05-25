import { Model } from '../core/mvc.js';

export const STATES = ['UNINITIALIZED', 'INITIALIZING', 'RUNNING', 'STOPPED'];

export const PARAM_DEFAULTS = {
  temperature:           1.0,
  model_dim:             8,
  buffer_bars:           4,
  lookahead_bars:        1,
  num_anticipated_bars:  1,
  mask_gap:              false,
  adapt_buffer:          true,
  warmup_policy:         'a_empty',
  sampling_seed:         -1,
  gen_timeout:           0,
  mask_mode:             'token',
};

// `structural: true` params shape the generation window/buffer and are locked
// once the session is RUNNING — changing them mid-flight would invalidate
// in-progress prompts.
export const PARAM_META = {
  temperature:          { type: 'float', min: 0.1, max: 3.0,  step: 0.05 },
  // model_dim is constrained to {4, 8, 12, 16}; the enum-like step is enforced
  // by the studio UI and the scenario generator. Server-side, buffer_bars must
  // be strictly less than model_dim.
  model_dim:            { type: 'int',   min: 4,   max: 16,   step: 4,    structural: true },
  buffer_bars:          { type: 'int',   min: 2,   max: 15,   step: 1,    structural: true },
  lookahead_bars:       { type: 'int',   min: 1,   max: 4,    step: 1,    structural: true },
  num_anticipated_bars: { type: 'int',   min: 1,   max: 4,    step: 1,    structural: true },
  mask_gap:             { type: 'bool'                                    },
  adapt_buffer:         { type: 'bool',                                   structural: true },
  warmup_policy:        { type: 'enum',  options: [
      ['a_empty',  'a · empty context'],
      ['a_masked', 'a · masked context'],
      ['b',          'b · AR bootstrap'],
      ['b_collapse', 'b · AR bootstrap (collapsed window)'],
    ], structural: true },
  sampling_seed:        { type: 'int',   min: -1,  max: 9999, step: 1    },
  gen_timeout:          { type: 'float', min: 0,   max: 60,   step: 0.5  },
  mask_mode:            { type: 'enum',  options: [
                            ['token',     'token (MaskBar)'],
                            ['attention', 'attention (span mask)'],
                          ] },
};

export const ATTR_DEFAULTS = {
  tension:           5,
  note_density:      -1,
  min_polyphony:     1,
  max_polyphony:     1,
  min_note_duration: -1,
  max_note_duration: -1,
};

// Attributes that only make sense for a drum agent track.
export const DRUM_ONLY_ATTRS = new Set(['note_density']);

export class SessionModel extends Model {
  constructor() {
    super({
      state:        'UNINITIALIZED',
      bridgeStatus: 'disconnected',
      serverStatus: 'idle',
      currentBar:   -1,
      bpm:          120,
      tsNum:        4,
      tsDen:        4,
      params:       { ...PARAM_DEFAULTS },
      attrs:        { ...ATTR_DEFAULTS },
      // realized attributes per bar: [{bar, tension, note_density, ...}]
      realizedAttrs: [],
      // sampled attribute tokens per generation window:
      // [{startBar, endBar, attrs: {note_density:int, ...}}]
      sampledAttrs:  [],
      // CC → param bindings: { [ccNum]: paramName }
      ccBindings:   {},
      // Per-attribute server capability (false = model doesn't tokenize this).
      attrCaps:     null,
      // True when the configured agent track is a drum track. Gates
      // drum-only attributes (e.g. note_density).
      agentIsDrum:  false,
    });
  }

  setState(s)        { this.set('state', s); }
  setBridgeStatus(s) { this.set('bridgeStatus', s); }
  setServerStatus(s) { this.set('serverStatus', s); }
  advanceBar()       { this.set('currentBar', this.get('currentBar') + 1); }

  setParam(name, value) {
    const params = { ...this.get('params'), [name]: value };
    this.set('params', params);
  }

  setAttr(name, value) {
    const attrs = { ...this.get('attrs'), [name]: value };
    this.set('attrs', attrs);
  }

  recordRealized(bar, feats) {
    const arr = [...this.get('realizedAttrs')];
    const existing = arr.findIndex(r => r.bar === bar);
    if (existing >= 0) arr[existing] = { bar, ...feats };
    else arr.push({ bar, ...feats });
    this.set('realizedAttrs', arr);
  }

  bindCC(ccNum, paramName) {
    const b = { ...this.get('ccBindings'), [ccNum]: paramName };
    this.set('ccBindings', b);
  }
  unbindCC(ccNum) {
    const b = { ...this.get('ccBindings') };
    delete b[ccNum];
    this.set('ccBindings', b);
  }

  recordSampled(startBar, endBar, attrs) {
    const arr = [...this.get('sampledAttrs'), { startBar, endBar, attrs }];
    this.set('sampledAttrs', arr);
  }

  reset() {
    this.update({
      state: 'UNINITIALIZED', currentBar: 0,
      realizedAttrs: [],
      sampledAttrs:  [],
      params: { ...PARAM_DEFAULTS },
      attrs:  { ...ATTR_DEFAULTS },
    });
  }
}

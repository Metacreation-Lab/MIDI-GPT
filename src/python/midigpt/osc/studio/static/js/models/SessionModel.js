import { Model } from '../core/mvc.js';

export const STATES = ['UNINITIALIZED', 'INITIALIZING', 'RUNNING', 'STOPPED'];

export const PARAM_DEFAULTS = {
  temperature:           1.0,
  model_dim:             4,
  buffer_bars:           8,
  lookahead_bars:        2,
  num_anticipated_bars:  1,
  mask_gap:              false,
  adapt_buffer:          true,
  sampling_seed:         -1,
  gen_timeout:           0,
  mask_mode:             'token',
};

export const PARAM_META = {
  temperature:          { type: 'float', min: 0.1, max: 3.0,  step: 0.05 },
  model_dim:            { type: 'int',   min: 1,   max: 16,   step: 1    },
  buffer_bars:          { type: 'int',   min: 2,   max: 32,   step: 1    },
  lookahead_bars:       { type: 'int',   min: 1,   max: 8,    step: 1    },
  num_anticipated_bars: { type: 'int',   min: 1,   max: 8,    step: 1    },
  mask_gap:             { type: 'bool'                                    },
  adapt_buffer:         { type: 'bool'                                    },
  sampling_seed:        { type: 'int',   min: -1,  max: 9999, step: 1    },
  gen_timeout:          { type: 'float', min: 0,   max: 60,   step: 0.5  },
  mask_mode:            { type: 'enum',  options: [
                            ['token',     'token (MaskBar)'],
                            ['attention', 'attention (span mask)'],
                          ] },
};

export const ATTR_DEFAULTS = {
  tension:       -1,   // -1 = not requested
  note_density:  -1,
  min_polyphony: -1,
  max_polyphony: -1,
};

export class SessionModel extends Model {
  constructor() {
    super({
      state:        'UNINITIALIZED',
      bridgeStatus: 'disconnected',
      serverStatus: 'idle',
      currentBar:   0,
      bpm:          120,
      tsNum:        4,
      tsDen:        4,
      params:       { ...PARAM_DEFAULTS },
      attrs:        { ...ATTR_DEFAULTS },
      // realized attributes per bar: [{bar, tension, note_density, ...}]
      realizedAttrs: [],
      // CC → param bindings: { [ccNum]: paramName }
      ccBindings:   {},
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

  reset() {
    this.update({
      state: 'UNINITIALIZED', currentBar: 0,
      realizedAttrs: [],
      params: { ...PARAM_DEFAULTS },
      attrs:  { ...ATTR_DEFAULTS },
    });
  }
}

import { View } from '../core/mvc.js';
import { PARAM_META, PARAM_DEFAULTS, ATTR_DEFAULTS } from '../models/SessionModel.js';

export class ParamPanelView extends View {
  constructor(el, sessionModel, oscBridge, midiCtrl) {
    super(el);
    this._session  = sessionModel;
    this._osc      = oscBridge;
    this._midi     = midiCtrl;
    this._build();
    sessionModel.on('change', p => {
      if ('ccBindings' in p) this._updateBindingBadges();
    });
  }

  _build() {
    this.el.innerHTML = `
      <div class="param-panel-inner">
        <div class="param-section">
          <h4>Session Parameters</h4>
          <div id="sessionParams" class="param-grid"></div>
        </div>
        <div class="param-section">
          <h4>Attribute Targets <small>(−1 = off)</small></h4>
          <div id="attrParams" class="param-grid"></div>
        </div>
        <div class="param-section">
          <h4>MIDI CC Bindings</h4>
          <div id="ccBindings" class="cc-bindings"></div>
          <button id="btnClearCC" class="btn btn-xs btn-danger">Clear all</button>
        </div>
      </div>`;

    const sessionGrid = this.el.querySelector('#sessionParams');
    for (const [name, meta] of Object.entries(PARAM_META)) {
      sessionGrid.appendChild(this._buildParamRow(name, meta, PARAM_DEFAULTS[name], (v) => {
        this._session.setParam(name, v);
        this._osc.paramSet(name, v);
      }));
    }

    const attrGrid = this.el.querySelector('#attrParams');
    for (const [name, def] of Object.entries(ATTR_DEFAULTS)) {
      const meta = { type: 'int', min: -1, max: 9, step: 1 };
      attrGrid.appendChild(this._buildParamRow(name, meta, def, (v) => {
        this._session.setAttr(name, v);
      }));
    }

    this.el.querySelector('#btnClearCC').onclick = () => {
      for (const k of Object.keys(this._session.get('ccBindings'))) {
        this._session.unbindCC(+k);
      }
      this._updateBindingBadges();
    };
  }

  _buildParamRow(name, meta, defaultVal, onChange) {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.dataset.param = name;

    if (meta.type === 'bool') {
      row.innerHTML = `
        <label class="param-label">${name}</label>
        <label class="toggle">
          <input type="checkbox" ${defaultVal ? 'checked' : ''}>
          <span class="slider-toggle"></span>
        </label>
        <button class="btn btn-xs btn-cc" title="Bind MIDI CC">CC</button>
        <span class="cc-badge"></span>`;
      const cb = row.querySelector('input[type=checkbox]');
      cb.onchange = () => onChange(cb.checked);
    } else {
      row.innerHTML = `
        <label class="param-label">${name}</label>
        <input type="range" class="param-slider"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${defaultVal}">
        <span class="param-value mono">${defaultVal}</span>
        <button class="btn btn-xs btn-cc" title="Bind MIDI CC">CC</button>
        <span class="cc-badge"></span>`;
      const slider = row.querySelector('.param-slider');
      const valEl  = row.querySelector('.param-value');
      slider.oninput = () => {
        const v = meta.type === 'int' ? +slider.value : parseFloat(slider.value);
        valEl.textContent = v;
        onChange(v);
      };
    }

    // CC bind button
    const btnCC = row.querySelector('.btn-cc');
    btnCC.onclick = () => {
      this._midi.startCapture(name);
      btnCC.textContent = '…';
      btnCC.classList.add('capturing');
      // Revert after 5s if no CC received
      setTimeout(() => {
        if (this._midi._captureMode) {
          this._midi.stopCapture();
          btnCC.textContent = 'CC';
          btnCC.classList.remove('capturing');
        }
      }, 5000);
    };

    return row;
  }

  _updateBindingBadges() {
    const bindings = this._session.get('ccBindings');
    // Build reverse map: paramName → ccNum
    const rev = {};
    for (const [cc, param] of Object.entries(bindings)) rev[param] = cc;

    for (const row of this.el.querySelectorAll('.param-row')) {
      const name  = row.dataset.param;
      const badge = row.querySelector('.cc-badge');
      const btnCC = row.querySelector('.btn-cc');
      if (!badge) continue;
      if (rev[name] != null) {
        badge.textContent = `CC${rev[name]}`;
        btnCC.textContent = 'CC';
        btnCC.classList.remove('capturing');
      } else {
        badge.textContent = '';
      }
    }

    // Also render CC list
    const list = this.el.querySelector('#ccBindings');
    list.innerHTML = '';
    for (const [cc, param] of Object.entries(bindings)) {
      const span = document.createElement('span');
      span.className = 'cc-tag';
      span.innerHTML = `CC${cc}→${param} <button class="cc-remove" data-cc="${cc}">✕</button>`;
      span.querySelector('.cc-remove').onclick = () => {
        this._session.unbindCC(+cc);
      };
      list.appendChild(span);
    }
  }
}

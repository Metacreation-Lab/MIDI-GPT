// ParamPanelView.js — right column vertical param panel

import { View } from '../core/mvc.js';
import { PARAM_META, PARAM_DEFAULTS, ATTR_DEFAULTS, DRUM_ONLY_ATTRS } from '../models/SessionModel.js';

export class ParamPanelView extends View {
  constructor(el, sessionModel, oscBridge, midiCtrl) {
    super(el);
    this._session = sessionModel;
    this._osc     = oscBridge;
    this._midi    = midiCtrl;
    this._build();
    sessionModel.on('change', (p) => {
      if ('ccBindings' in p)  this._updateBindingBadges();
      if ('state' in p)       this._updateLockState(p.state);
      if ('agentIsDrum' in p) this._applyAttrVisibility();
      if ('attrCaps' in p)    this._applyAttrVisibility();
    });
    this._updateLockState(sessionModel.get('state'));
  }

  _updateLockState(state) {
    const locked = state === 'RUNNING';
    for (const row of this.el.querySelectorAll('.param-row[data-structural="1"]')) {
      row.classList.toggle('locked', locked);
      for (const inp of row.querySelectorAll('input, select')) inp.disabled = locked;
    }
  }

  _build() {
    this.el.innerHTML = '';

    // ── Generation params ──────────────────────────────────────────────────
    const genSection = document.createElement('div');
    genSection.className = 'param-section';
    genSection.innerHTML = '<h4>Generation</h4><div id="sessionParams" class="param-grid"></div>';
    this.el.appendChild(genSection);

    const sessionGrid = genSection.querySelector('#sessionParams');
    for (const [name, meta] of Object.entries(PARAM_META)) {
      const row = this._buildParamRow(name, meta, PARAM_DEFAULTS[name], (v) => {
        this._session.setParam(name, v);
        this._osc.paramSet(name, v);
      });
      sessionGrid.appendChild(row);
    }

    // ── Attributes section ─────────────────────────────────────────────────
    const attrSection = document.createElement('div');
    attrSection.className = 'param-section';
    attrSection.innerHTML = '<h4>Attributes <small>(-1 = off)</small></h4><div id="attrParams" class="param-grid"></div>';
    this.el.appendChild(attrSection);

    const attrGrid = attrSection.querySelector('#attrParams');
    for (const [name, def] of Object.entries(ATTR_DEFAULTS)) {
      const meta = { type: 'int', min: -1, max: 9, step: 1 };
      const row  = this._buildParamRow(name, meta, def, (v) => {
        this._session.setAttr(name, v);
        this._osc.attrSet(name, v);
      });
      row.dataset.attr = name;
      attrGrid.appendChild(row);
    }

    // Hide attributes the loaded model can't tokenize (capabilities arrive
    // on /midigpt/capabilities right before /session/ready) or that are
    // drum-only when the agent track is melodic (e.g. note_density).
    this._attrGrid = attrGrid;
    this._applyAttrVisibility();
    this._session.on('caps:update', () => this._applyAttrVisibility());

    // ── MIDI CC section ────────────────────────────────────────────────────
    const ccSection = document.createElement('div');
    ccSection.className = 'param-section';
    ccSection.innerHTML = `
      <h4>MIDI CC Bindings</h4>
      <div id="ccBindings" class="cc-bindings"></div>
      <button id="btnClearCC" class="btn btn-xs btn-danger">Clear all</button>`;
    this.el.appendChild(ccSection);

    ccSection.querySelector('#btnClearCC').onclick = () => {
      for (const k of Object.keys(this._session.get('ccBindings'))) {
        this._session.unbindCC(+k);
      }
      this._updateBindingBadges();
    };
  }

  _buildParamRow(name, meta, defaultVal, onChange) {
    const row = document.createElement('div');
    row.className  = 'param-row';
    row.dataset.param = name;
    if (meta.structural) row.dataset.structural = '1';

    if (meta.type === 'bool') {
      row.innerHTML = `
        <label class="param-label">${name}</label>
        <label class="toggle">
          <input type="checkbox" ${defaultVal ? 'checked' : ''}>
          <span class="slider-toggle"></span>
        </label>
        <span class="cc-badge"></span>`;
      const cb = row.querySelector('input[type=checkbox]');
      cb.onchange = () => onChange(cb.checked);
    } else if (meta.type === 'enum') {
      const opts = meta.options.map(([v, label]) =>
        `<option value="${v}" ${v === defaultVal ? 'selected' : ''}>${label}</option>`
      ).join('');
      row.innerHTML = `
        <label class="param-label">${name}</label>
        <select class="param-select">${opts}</select>
        <span class="cc-badge"></span>`;
      const sel = row.querySelector('.param-select');
      sel.onchange = () => onChange(sel.value);
    } else {
      row.innerHTML = `
        <label class="param-label">${name}</label>
        <input type="range" class="param-slider"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${defaultVal}">
        <span class="param-value mono">${defaultVal}</span>
        <span class="cc-badge"></span>`;
      const slider = row.querySelector('.param-slider');
      const valEl  = row.querySelector('.param-value');
      slider.oninput = () => {
        const v = meta.type === 'int' ? +slider.value : parseFloat(slider.value);
        valEl.textContent = v;
        onChange(v);
      };
    }

    return row;
  }

  _applyAttrVisibility() {
    if (!this._attrGrid) return;
    const caps      = this._session.get('attrCaps');
    const isDrum    = !!this._session.get('agentIsDrum');
    for (const row of this._attrGrid.querySelectorAll('[data-attr]')) {
      const name = row.dataset.attr;
      const hideByCap  = caps && caps[name] === false;
      const hideByDrum = DRUM_ONLY_ATTRS.has(name) && !isDrum;
      row.style.display = (hideByCap || hideByDrum) ? 'none' : '';
    }
  }

  _updateBindingBadges() {
    const bindings = this._session.get('ccBindings');
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

    const list = this.el.querySelector('#ccBindings');
    if (!list) return;
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

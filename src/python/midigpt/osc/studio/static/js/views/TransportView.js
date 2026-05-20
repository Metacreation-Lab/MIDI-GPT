import { View } from '../core/mvc.js';

export class TransportView extends View {
  constructor(el, sessionModel, appCtrl) {
    super(el);
    this._session = sessionModel;
    this._app     = appCtrl;
    this._build();
    sessionModel.on('change', (patch) => this._onModelChange(patch));
  }

  _build() {
    this.el.innerHTML = `
      <div class="transport-inner">
        <div class="transport-session">
          <button id="btnConnect"    class="btn btn-secondary btn-sm">Connect</button>
          <button id="btnInitSession" class="btn btn-secondary btn-sm">Init</button>
          <button id="btnStart"      class="btn btn-primary   btn-sm" disabled>▶ Start</button>
          <button id="btnStop"       class="btn btn-danger    btn-sm" disabled>■ Stop</button>
        </div>
        <div class="transport-tempo">
          <label>BPM</label>
          <input id="inpBpm" type="number" min="40" max="240" value="120" class="inp-sm">
          <label>TS</label>
          <input id="inpTsNum" type="number" min="1" max="12" value="4" class="inp-sm w40">
          <span>/</span>
          <input id="inpTsDen" type="number" min="1" max="16" value="4" class="inp-sm w40">
        </div>
        <div class="transport-info">
          <span class="badge" id="bridgeStatus">disconnected</span>
          <span class="badge" id="serverStatus">idle</span>
          <span id="barCounter" class="mono">Bar: 0</span>
          <span id="sessionState" class="mono">UNINITIALIZED</span>
        </div>
      </div>`;

    this.el.querySelector('#btnConnect').onclick    = () => this._app.connect();
    this.el.querySelector('#btnInitSession').onclick = () => this._app.initSession();
    this.el.querySelector('#btnStart').onclick      = () => this._app.startSession();
    this.el.querySelector('#btnStop').onclick       = () => this._app.stopSession();

    const bpmInput = this.el.querySelector('#inpBpm');
    bpmInput.oninput = () => {
      this._session.set('bpm', +bpmInput.value);
      this._app._midi.updateTempo(
        +bpmInput.value,
        +this.el.querySelector('#inpTsNum').value,
        +this.el.querySelector('#inpTsDen').value,
      );
    };
  }

  _onModelChange(patch) {
    if ('bridgeStatus' in patch) {
      const el = this.el.querySelector('#bridgeStatus');
      el.textContent = patch.bridgeStatus;
      el.className   = `badge ${patch.bridgeStatus === 'connected' ? 'badge-ok' : 'badge-err'}`;
    }
    if ('serverStatus' in patch) {
      const el = this.el.querySelector('#serverStatus');
      el.textContent = patch.serverStatus;
    }
    if ('currentBar' in patch) {
      this.el.querySelector('#barCounter').textContent = `Bar: ${patch.currentBar}`;
    }
    if ('state' in patch) {
      const state = patch.state;
      this.el.querySelector('#sessionState').textContent = state;
      this.el.querySelector('#btnStart').disabled = state !== 'INITIALIZING';
      this.el.querySelector('#btnStop').disabled  = state !== 'RUNNING';
    }
  }
}

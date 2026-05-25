// DebugView — unified log of:
//   • OSC messages (sent green, received blue)
//   • /midigpt/debug prompt + token blocks (yellow header)
//
// /midigpt/debug arrives chunked as (target_bar, seq, total, text); we
// reassemble per target_bar and append the completed block.

const NOISY_OSC = new Set([
  '/midigpt/note',
  '/midigpt/bar/end',
  '/midigpt/generated/note',
  '/midigpt/generated/features',
]);

export class DebugView {
  constructor(rootEl, osc) {
    this._root = rootEl;
    this._pending = new Map(); // target_bar → { total, parts: [] }
    this._maxEntries = 200;
    this._showNotes = false;

    this._root.innerHTML = `
      <div class="debug-header">
        <span>Debug</span>
        <span>
          <label style="font-size:10px;color:var(--text-dim);cursor:pointer;">
            <input type="checkbox" data-act="notes" /> notes
          </label>
          <button type="button" data-act="clear">clear</button>
        </span>
      </div>
      <div class="debug-body"></div>
    `;
    this._body = this._root.querySelector('.debug-body');
    this._root.querySelector('[data-act="clear"]')
      .addEventListener('click', () => this.clear());
    this._root.querySelector('[data-act="notes"]')
      .addEventListener('change', (e) => { this._showNotes = e.target.checked; });

    osc.onSent((addr, args)     => this._onOsc('out', addr, args));
    osc.onReceived((addr, args) => this._onOsc('in',  addr, args));
    osc.on('/midigpt/debug', (targetBar, seq, total, chunk) =>
      this._onChunk(targetBar, seq, total, chunk));
  }

  _onOsc(dir, address, args) {
    if (!this._showNotes && NOISY_OSC.has(address)) return;
    if (address === '/midigpt/debug') return; // rendered separately
    const cls = dir === 'out' ? 'debug-out' : 'debug-in';
    const arrow = dir === 'out' ? '→' : '←';
    const text  = `${arrow} ${address} ${formatArgs(args)}`;
    this._appendLine(cls, text);
  }

  _onChunk(targetBar, seq, total, chunk) {
    let pend = this._pending.get(targetBar);
    if (!pend) {
      pend = { total, parts: new Array(total).fill(null) };
      this._pending.set(targetBar, pend);
    }
    pend.parts[seq] = chunk;
    if (pend.parts.every(p => p !== null)) {
      this._pending.delete(targetBar);
      this._appendBlock(targetBar, pend.parts.join(''));
    }
  }

  _appendLine(cls, text) {
    const line = document.createElement('div');
    line.className = `debug-line ${cls}`;
    line.textContent = text;
    this._body.appendChild(line);
    this._trim();
    this._body.scrollTop = this._body.scrollHeight;
  }

  _appendBlock(targetBar, text) {
    const block = document.createElement('div');
    block.className = 'debug-line debug-block';
    block.innerHTML = `<span class="debug-tag">▶ prompt target_bar=${targetBar}</span>\n`
      + escapeHtml(text);
    this._body.appendChild(block);
    this._trim();
    this._body.scrollTop = this._body.scrollHeight;
  }

  _trim() {
    while (this._body.children.length > this._maxEntries) {
      this._body.removeChild(this._body.firstChild);
    }
  }

  clear() {
    this._body.innerHTML = '';
    this._pending.clear();
  }
}

function formatArgs(args) {
  return (args ?? []).map(a => {
    if (typeof a === 'string') return JSON.stringify(a);
    if (typeof a === 'number' && !Number.isInteger(a)) return a.toFixed(3);
    return String(a);
  }).join(' ');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

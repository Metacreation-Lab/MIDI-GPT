// OscBridge — thin wrapper over socket.io connecting to the Flask bridge.
// Translates high-level events to/from raw {address, args} OSC envelopes.

export class OscBridge {
  constructor(url = '') {
    this._handlers = {};         // address → [fn]
    this._sentListeners = [];    // (address, args) → void
    this._recvListeners = [];    // (address, args) → void
    this._connHandlers = [];
    this._disconnHandlers = [];
    this._socket = null;
    this._url = url || window.location.origin;
  }

  connect() {
    // socket.io-client loaded via CDN in index.html
    this._socket = io(this._url);

    this._socket.on('connect',    () => this._connHandlers.forEach(fn => fn()));
    this._socket.on('disconnect', () => this._disconnHandlers.forEach(fn => fn()));
    this._socket.on('bridge:connected', () => this._connHandlers.forEach(fn => fn()));

    this._socket.on('osc:in', ({ address, args }) => {
      (this._handlers[address] ?? []).forEach(fn => fn(...args));
      (this._handlers['*'] ?? []).forEach(fn => fn(address, ...args));
      this._recvListeners.forEach(fn => fn(address, args));
    });
  }

  onConnect(fn)    { this._connHandlers.push(fn); }
  onDisconnect(fn) { this._disconnHandlers.push(fn); }

  on(address, fn) {
    (this._handlers[address] ??= []).push(fn);
  }

  // Wire-level taps for debugging.
  onSent(fn)     { this._sentListeners.push(fn); }
  onReceived(fn) { this._recvListeners.push(fn); }

  _logSent(address, args) {
    this._sentListeners.forEach(fn => fn(address, args));
  }

  // Low-level: send raw OSC
  send(address, ...args) {
    this._socket?.emit('osc:out', { address, args });
    this._logSent(address, args);
  }

  // Typed helpers — mirror the Flask bridge's named events
  sessionInit(name)   {
    this._socket?.emit('session:init',  { name });
    this._logSent('/midigpt/session/init', [name]);
  }
  sessionStart()      {
    this._socket?.emit('session:start', {});
    this._logSent('/midigpt/session/start', []);
  }
  sessionStop()       {
    this._socket?.emit('session:stop',  {});
    this._logSent('/midigpt/session/stop', []);
  }

  trackCreate(trackId, instrument, trackType, isAgent) {
    this._socket?.emit('track:create', { track_id: trackId, instrument, track_type: trackType, is_agent: isAgent ? 1 : 0 });
    this._logSent('/midigpt/track/create', [trackId, instrument, trackType, isAgent ? 1 : 0]);
  }
  trackRemove(trackId) {
    this._socket?.emit('track:remove', { track_id: trackId });
    this._logSent('/midigpt/track/remove', [trackId]);
  }

  note(trackId, pitch, velocity, onset, duration, barIndex) {
    this._socket?.emit('note', { track_id: trackId, pitch, velocity, onset, duration, bar_index: barIndex });
    this._logSent('/midigpt/note', [trackId, pitch, velocity, onset, duration, barIndex]);
  }
  barEnd(barIndex, tsNum, tsDen) {
    this._socket?.emit('bar:end', { bar_index: barIndex, ts_num: tsNum, ts_den: tsDen });
    this._logSent('/midigpt/bar/end', [barIndex, tsNum, tsDen]);
  }

  paramSet(name, value)     {
    this._socket?.emit('param:set',      { name, value });
    this._logSent('/midigpt/param/set', [name, value]);
  }
  paramSetOnce(name, value) {
    this._socket?.emit('param:set_once', { name, value });
    this._logSent('/midigpt/param/set_once', [name, value]);
  }
  paramReset(name)          {
    this._socket?.emit('param:reset',    { name });
    this._logSent('/midigpt/param/reset', [name]);
  }
  attrSet(name, value)      {
    this._socket?.emit('attr:set',       { name, value });
    this._logSent('/midigpt/attr/set', [name, value]);
  }
}

// Observable / Model / View base classes — adapted from mmm-studio pattern.

export class Observable {
  constructor() { this._listeners = {}; }
  on(event, fn) {
    (this._listeners[event] ??= []).push(fn);
    return () => this.off(event, fn);
  }
  off(event, fn) {
    this._listeners[event] = (this._listeners[event] ?? []).filter(f => f !== fn);
  }
  emit(event, data) {
    (this._listeners[event] ?? []).forEach(fn => fn(data));
    (this._listeners['*'] ?? []).forEach(fn => fn(event, data));
  }
}

export class Model extends Observable {
  constructor(initial = {}) {
    super();
    this._state = { ...initial };
  }
  get(key)       { return this._state[key]; }
  set(key, val)  { this._state[key] = val; this.emit('change', { [key]: val }); }
  update(patch)  { Object.assign(this._state, patch); this.emit('change', patch); }
  state()        { return { ...this._state }; }
}

export class View extends Observable {
  constructor(el) {
    super();
    this.el = typeof el === 'string' ? document.querySelector(el) : el;
  }
  render() {}
}

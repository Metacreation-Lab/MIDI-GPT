// Splitters — VSCode-style drag-to-resize for the main 3-column / 2-row layout.

const STORAGE_KEY = 'midigpt-studio-splitters';

function loadPrefs() {
  try   { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch { return {}; }
}
function savePrefs(p) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch {}
}

export function initSplitters() {
  const root = document.documentElement.style;
  const prefs = loadPrefs();

  if (prefs.headerW) root.setProperty('--header-w', prefs.headerW + 'px');
  if (prefs.paramsW) root.setProperty('--params-w', prefs.paramsW + 'px');
  if (prefs.attrH)   root.setProperty('--attr-h',   prefs.attrH   + 'px');

  for (const el of document.querySelectorAll('.splitter')) {
    el.addEventListener('pointerdown', (e) => startDrag(e, el, prefs));
  }
}

function startDrag(e, splitter, prefs) {
  e.preventDefault();
  splitter.setPointerCapture(e.pointerId);
  splitter.classList.add('dragging');

  const target = splitter.dataset.target;
  const startX = e.clientX;
  const startY = e.clientY;
  const root   = document.documentElement.style;
  const cs     = getComputedStyle(document.documentElement);

  let startHeader = parseFloat(cs.getPropertyValue('--header-w')) || 200;
  let startParams = parseFloat(cs.getPropertyValue('--params-w')) || 270;
  let startAttr   = parseFloat(cs.getPropertyValue('--attr-h'))   || 160;

  const onMove = (ev) => {
    if (target === 'left') {
      const w = Math.max(120, Math.min(500, startHeader + (ev.clientX - startX)));
      root.setProperty('--header-w', w + 'px');
      prefs.headerW = w;
    } else if (target === 'right') {
      const w = Math.max(160, Math.min(600, startParams - (ev.clientX - startX)));
      root.setProperty('--params-w', w + 'px');
      prefs.paramsW = w;
    } else if (target === 'attr') {
      const h = Math.max(60, Math.min(600, startAttr - (ev.clientY - startY)));
      root.setProperty('--attr-h', h + 'px');
      prefs.attrH = h;
    }
    window.dispatchEvent(new Event('resize'));
  };
  const onUp = () => {
    splitter.removeEventListener('pointermove', onMove);
    splitter.removeEventListener('pointerup',   onUp);
    splitter.classList.remove('dragging');
    savePrefs(prefs);
  };
  splitter.addEventListener('pointermove', onMove);
  splitter.addEventListener('pointerup',   onUp);
}

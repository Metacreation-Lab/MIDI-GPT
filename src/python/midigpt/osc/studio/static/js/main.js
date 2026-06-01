import { SessionModel }       from './models/SessionModel.js';
import { OscBridge }          from './controllers/OscBridge.js';
import { MidiController }     from './controllers/MidiController.js';
import { AudioController }    from './controllers/AudioController.js';
import { AppController }      from './controllers/AppController.js';
import { TransportView }      from './views/TransportView.js';
import { TrackHeaderView }    from './views/TrackHeaderView.js';
import { ContextView }        from './views/ContextView.js';
import { AttributeGraphView } from './views/AttributeGraphView.js';
import { ParamPanelView }     from './views/ParamPanelView.js';
import { TimingView }         from './views/TimingView.js';
import { DebugView }          from './views/DebugView.js';
import { initSplitters }      from './views/Splitters.js';

async function boot() {
  initSplitters();
  const session = new SessionModel();
  const osc     = new OscBridge();
  const midi    = new MidiController(session, osc);
  const audio   = new AudioController();
  const app     = new AppController(session, osc, midi, audio);
  app.init();

  const transport    = new TransportView(
    document.getElementById('transport'), session, app);

  const contextView  = new ContextView(
    document.getElementById('contextCanvas'), session);
  contextView.setServerCanvas(document.getElementById('serverCanvas'));
  document.getElementById('simple-mode-toggle')
    ?.addEventListener('change', (e) => contextView.setSimpleMode(e.target.checked));

  const compareStatus = document.getElementById('viz-compare-status');
  session.on('prompt:state', (snap) => {
    if (!compareStatus) return;
    const r = contextView.compareWithServerState(snap);
    compareStatus.textContent = r.text;
    compareStatus.className = `viz-compare-status ${r.ok ? 'ok' : 'bad'}`;
  });

  const attrGraph    = new AttributeGraphView(
    document.getElementById('attrCanvas'), session);

  const paramPanel   = new ParamPanelView(
    document.getElementById('param-panel'), session, osc, midi);

  const timing = new TimingView(document.getElementById('timing-panel'), session);

  const debugView = new DebugView(document.getElementById('debug-panel'), osc);

  const trackHeaders = new TrackHeaderView(
    document.getElementById('track-headers'), app, midi, audio, session);

  session.on('tracks:changed', (tracks) => {
    contextView.setTracks(tracks);
    trackHeaders.setTracks(tracks);
    audio.setTracks(tracks);
  });

  session.on('generation:open',  ({ trackId, barIndex }) => {
    contextView.onGeneratedOpen(trackId, barIndex);
  });

  session.on('generation:close', ({ trackId, barIndex, notes }) => {
    contextView.onGeneratedClose(trackId, barIndex, notes);
  });

  // Override AppController's noteOn to also drive ContextView live notes
  midi._onNoteOn = (track, pitch, velocity, durationMs) => {
    audio.noteOn(track, pitch, velocity, durationMs);
    contextView.onLiveNote(track, pitch, velocity);
  };

  document.getElementById('btnStart')?.addEventListener('click', async () => {
    await audio.init();
    audio.setTracks(app.getTracks());
  }, { once: true });

  osc.connect();
  await midi.requestAccess();

  // Default setup: piano_bell loop + agent track, ready to Init + Start.
  await setupDefaultTracks(app);

  window._studio = { session, osc, midi, audio, app };
  console.log('MIDI-GPT Studio ready.');
}

async function setupDefaultTracks(app) {
  const cond = app.addConditioningTrack({ name: 'Piano' });
  try {
    const buf = await fetch('/static/midi/piano_bell.mid').then(r => r.arrayBuffer());
    if (typeof Midi !== 'undefined') {
      const m  = new Midi(buf);
      const ppq = m.header.ppq;
      const bars = [];
      const ticksPerBar = ppq * 4;
      for (const trk of m.tracks) {
        for (const note of trk.notes) {
          const barIdx = Math.floor(note.ticks / ticksPerBar);
          while (bars.length <= barIdx) bars.push({ notes: [], tsNum: 4, tsDen: 4 });
          bars[barIdx].notes.push({
            pitch:    note.midi,
            velocity: Math.round(note.velocity * 127),
            onset:    (note.ticks % ticksPerBar) / ticksPerBar,
            duration: note.durationTicks / ticksPerBar,
          });
        }
      }
      cond.setLoopBars(bars);
    }
  } catch (e) { console.warn('Failed to load default piano_bell.mid', e); }
  const agent = app.addAgentTrack();
  agent.set('instrument', 4); // Electric Piano 1
}

document.addEventListener('DOMContentLoaded', boot);

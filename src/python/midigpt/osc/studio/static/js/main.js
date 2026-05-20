import { SessionModel }      from './models/SessionModel.js';
import { OscBridge }         from './controllers/OscBridge.js';
import { MidiController }    from './controllers/MidiController.js';
import { AudioController }   from './controllers/AudioController.js';
import { AppController }     from './controllers/AppController.js';
import { TransportView }     from './views/TransportView.js';
import { TrackListView }     from './views/TrackListView.js';
import { PianoRollView }     from './views/PianoRollView.js';
import { AttributeGraphView} from './views/AttributeGraphView.js';
import { ParamPanelView }    from './views/ParamPanelView.js';

async function boot() {
  // Models
  const session = new SessionModel();

  // Infrastructure
  const osc   = new OscBridge();
  const midi  = new MidiController(session, osc);
  const audio = new AudioController();
  const app   = new AppController(session, osc, midi, audio);

  // Wire app logic
  app.init();

  // Views
  const transport = new TransportView(
    document.getElementById('transport'), session, app);
  const trackList = new TrackListView(
    document.getElementById('trackList'), app, midi, audio, session);
  const pianoRoll = new PianoRollView(
    document.getElementById('pianoRoll'), session);
  const attrGraph = new AttributeGraphView(
    document.getElementById('attrGraph'), session);
  const paramPanel = new ParamPanelView(
    document.getElementById('paramPanel'), session, osc, midi);

  // Keep piano roll in sync with tracks
  session.on('tracks:changed', (tracks) => pianoRoll.setTracks(tracks));

  // Init MIDI
  await midi.requestAccess();

  // Init audio (requires a user gesture — triggered on first Start click)
  document.getElementById('btnStart')?.addEventListener('click', async () => {
    await audio.init('/static/default.sf2');
  }, { once: true });

  // Connect WebSocket bridge
  osc.connect();

  // Debug helper
  window._studio = { session, osc, midi, audio, app };
  console.log('MIDI-GPT Studio ready. Access via window._studio.');
}

document.addEventListener('DOMContentLoaded', boot);

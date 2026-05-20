import argparse
import logging
import queue
import threading
from typing import Optional

try:
    from pythonosc import dispatcher, osc_server, udp_client
    from pythonosc.osc_message_builder import OscMessageBuilder
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "python-osc is required for the OSC server. "
        "Install it with: pip install midigpt[realtime]"
    ) from None

from midigpt.inference import InferenceEngine
from midigpt._types import Score

from .piece_state import PieceState, bar_ticks
from .gen import (
    PARAM_DEFAULTS,
    compute_target_bar,
    compute_num_anticipation,
    run_inference,
    compute_bar_features,
    validate_param,
)

log = logging.getLogger(__name__)

ERR_INVALID_STATE   = 1
ERR_UNKNOWN_TRACK   = 2
ERR_DUPLICATE_TRACK = 3
ERR_INVALID_PARAM   = 4
ERR_GENERATION      = 5
ERR_NO_AGENT        = 6
ERR_MULTI_AGENT     = 7
ERR_AGENT_NOTE      = 8


class MidiGPTServer:
    STATES = ("UNINITIALIZED", "INITIALIZING", "RUNNING", "STOPPED")

    def __init__(self, ckpt: str, listen_port: int, max_attempts: int) -> None:
        self._ckpt = ckpt
        self._listen_port = listen_port
        self._max_attempts = max_attempts

        self._state = "UNINITIALIZED"
        self._state_lock = threading.Lock()

        self._piece: Optional[PieceState] = None

        self._client: Optional[udp_client.SimpleUDPClient] = None
        self._client_lock = threading.Lock()

        self._params: dict = dict(PARAM_DEFAULTS)
        self._params["max_attempts"] = max_attempts
        self._params_lock = threading.Lock()

        self._once_params: dict = {}

        self._gen_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self._gen_thread = threading.Thread(
            target=self._gen_worker, daemon=True, name="midigpt-gen"
        )

        log.info("Loading checkpoint: %s", ckpt)
        self._engine: InferenceEngine = InferenceEngine.from_checkpoint(ckpt)
        log.info("Checkpoint loaded and warmed up")

        self._gen_thread.start()

    # ── Reply helpers ─────────────────────────────────────────────────────────

    def _send(self, address: str, *args) -> None:
        with self._client_lock:
            client = self._client
        if client is None:
            return
        try:
            builder = OscMessageBuilder(address=address)
            for arg in args:
                if isinstance(arg, bool):
                    builder.add_arg(arg, "T" if arg else "F")
                elif isinstance(arg, int):
                    builder.add_arg(arg, "i")
                elif isinstance(arg, float):
                    builder.add_arg(arg, "f")
                elif isinstance(arg, str):
                    builder.add_arg(arg, "s")
                else:
                    builder.add_arg(arg)
            client.send(builder.build())
        except Exception as exc:  # noqa: BLE001
            log.warning("OSC send failed (%s %s): %s", address, args, exc)

    def _error(self, code: int, msg: str) -> None:
        log.warning("OSC error %d: %s", code, msg)
        self._send("/midigpt/error", code, msg)

    def _update_client(self, client_addr: tuple) -> None:
        ip, port = client_addr
        with self._client_lock:
            if (self._client is None
                    or self._client._address != ip
                    or self._client._port != port):
                self._client = udp_client.SimpleUDPClient(ip, port)

    # ── State helpers ─────────────────────────────────────────────────────────

    def _require_state(self, client_addr, *allowed) -> bool:
        self._update_client(client_addr)
        with self._state_lock:
            if self._state not in allowed:
                self._error(
                    ERR_INVALID_STATE,
                    f"Message not allowed in state {self._state!r}"
                )
                return False
        return True

    def _set_state(self, new_state: str) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        log.info("State: %s → %s", old, new_state)

    def _get_state(self) -> str:
        with self._state_lock:
            return self._state

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def _get_params(self) -> dict:
        with self._params_lock:
            merged = dict(self._params)
            merged.update(self._once_params)
        return merged

    def _clear_once(self) -> None:
        with self._params_lock:
            self._once_params.clear()

    # ── OSC handlers: session ─────────────────────────────────────────────────

    def handle_session_init(self, client_addr, _address, *args) -> None:
        self._update_client(client_addr)
        session_name = args[0] if args else "unnamed"
        log.info("/session/init name=%r", session_name)

        with self._state_lock:
            self._state = "INITIALIZING"

        resolution = self._engine._tokenizer._vocab.config().resolution
        self._piece = PieceState(resolution=resolution)

        with self._params_lock:
            self._params = dict(PARAM_DEFAULTS)
            self._params["max_attempts"] = self._max_attempts
            self._once_params.clear()

        self._send("/midigpt/session/ready")

    def handle_session_start(self, client_addr, _address, *_args) -> None:
        if not self._require_state(client_addr, "INITIALIZING"):
            return
        piece = self._piece
        if piece is None or not piece.has_agent():
            self._error(ERR_NO_AGENT, "No agent track defined")
            return
        if not piece.has_conditioning_tracks():
            self._error(ERR_INVALID_STATE, "No conditioning tracks defined")
            return

        params = self._get_params()
        B = params["buffer_bars"]
        if B < 2:
            self._error(ERR_INVALID_PARAM, f"buffer_bars={B} < 2")
            return
        if B < 4:
            log.warning("buffer_bars=%d < 4 — very little context at first generation", B)

        self._set_state("RUNNING")
        self._send("/midigpt/session/started")
        log.info("/session/start ok — buffer=%d lookahead=%d j=%d model_dim=%d",
                 B, params["lookahead_bars"], params["num_anticipated_bars"],
                 params["model_dim"])

    def handle_session_stop(self, client_addr, _address, *_args) -> None:
        self._update_client(client_addr)
        log.info("/session/stop")
        self._set_state("STOPPED")
        self._send("/midigpt/session/stopped")

    # ── OSC handlers: track management ───────────────────────────────────────

    def handle_track_create(self, client_addr, _address, *args) -> None:
        if not self._require_state(client_addr, "INITIALIZING", "RUNNING"):
            return
        if len(args) < 4:
            self._error(ERR_INVALID_PARAM,
                        "/track/create requires track_id instrument track_type is_agent")
            return
        track_id, instrument, track_type, is_agent_int = (
            int(args[0]), int(args[1]), int(args[2]), int(args[3])
        )
        is_agent = bool(is_agent_int)

        err = self._piece.create_track(track_id, instrument, track_type, is_agent)
        if err:
            code = ERR_DUPLICATE_TRACK if "Duplicate" in err else ERR_MULTI_AGENT
            self._error(code, err)
            return
        log.info("Track created: id=%d inst=%d type=%d agent=%s",
                 track_id, instrument, track_type, is_agent)

    def handle_track_remove(self, client_addr, _address, *args) -> None:
        if not self._require_state(client_addr, "INITIALIZING", "RUNNING"):
            return
        if not args:
            self._error(ERR_INVALID_PARAM, "/track/remove requires track_id")
            return
        track_id = int(args[0])
        err = self._piece.remove_track(track_id)
        if err:
            self._error(ERR_UNKNOWN_TRACK if "Unknown" in err else ERR_INVALID_STATE, err)
            return
        log.info("Track removed: id=%d", track_id)

    def handle_track_set_ignore(self, client_addr, _address, *args) -> None:
        if not self._require_state(client_addr, "INITIALIZING", "RUNNING"):
            return
        if len(args) < 2:
            self._error(ERR_INVALID_PARAM, "/track/set_ignore requires track_id ignored")
            return
        track_id, ignored = int(args[0]), int(args[1])
        err = self._piece.set_track_param(track_id, "ignore", ignored)
        if err:
            self._error(ERR_UNKNOWN_TRACK, err)

    # ── OSC handlers: note input ──────────────────────────────────────────────

    def handle_note(self, client_addr, _address, *args) -> None:
        if not self._require_state(client_addr, "RUNNING"):
            return
        if len(args) < 6:
            self._error(ERR_INVALID_PARAM,
                        "/note requires track_id pitch velocity onset duration bar_index")
            return
        track_id  = int(args[0])
        pitch     = int(args[1])
        velocity  = int(args[2])
        onset     = float(args[3])
        duration  = float(args[4])
        bar_index = int(args[5])

        err = self._piece.push_note(track_id, pitch, velocity, onset, duration, bar_index)
        if err == "agent_track_note_ignored":
            self._error(ERR_AGENT_NOTE, "Note for agent track ignored")
        elif err:
            self._error(ERR_UNKNOWN_TRACK, err)

    # ── OSC handlers: bar control ─────────────────────────────────────────────

    def handle_bar_end(self, client_addr, _address, *args) -> None:
        if not self._require_state(client_addr, "RUNNING"):
            return
        if len(args) < 3:
            self._error(ERR_INVALID_PARAM, "/bar/end requires bar_index ts_num ts_den")
            return
        bar_index = int(args[0])
        ts_num    = int(args[1])
        ts_den    = int(args[2])

        piece = self._piece
        piece.end_bar(bar_index, ts_num, ts_den)
        log.debug("/bar/end bar=%d ts=%d/%d completed=%d",
                  bar_index, ts_num, ts_den, piece.bars_completed)

        self._maybe_trigger_generation(ts_num, ts_den)  # noqa: not used yet, reserved for future ts-aware scheduling

    def _maybe_trigger_generation(self, _last_ts_num: int, _last_ts_den: int) -> None:
        piece = self._piece
        params = self._get_params()

        k = params["lookahead_bars"]
        B = params["buffer_bars"]
        j = params["num_anticipated_bars"]
        adapt = params["adapt_buffer"]

        target_bar = compute_target_bar(piece.bars_completed, k, B, adapt)
        if target_bar is None:
            return

        piece.extend_for_generation(target_bar, j)
        total_bars = target_bar + j
        num_anticipation = compute_num_anticipation(target_bar, j, total_bars)
        if num_anticipation <= 0:
            return

        request = {
            "target_bar":       target_bar,
            "num_anticipation": num_anticipation,
            "agent_track_id":   piece.agent_track_id,
            "params":           params,
        }

        try:
            self._gen_queue.put_nowait(request)
            log.info("Generation queued: target=%d j=%d playhead=%d",
                     target_bar, num_anticipation, piece.bars_completed)
        except queue.Full:
            log.warning("Generation skipped (inference still running for prior step)")

    # ── OSC handlers: parameters ──────────────────────────────────────────────

    def handle_param_set(self, client_addr, _address, *args) -> None:
        self._update_client(client_addr)
        if len(args) < 2:
            self._error(ERR_INVALID_PARAM, "/param/set requires param_name value")
            return
        name = str(args[0])
        value = _coerce_param(name, args[1])
        err = validate_param(name, value)
        if err:
            self._error(ERR_INVALID_PARAM, err)
            return
        with self._params_lock:
            self._params[name] = value
        log.debug("param set %s = %r", name, value)

    def handle_param_set_once(self, client_addr, _address, *args) -> None:
        self._update_client(client_addr)
        if len(args) < 2:
            self._error(ERR_INVALID_PARAM, "/param/set_once requires param_name value")
            return
        name = str(args[0])
        value = _coerce_param(name, args[1])
        err = validate_param(name, value)
        if err:
            self._error(ERR_INVALID_PARAM, err)
            return
        with self._params_lock:
            self._once_params[name] = value
        log.debug("param set_once %s = %r", name, value)

    def handle_param_reset(self, client_addr, _address, *args) -> None:
        self._update_client(client_addr)
        if not args:
            self._error(ERR_INVALID_PARAM, "/param/reset requires param_name")
            return
        name = str(args[0])
        if name not in PARAM_DEFAULTS:
            self._error(ERR_INVALID_PARAM, f"Unknown parameter: {name!r}")
            return
        with self._params_lock:
            self._params[name] = PARAM_DEFAULTS[name]
            self._once_params.pop(name, None)

    def handle_param_reset_all(self, client_addr, _address, *_args) -> None:
        self._update_client(client_addr)
        with self._params_lock:
            self._params = dict(PARAM_DEFAULTS)
            self._params["max_attempts"] = self._max_attempts
            self._once_params.clear()
        log.info("All params reset to defaults")

    def handle_track_param_set(self, client_addr, _address, *args) -> None:
        self._update_client(client_addr)
        if len(args) < 3:
            self._error(ERR_INVALID_PARAM,
                        "/track/param/set requires track_id param_name value")
            return
        track_id = int(args[0])
        name     = str(args[1])
        value    = args[2]
        err = self._piece.set_track_param(track_id, name, value)
        if err:
            self._error(
                ERR_UNKNOWN_TRACK if "Unknown track" in err else ERR_INVALID_PARAM,
                err
            )

    # ── Generation worker thread ──────────────────────────────────────────────

    def _gen_worker(self) -> None:
        log.info("Generation worker started")
        while True:
            try:
                req = self._gen_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            target_bar       = req["target_bar"]
            num_anticipation = req["num_anticipation"]
            agent_track_id   = req["agent_track_id"]
            params           = req["params"]

            log.info("Inference start: target=%d j=%d", target_bar, num_anticipation)
            self._send("/midigpt/status", "generating")

            timeout = params.get("gen_timeout", 0)
            _res: list = [None]
            _exc: list = [None]

            def _infer():
                try:
                    score   = self._piece.to_score()
                    request = self._piece.build_generation_request(
                        target_bar, num_anticipation, params
                    )
                    _res[0], _ = run_inference(self._engine, score, request)
                except Exception as exc:  # noqa: BLE001
                    _exc[0] = exc

            _t = threading.Thread(target=_infer, daemon=True)
            _t.start()
            _t.join(timeout=timeout if timeout > 0 else None)

            if _t.is_alive():
                log.warning("Inference timeout after %.1fs — skipping bar", timeout)
                self._error(ERR_GENERATION, f"generation timeout ({timeout}s)")
                self._clear_once()
                self._send("/midigpt/status", "ready")
                continue

            if _exc[0] is not None:
                log.error("Inference failed: %s", _exc[0])
                self._error(ERR_GENERATION, str(_exc[0]))
                self._clear_once()
                self._send("/midigpt/status", "ready")
                continue

            result_score: Score = _res[0]
            log.info("Inference done")

            generated = self._piece.merge_generated(
                result_score, target_bar, num_anticipation, result_score.resolution
            )

            for b_global, inline_events, (ts_n, ts_d) in generated:
                self._send_generated_bar(
                    agent_track_id, b_global, inline_events,
                    ts_n, ts_d, result_score.resolution
                )

            self._clear_once()
            self._send("/midigpt/status", "ready")

    def _send_generated_bar(
        self,
        track_id: int,
        bar_index: int,
        events: list,
        ts_num: int,
        ts_den: int,
        result_resolution: int,
    ) -> None:
        note_ons = [e for e in events if e.get("velocity", 0) > 0]
        ticks = bar_ticks(ts_num, ts_den, result_resolution)

        self._send("/midigpt/generated/open", track_id, bar_index, len(note_ons))

        for ev in note_ons:
            onset    = ev["time"] / ticks if ticks > 0 else 0.0
            duration = ev.get("internal_duration", 1) / ticks if ticks > 0 else 0.0
            onset    = max(0.0, min(onset, 0.9999))
            duration = max(0.0001, min(duration, 1.0))
            self._send(
                "/midigpt/generated/note",
                track_id, bar_index,
                int(ev["pitch"]), int(ev["velocity"]),
                float(onset), float(duration),
            )

        self._send("/midigpt/generated/close", track_id, bar_index)

        feats = compute_bar_features(events, ts_num, ts_den, result_resolution)
        if feats:
            self._send(
                "/midigpt/generated/features",
                track_id, bar_index,
                float(feats["note_density"]),
                float(feats["mean_pitch"]),
                float(feats["mean_velocity"]),
                int(feats["max_polyphony"]),
                float(feats["mean_duration"]),
            )

        log.info("Sent bar %d: %d notes", bar_index, len(note_ons))

    # ── Server start ──────────────────────────────────────────────────────────

    def serve(self, host: str = "0.0.0.0") -> None:
        disp = self._build_dispatcher()
        server = osc_server.BlockingOSCUDPServer((host, self._listen_port), disp)
        log.info("Listening on %s:%d", host, self._listen_port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("Shutting down")
        finally:
            server.server_close()

    def _build_dispatcher(self) -> dispatcher.Dispatcher:
        d = dispatcher.Dispatcher()
        nra = True

        d.map("/midigpt/session/init",    self.handle_session_init,    needs_reply_address=nra)
        d.map("/midigpt/session/start",   self.handle_session_start,   needs_reply_address=nra)
        d.map("/midigpt/session/stop",    self.handle_session_stop,    needs_reply_address=nra)

        d.map("/midigpt/track/create",    self.handle_track_create,    needs_reply_address=nra)
        d.map("/midigpt/track/remove",    self.handle_track_remove,    needs_reply_address=nra)
        d.map("/midigpt/track/set_ignore",self.handle_track_set_ignore,needs_reply_address=nra)

        d.map("/midigpt/note",            self.handle_note,            needs_reply_address=nra)
        d.map("/midigpt/bar/end",         self.handle_bar_end,         needs_reply_address=nra)

        d.map("/midigpt/param/set",       self.handle_param_set,       needs_reply_address=nra)
        d.map("/midigpt/param/set_once",  self.handle_param_set_once,  needs_reply_address=nra)
        d.map("/midigpt/param/reset",     self.handle_param_reset,     needs_reply_address=nra)
        d.map("/midigpt/param/reset_all", self.handle_param_reset_all, needs_reply_address=nra)
        d.map("/midigpt/track/param/set", self.handle_track_param_set, needs_reply_address=nra)

        d.set_default_handler(self._handle_unknown)
        return d

    def _handle_unknown(self, address, *args) -> None:
        log.debug("Unhandled OSC: %s %s", address, args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_param(name: str, value):
    bool_params = {"mask_gap", "adapt_buffer"}
    int_params  = {"lookahead_bars", "buffer_bars", "num_anticipated_bars",
                   "model_dim", "sampling_seed", "max_attempts"}
    if name in bool_params:
        return bool(value)
    if name in int_params:
        return int(value)
    return float(value)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI-GPT real-time OSC server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True,
                   help="Packed .pt bundle (use InferenceEngine.from_checkpoint)")
    p.add_argument("--port", type=int, default=7400,
                   help="UDP port to listen on (Max → Server)")
    p.add_argument("--host", default="0.0.0.0",
                   help="Host/IP to bind")
    p.add_argument("--max_attempts", type=int, default=3,
                   help="Max retry attempts per inference call")
    p.add_argument("--buffer", type=int, default=None,
                   help="Override buffer_bars parameter at startup")
    p.add_argument("--lookahead", type=int, default=None,
                   help="Override lookahead_bars parameter at startup")
    p.add_argument("--model_dim", type=int, default=None,
                   help="Override model_dim parameter at startup")
    p.add_argument("--gen_timeout", type=float, default=None,
                   help="Inference timeout in seconds; 0 = disabled (also settable via OSC)")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = MidiGPTServer(
        ckpt=args.ckpt,
        listen_port=args.port,
        max_attempts=args.max_attempts,
    )

    if args.buffer is not None:
        server._params["buffer_bars"] = args.buffer
    if args.lookahead is not None:
        server._params["lookahead_bars"] = args.lookahead
    if args.model_dim is not None:
        server._params["model_dim"] = args.model_dim
    if args.gen_timeout is not None:
        server._params["gen_timeout"] = args.gen_timeout

    server.serve(host=args.host)


if __name__ == "__main__":
    main()

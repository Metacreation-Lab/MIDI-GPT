"""
MIDI-GPT OSC Studio — Flask + flask-socketio WebSocket bridge.

Translates between browser WebSocket (socket.io) and the MidiGPT OSC server.
Also serves the static frontend.

Usage:
    python -m midigpt.osc.studio.app --ckpt models/oracle_bundle.pt
    # Then open http://localhost:5000 in a browser.
    # The OSC server must be running separately on --osc-host/--osc-port.
"""
from __future__ import annotations
import argparse
import logging
import threading

log = logging.getLogger(__name__)

try:
    import eventlet
    eventlet.monkey_patch()
    from flask import Flask, send_from_directory
    from flask_socketio import SocketIO, emit
    from pythonosc import dispatcher as osc_dispatcher, udp_client
    from pythonosc.osc_server import BlockingOSCUDPServer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Studio requires the [realtime] extra: pip install midigpt[realtime]\n({exc})"
    ) from None

import pathlib

STATIC_DIR = pathlib.Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["SECRET_KEY"] = "midigpt-studio"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Global state — set by main() before serve
_osc_client: udp_client.SimpleUDPClient | None = None
_listen_port: int = 7401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_osc(address: str, *args) -> None:
    if _osc_client is None:
        log.warning("OSC client not initialised — dropping %s", address)
        return
    try:
        from pythonosc.osc_message_builder import OscMessageBuilder
        builder = OscMessageBuilder(address=address)
        for a in args:
            if isinstance(a, bool):
                builder.add_arg(a, "T" if a else "F")
            elif isinstance(a, int):
                builder.add_arg(a, "i")
            elif isinstance(a, float):
                builder.add_arg(a, "f")
            else:
                builder.add_arg(str(a), "s")
        _osc_client.send(builder.build())
    except Exception as exc:
        log.warning("OSC send failed (%s): %s", address, exc)


def _relay_to_browser(address: str, *args) -> None:
    """Forward an OSC message received from the server to all browser clients."""
    sio.emit("osc:in", {"address": address, "args": list(args)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ---------------------------------------------------------------------------
# SocketIO — browser → OSC server
# ---------------------------------------------------------------------------

@sio.on("connect")
def on_connect():
    log.info("Browser connected: %s", sio.sid if hasattr(sio, "sid") else "?")
    emit("bridge:connected")


@sio.on("disconnect")
def on_disconnect():
    log.info("Browser disconnected")


@sio.on("osc:out")
def on_osc_out(data: dict):
    """Generic OSC forward: {address: str, args: list}."""
    address = data.get("address", "")
    args    = data.get("args", [])
    log.debug("osc:out %s %s", address, args)
    _send_osc(address, *args)


# Convenience typed events so the JS side doesn't have to build raw OSC dicts.

@sio.on("session:init")
def on_session_init(data):
    _send_osc("/midigpt/session/init", str(data.get("name", "studio")))

@sio.on("session:start")
def on_session_start(_data=None):
    _send_osc("/midigpt/session/start")

@sio.on("session:stop")
def on_session_stop(_data=None):
    _send_osc("/midigpt/session/stop")

@sio.on("track:create")
def on_track_create(data):
    _send_osc("/midigpt/track/create",
              int(data["track_id"]),
              int(data["instrument"]),
              int(data["track_type"]),
              int(data["is_agent"]))

@sio.on("track:remove")
def on_track_remove(data):
    _send_osc("/midigpt/track/remove", int(data["track_id"]))

@sio.on("note")
def on_note(data):
    _send_osc("/midigpt/note",
              int(data["track_id"]),
              int(data["pitch"]),
              int(data["velocity"]),
              float(data["onset"]),
              float(data["duration"]),
              int(data["bar_index"]))

@sio.on("bar:end")
def on_bar_end(data):
    _send_osc("/midigpt/bar/end",
              int(data["bar_index"]),
              int(data["ts_num"]),
              int(data["ts_den"]))

@sio.on("param:set")
def on_param_set(data):
    _send_osc("/midigpt/param/set", str(data["name"]), data["value"])

@sio.on("param:set_once")
def on_param_set_once(data):
    _send_osc("/midigpt/param/set_once", str(data["name"]), data["value"])

@sio.on("param:reset")
def on_param_reset(data):
    _send_osc("/midigpt/param/reset", str(data["name"]))


# ---------------------------------------------------------------------------
# OSC listener — OSC server → browser
# ---------------------------------------------------------------------------

def _build_osc_dispatcher() -> osc_dispatcher.Dispatcher:
    d = osc_dispatcher.Dispatcher()

    def _fwd(address, *args):
        _relay_to_browser(address, *args)

    for addr in [
        "/midigpt/session/ready",
        "/midigpt/session/started",
        "/midigpt/session/stopped",
        "/midigpt/generated/open",
        "/midigpt/generated/note",
        "/midigpt/generated/close",
        "/midigpt/generated/features",
        "/midigpt/status",
        "/midigpt/error",
    ]:
        d.map(addr, _fwd)

    d.set_default_handler(_fwd)
    return d


def _run_osc_listener(host: str, port: int) -> None:
    disp = _build_osc_dispatcher()
    try:
        server = BlockingOSCUDPServer((host, port), disp)
        log.info("OSC listener on %s:%d", host, port)
        server.serve_forever()
    except Exception as exc:
        log.error("OSC listener failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI-GPT OSC Studio (WebSocket bridge + UI server)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000, help="Web server port")
    p.add_argument("--osc-host", default="127.0.0.1", help="OSC server host")
    p.add_argument("--osc-port", type=int, default=7400, help="OSC server send port")
    p.add_argument("--listen-port", type=int, default=7401,
                   help="Port to receive OSC replies on")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    global _osc_client, _listen_port

    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _osc_client   = udp_client.SimpleUDPClient(args.osc_host, args.osc_port)
    _listen_port  = args.listen_port

    t = threading.Thread(
        target=_run_osc_listener,
        args=("0.0.0.0", args.listen_port),
        daemon=True,
        name="osc-listener",
    )
    t.start()

    log.info("Studio UI → http://%s:%d", args.host if args.host != "0.0.0.0" else "localhost", args.port)
    sio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

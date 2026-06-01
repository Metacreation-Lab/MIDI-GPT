"""
midigpt.osc_listen

Minimal OSC listener — prints every incoming message to stdout.
No model or checkpoint required.

Usage:
    midigpt-listen [--port 7400] [--host 0.0.0.0]
"""

import argparse
import logging

try:
    from pythonosc import dispatcher, osc_server
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "python-osc is required for the OSC listener. "
        "Install it with: pip install midigpt[realtime]"
    ) from None

log = logging.getLogger(__name__)


def _handle_any(address, *args) -> None:
    parts = [repr(a) for a in args]
    print(f"{address}  {' '.join(parts)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print all incoming MIDI-GPT OSC messages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=7400)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--log_level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s"
    )

    disp = dispatcher.Dispatcher()
    disp.set_default_handler(_handle_any)

    server = osc_server.BlockingOSCUDPServer((args.host, args.port), disp)
    print(f"Listening on {args.host}:{args.port}  (Ctrl-C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

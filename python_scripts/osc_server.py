"""Thin wrapper — canonical source is midigpt.osc_server.

Install the package first:
    pip install -e ".[osc]"

Then either run:
    midigpt-server --ckpt model.pt          # installed CLI
    python python_scripts/osc_server.py --ckpt model.pt  # this script
"""
from midigpt.osc_server import main

if __name__ == "__main__":
    main()

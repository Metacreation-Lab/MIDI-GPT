import pytest
from midigpt_refactor.server.osc_server import MidiGPTServer

def test_server_init():
    class DummyEngine:
        pass
        
    server = MidiGPTServer(DummyEngine(), listen_port=7500, max_attempts=5)
    assert server._listen_port == 7500
    assert server._max_attempts == 5
    assert server._state == "UNINITIALIZED"

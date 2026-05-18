import time
import random
from pythonosc import udp_client, dispatcher, osc_server

# --- CONFIGURATION ---
MAX_IP = "127.0.0.1"
MAX_RECV_PORT = 7401     # Python sends notes HERE (Max's udpreceive)
PYTHON_RECV_PORT = 7400  # Python listens HERE (Max's udpsend)
LOOKAHEAD = 4            # How many bars ahead to generate

# --- STATE VARIABLES ---
client = udp_client.SimpleUDPClient(MAX_IP, MAX_RECV_PORT)
is_playing = False

def send_bar(bar_index):
    """Blasts Max with a randomized wall of MIDI data to stress-test the parser."""
    print(f"  -> [STRESS TEST] Generating Bar {bar_index}...")
    
    client.send_message("/midigpt/start", [])
    
    notes = []
    
    # 1. Generate 15 to 30 random notes per bar (High Density)
    num_notes = random.randint(15, 30) 
    
    for _ in range(num_notes):
        track = 1
        # Random pitch between C3 (48) and C6 (84)
        pitch = random.randint(48, 84) 
        velocity = random.randint(40, 127)
        
        # 2. Force Polyphony: We quantize the onsets to 16th notes.
        # Because we have ~20 notes and only 16 slots, multiple notes WILL
        # share the exact same onset phase, creating massive chords!
        onset = random.choice([x * 0.0625 for x in range(16)])
        
        # 3. Random Durations: From 16th-note staccato blips to half-bar drones
        duration = random.uniform(0.0625, 0.5)
        
        # Keep durations strictly within the current bar to prevent bleed-over errors
        if onset + duration > 1.0:
            duration = 1.0 - onset
            
        notes.append([track, pitch, velocity, onset, duration, bar_index])
        
    # 4. The "Nasty Legato" Test
    # We forcefully inject three identical pitches back-to-back to guarantee 
    # your gap-hack math in Max is working correctly.
    notes.append([1, 40, 120, 0.00, 0.25, bar_index]) # Low E
    notes.append([1, 40, 120, 0.25, 0.25, bar_index]) # Low E
    notes.append([1, 40, 120, 0.50, 0.25, bar_index]) # Low E
    
    # 5. Blast them over the network sequentially
    for note in notes:
        client.send_message("/midigpt/note", note)
        time.sleep(0.001) # Micro-pause for network stability
        
    client.send_message("/midigpt/close", [])
    print(f"     Sent {len(notes)} notes successfully.")

# --- OSC HANDLERS (Reacting to Max) ---

def on_session_start(address, *args):
    global is_playing
    print("\n[AI] Received /midigpt/session/start. Waking up!")
    is_playing = True
    
    # Pre-fill the buffer with the initial lookahead bars
    for i in range(1, LOOKAHEAD + 1):
        send_bar(i)
    print(f"[AI] Pre-filled bars 1 to {LOOKAHEAD}. Waiting for Max playback...")

def on_bar_end(address, *args):
    global is_playing
    if not is_playing or not args:
        return
        
    completed_bar = args[0]
    print(f"\n[AI] Max reports Bar {completed_bar} just finished playing.")
    
    # Calculate the next future bar to generate
    next_bar_to_send = completed_bar + 1 + LOOKAHEAD
    send_bar(next_bar_to_send)

def on_session_stop(address, *args):
    global is_playing
    print("\n[AI] Received /midigpt/session/stop. Halting generation.")
    is_playing = False

# --- MAIN SERVER LOOP ---
if __name__ == "__main__":
    # Map the OSC addresses to our functions
    disp = dispatcher.Dispatcher()
    disp.map("/midigpt/session/start", on_session_start)
    disp.map("/midigpt/bar/end", on_bar_end)
    disp.map("/midigpt/session/stop", on_session_stop)

    # Start the server
    server = osc_server.ThreadingOSCUDPServer((MAX_IP, PYTHON_RECV_PORT), disp)
    print(f"MidiGPT Simulator active.")
    print(f"Listening for Max on port {PYTHON_RECV_PORT}...")
    print(f"Sending notes to Max on port {MAX_RECV_PORT}...\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulator shut down.")
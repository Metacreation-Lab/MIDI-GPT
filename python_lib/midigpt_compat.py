#!/usr/bin/env python3
"""
MIDI-GPT Compatibility Wrapper

Provides backward compatibility by converting between protobuf and legacy JSON formats.
This wrapper makes the refactored library behave like the Python 3.8 version.

Usage:
    Instead of:  import midigpt
    Use:         from midigpt_compat import midigpt

Your existing code will work unchanged.
"""

import midigpt as _midigpt_core
import json
import tempfile
import os

class ExpressiveEncoderCompat:
    """Backward compatible wrapper for ExpressiveEncoder"""
    
    def __init__(self):
        self._encoder = _midigpt_core.ExpressiveEncoder()
    
    def midi_to_json(self, filepath):
        """Returns legacy message array format (not protobuf format)"""
        # Get protobuf format from core library
        protobuf_json = self._encoder.midi_to_json(filepath)
        
        # Convert to legacy format
        return self._protobuf_to_legacy(protobuf_json)
    
    def midi_to_json_protobuf(self, filepath):
        """Returns new protobuf format explicitly"""
        return self._encoder.midi_to_json(filepath)
    
    def json_to_midi(self, json_string, filepath):
        """Accepts both legacy and protobuf formats"""
        # Detect format
        try:
            data = json.loads(json_string)
            if isinstance(data, list):
                # Legacy format - convert to protobuf first
                protobuf_json = self._legacy_to_protobuf(json_string)
                self._encoder.json_to_midi(protobuf_json, filepath)
            else:
                # Protobuf format - use directly
                self._encoder.json_to_midi(json_string, filepath)
        except Exception as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def midi_to_tokens(self, filepath):
        """Tokenize MIDI file"""
        return self._encoder.midi_to_tokens(filepath)
    
    def json_to_tokens(self, json_string):
        """Convert JSON to tokens"""
        return self._encoder.json_to_tokens(json_string)
    
    def tokens_to_json(self, tokens):
        """Convert tokens to JSON"""
        return self._encoder.tokens_to_json(tokens)
    
    def tokens_to_midi(self, tokens, filepath):
        """Convert tokens to MIDI file"""
        return self._encoder.tokens_to_midi(tokens, filepath)
    
    def pretty(self, token):
        """Get human-readable token representation"""
        return self._encoder.pretty(token)
    
    def vocab_size(self):
        """Get vocabulary size"""
        return self._encoder.vocab_size()
    
    def encode(self, piece):
        """Encode piece to tokens"""
        return self._encoder.encode(piece)
    
    def decode(self, tokens, piece):
        """Decode tokens to piece"""
        return self._encoder.decode(tokens, piece)
    
    @property
    def config(self):
        """Get encoder config"""
        return self._encoder.config
    
    @property
    def rep(self):
        """Get encoder representation"""
        return self._encoder.rep
    
    def _protobuf_to_legacy(self, protobuf_json):
        """Convert protobuf JSON to legacy message array format"""
        try:
            data = json.loads(protobuf_json)
            messages = []
            
            # Extract tracks
            tracks = data.get('tracks', [])
            
            for track_idx, track in enumerate(tracks):
                # Add TRACK_HEADER
                track_header = {
                    "msg_type": "TRACK_HEADER",
                    "track_id": track_idx,
                    "instrument": track.get("instrument", 0),
                    "track_type": track.get("trackType", "STANDARD_TRACK"),
                    "num_bars": len(track.get("bars", []))
                }
                
                # Add track features if available
                if "internalFeatures" in track and track["internalFeatures"]:
                    features = track["internalFeatures"][0]
                    track_header.update({
                        "density": features.get("onsetDensity", 5),
                        "min_polyphony_q": features.get("minPolyphonyQ", 1),
                        "max_polyphony_q": features.get("maxPolyphonyQ", 4),
                        "min_note_duration_q": features.get("minNoteDurationQ", 1),
                        "max_note_duration_q": features.get("maxNoteDurationQ", 8)
                    })
                else:
                    # Default values
                    track_header.update({
                        "density": 5,
                        "min_polyphony_q": 1,
                        "max_polyphony_q": 4,
                        "min_note_duration_q": 1,
                        "max_note_duration_q": 8
                    })
                
                messages.append(track_header)
                
                # Process bars
                for bar_idx, bar in enumerate(track.get("bars", [])):
                    messages.append({
                        "msg_type": "BAR",
                        "bar_id": bar_idx
                    })
                    
                    # Get events for this bar
                    bar_events = []
                    events = data.get('events', [])
                    
                    for event_idx in bar.get("events", []):
                        if event_idx < len(events):
                            event = events[event_idx]
                            bar_events.append({
                                "time": event.get("time", 0),
                                "pitch": event.get("pitch", 60),
                                "velocity": event.get("velocity", 80)
                            })
                    
                    # Sort events by time
                    bar_events.sort(key=lambda x: x["time"])
                    
                    # Convert to NOTE_ON/NOTE_OFF with TIME_DELTA
                    current_time = 0
                    for event in bar_events:
                        event_time = event["time"]
                        
                        # Add time delta if needed
                        if event_time > current_time:
                            messages.append({
                                "msg_type": "TIME_DELTA",
                                "delta": event_time - current_time
                            })
                            current_time = event_time
                        
                        # Add note event
                        if event["velocity"] > 0:
                            messages.append({
                                "msg_type": "NOTE_ON",
                                "pitch": event["pitch"],
                                "velocity": event["velocity"]
                            })
                        else:
                            messages.append({
                                "msg_type": "NOTE_OFF",
                                "pitch": event["pitch"]
                            })
            
            # Add END_OF_SONG
            messages.append({"msg_type": "END_OF_SONG"})
            
            return json.dumps(messages)
            
        except Exception as e:
            raise ValueError(f"Failed to convert protobuf to legacy format: {e}")
    
    def _legacy_to_protobuf(self, legacy_json):
        """Convert legacy message array to protobuf format"""
        try:
            messages = json.loads(legacy_json)
            
            # Build protobuf structure
            piece = {
                "tracks": [],
                "events": [],
                "resolution": 12,
                "tempo": 120,
                "internalTicksPerQuarter": 480,
                "internalHasTimeSignatures": False
            }
            
            current_track = None
            current_bar = None
            current_time = 0
            event_idx = 0
            
            for msg in messages:
                msg_type = msg.get("msg_type", "")
                
                if msg_type == "TRACK_HEADER":
                    current_track = {
                        "bars": [],
                        "instrument": msg.get("instrument", 0),
                        "trackType": "STANDARD_TRACK",
                        "internalFeatures": []
                    }
                    piece["tracks"].append(current_track)
                    current_time = 0
                
                elif msg_type == "BAR" and current_track is not None:
                    current_bar = {
                        "events": [],
                        "internalBeatLength": 48  # Default
                    }
                    current_track["bars"].append(current_bar)
                    current_time = 0
                
                elif msg_type == "TIME_DELTA":
                    current_time += msg.get("delta", 0)
                
                elif msg_type in ["NOTE_ON", "NOTE_OFF"] and current_bar is not None:
                    event = {
                        "time": current_time,
                        "pitch": msg.get("pitch", 60),
                        "velocity": msg.get("velocity", 80) if msg_type == "NOTE_ON" else 0,
                        "internalDuration": 0,
                        "delta": 0
                    }
                    piece["events"].append(event)
                    current_bar["events"].append(event_idx)
                    event_idx += 1
            
            return json.dumps(piece)
            
        except Exception as e:
            raise ValueError(f"Failed to convert legacy to protobuf format: {e}")


# Create compatibility module interface
class MidiGPTCompat:
    """Compatibility module that mimics the original midigpt interface"""
    
    # Expose the compatible encoder
    ExpressiveEncoder = ExpressiveEncoderCompat
    
    # Expose other classes and functions from core module
    CallbackManager = _midigpt_core.CallbackManager
    sample_multi_step = _midigpt_core.sample_multi_step
    version = _midigpt_core.version
    
    # Expose any other attributes from core module
    def __getattr__(self, name):
        return getattr(_midigpt_core, name)


# Create the midigpt compatibility instance
midigpt = MidiGPTCompat()

# For direct imports
__all__ = ['midigpt', 'ExpressiveEncoderCompat']

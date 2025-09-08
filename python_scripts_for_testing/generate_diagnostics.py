#!/usr/bin/env python3
"""
MIDI-GPT Diagnostic Data Generator
Generates comprehensive diagnostic output for compatibility testing
Run this script in both the original and refactored environments
"""

import sys
import os
import json
import traceback
import hashlib
from pathlib import Path
import argparse
from datetime import datetime

def safe_execute(func, *args, **kwargs):
    """Execute function safely and return result or error info"""
    try:
        result = func(*args, **kwargs)
        return {"success": True, "data": result, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e), "traceback": traceback.format_exc()}

def generate_diagnostics(midi_files, checkpoint_path=None, output_file="diagnostics.json"):
    """Generate comprehensive diagnostic data"""
    
    print(f"Generating diagnostics for MIDI-GPT...")
    print(f"Python version: {sys.version}")
    print(f"Output file: {output_file}")
    
    diagnostics = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "script_version": "1.0"
        },
        "library_info": {},
        "api_tests": {},
        "midi_file_tests": {},
        "encoder_tests": {},
        "sampling_tests": {}
    }
    
    # Import library
    try:
        import midigpt
        print("✅ Successfully imported midigpt")
        diagnostics["library_info"]["import_success"] = True
    except ImportError as e:
        print(f"❌ Failed to import midigpt: {e}")
        diagnostics["library_info"]["import_success"] = False
        diagnostics["library_info"]["import_error"] = str(e)
        # Save what we can and exit
        with open(output_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        return False
    
    # Test basic library info
    print("\n=== Testing Library Info ===")
    
    # Version
    version_result = safe_execute(midigpt.version)
    if version_result["success"]:
        print(f"✅ Version: {version_result['data']}")
    else:
        print(f"❌ Version failed: {version_result['error']}")
    diagnostics["library_info"]["version"] = version_result
    
    # Available functions/classes
    available_attrs = dir(midigpt)
    important_attrs = [
        'ExpressiveEncoder', 'CallbackManager', 'sample_multi_step',
        'getEncoderType', 'getEncoder', 'select_random_segment',
        'status_from_piece', 'default_sample_param', 'version'
    ]
    
    attr_tests = {}
    for attr in important_attrs:
        attr_tests[attr] = attr in available_attrs
        status = "✅" if attr_tests[attr] else "❌"
        print(f"{status} {attr}")
    
    diagnostics["api_tests"]["available_attributes"] = attr_tests
    diagnostics["api_tests"]["all_attributes"] = available_attrs
    
    # Test ExpressiveEncoder instantiation
    print("\n=== Testing ExpressiveEncoder ===")
    encoder_creation = safe_execute(midigpt.ExpressiveEncoder)
    if encoder_creation["success"]:
        print("✅ ExpressiveEncoder created successfully")
        encoder = encoder_creation["data"]
        
        # Test encoder methods
        encoder_methods = dir(encoder)
        important_methods = ['midi_to_json', 'json_to_midi', 'midi_to_tokens', 'encode', 'decode']
        method_tests = {}
        for method in important_methods:
            method_tests[method] = method in encoder_methods
            status = "✅" if method_tests[method] else "❌"
            print(f"{status} {method}")
        
        diagnostics["encoder_tests"]["creation"] = encoder_creation
        diagnostics["encoder_tests"]["available_methods"] = method_tests
        diagnostics["encoder_tests"]["all_methods"] = encoder_methods
    else:
        print(f"❌ ExpressiveEncoder creation failed: {encoder_creation['error']}")
        diagnostics["encoder_tests"]["creation"] = encoder_creation
    
    # Test utility functions
    print("\n=== Testing Utility Functions ===")
    
    # Default sample param
    default_param_result = safe_execute(midigpt.default_sample_param)
    if default_param_result["success"]:
        print("✅ default_sample_param() works")
        try:
            param_data = json.loads(default_param_result["data"])
            print(f"   Sample keys: {list(param_data.keys())[:5]}")
        except:
            print("   (Result not valid JSON)")
    else:
        print(f"❌ default_sample_param() failed: {default_param_result['error']}")
    diagnostics["api_tests"]["default_sample_param"] = default_param_result
    
    # CallbackManager
    callback_result = safe_execute(midigpt.CallbackManager)
    if callback_result["success"]:
        print("✅ CallbackManager() works")
    else:
        print(f"❌ CallbackManager() failed: {callback_result['error']}")
    diagnostics["api_tests"]["callback_manager"] = callback_result
    
    # Process MIDI files
    if midi_files and encoder_creation["success"]:
        print(f"\n=== Testing MIDI Files ===")
        encoder = encoder_creation["data"]
        
        for i, midi_file in enumerate(midi_files):
            if not os.path.exists(midi_file):
                print(f"❌ File not found: {midi_file}")
                continue
                
            print(f"\n--- File {i+1}: {os.path.basename(midi_file)} ---")
            file_key = f"file_{i+1}_{os.path.basename(midi_file)}"
            file_tests = {}
            
            # File info
            file_size = os.path.getsize(midi_file)
            with open(midi_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            file_tests["file_info"] = {
                "path": midi_file,
                "size_bytes": file_size,
                "md5_hash": file_hash
            }
            print(f"   Size: {file_size} bytes, MD5: {file_hash[:8]}...")
            
            # MIDI to JSON
            midi_to_json_result = safe_execute(encoder.midi_to_json, midi_file)
            if midi_to_json_result["success"]:
                json_str = midi_to_json_result["data"]
                json_hash = hashlib.md5(json_str.encode()).hexdigest()
                try:
                    json_data = json.loads(json_str)
                    json_keys = list(json_data.keys()) if isinstance(json_data, dict) else None
                    print(f"✅ midi_to_json: {len(json_str)} chars, MD5: {json_hash[:8]}")
                    if json_keys:
                        print(f"   JSON keys: {json_keys}")
                except:
                    print(f"✅ midi_to_json: {len(json_str)} chars (invalid JSON)")
                
                # Store truncated version to avoid huge files
                midi_to_json_result["data_hash"] = json_hash
                midi_to_json_result["data_length"] = len(json_str)
                if len(json_str) > 1000:
                    midi_to_json_result["data_sample"] = json_str[:500] + "..." + json_str[-500:]
                    midi_to_json_result["data"] = "[TRUNCATED]"
            else:
                print(f"❌ midi_to_json failed: {midi_to_json_result['error']}")
            
            file_tests["midi_to_json"] = midi_to_json_result
            
            # MIDI to tokens
            midi_to_tokens_result = safe_execute(encoder.midi_to_tokens, midi_file)
            if midi_to_tokens_result["success"]:
                tokens = midi_to_tokens_result["data"]
                tokens_hash = hashlib.md5(str(tokens).encode()).hexdigest()
                print(f"✅ midi_to_tokens: {len(tokens)} tokens, MD5: {tokens_hash[:8]}")
                if len(tokens) > 0:
                    print(f"   First 10 tokens: {tokens[:10]}")
                    print(f"   Last 10 tokens: {tokens[-10:]}")
                
                # Store summary to avoid huge arrays
                midi_to_tokens_result["data_hash"] = tokens_hash
                midi_to_tokens_result["data_length"] = len(tokens)
                midi_to_tokens_result["data_sample"] = {
                    "first_10": tokens[:10] if len(tokens) > 10 else tokens,
                    "last_10": tokens[-10:] if len(tokens) > 10 else [],
                    "min_token": min(tokens) if tokens else None,
                    "max_token": max(tokens) if tokens else None
                }
                midi_to_tokens_result["data"] = "[TRUNCATED]"
            else:
                print(f"❌ midi_to_tokens failed: {midi_to_tokens_result['error']}")
            
            file_tests["midi_to_tokens"] = midi_to_tokens_result
            
            diagnostics["midi_file_tests"][file_key] = file_tests
    
    # Test sampling if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path) and encoder_creation["success"] and midi_files:
        print(f"\n=== Testing Sampling with {os.path.basename(checkpoint_path)} ===")
        
        # Use first MIDI file for sampling test
        test_midi = midi_files[0]
        encoder = encoder_creation["data"]
        
        # Get MIDI as JSON
        midi_json_result = safe_execute(encoder.midi_to_json, test_midi)
        if midi_json_result["success"]:
            try:
                midi_data = json.loads(midi_json_result["data"])
                
                # Simple test configuration
                status_config = {
                    'tracks': [{
                        'track_id': 0,
                        'temperature': 0.5,
                        'instrument': 'acoustic_grand_piano',
                        'density': 10,
                        'track_type': 10,
                        'ignore': False,
                        'selected_bars': [False, True, False, False],  # Generate bar 1 only
                        'min_polyphony_q': 'POLYPHONY_ANY',
                        'max_polyphony_q': 'POLYPHONY_ANY',
                        'autoregressive': False,
                        'polyphony_hard_limit': 9
                    }]
                }
                
                param_config = {
                    'tracks_per_step': 1,
                    'bars_per_step': 1,
                    'model_dim': 4,
                    'percentage': 100,
                    'batch_size': 1,
                    'temperature': 1.0,
                    'max_steps': 5,  # Very short test
                    'polyphony_hard_limit': 6,
                    'shuffle': False,
                    'verbose': False,
                    'ckpt': checkpoint_path,
                    'sampling_seed': 42,  # Fixed seed
                    'mask_top_k': 0
                }
                
                piece_json = json.dumps(midi_data)
                status_json = json.dumps(status_config)
                param_json = json.dumps(param_config)
                
                # Test sampling
                callbacks = midigpt.CallbackManager()
                sampling_result = safe_execute(
                    midigpt.sample_multi_step,
                    piece_json, status_json, param_json, 1, callbacks
                )
                
                if sampling_result["success"]:
                    result_tuple = sampling_result["data"]
                    if isinstance(result_tuple, tuple) and len(result_tuple) >= 1:
                        generated_json = result_tuple[0]
                        result_hash = hashlib.md5(generated_json.encode()).hexdigest()
                        print(f"✅ Sampling successful, result MD5: {result_hash[:8]}")
                        
                        # Store hash instead of full data
                        sampling_result["data_hash"] = result_hash
                        sampling_result["data_length"] = len(generated_json)
                        sampling_result["data"] = "[TRUNCATED]"
                    else:
                        print(f"✅ Sampling returned: {type(result_tuple)}")
                else:
                    print(f"❌ Sampling failed: {sampling_result['error']}")
                
                diagnostics["sampling_tests"]["test_run"] = sampling_result
                
            except Exception as e:
                print(f"❌ Sampling setup failed: {e}")
                diagnostics["sampling_tests"]["setup_error"] = str(e)
    
    # Save diagnostics
    print(f"\n=== Saving Diagnostics ===")
    try:
        with open(output_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"✅ Diagnostics saved to: {output_file}")
        
        # Print summary
        print(f"\n=== Summary ===")
        print(f"Library import: {'✅' if diagnostics['library_info']['import_success'] else '❌'}")
        if 'version' in diagnostics['library_info']:
            version_ok = diagnostics['library_info']['version']['success']
            print(f"Version check: {'✅' if version_ok else '❌'}")
        
        encoder_ok = diagnostics['encoder_tests'].get('creation', {}).get('success', False)
        print(f"Encoder creation: {'✅' if encoder_ok else '❌'}")
        
        midi_files_tested = len(diagnostics['midi_file_tests'])
        print(f"MIDI files processed: {midi_files_tested}")
        
        sampling_tested = 'test_run' in diagnostics['sampling_tests']
        print(f"Sampling tested: {'✅' if sampling_tested else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save diagnostics: {e}")
        return False

def create_simple_midi_file(output_path):
    """Create a simple test MIDI file using basic binary format"""
    # Simple MIDI file with one note
    midi_data = bytes([
        # Header chunk
        0x4D, 0x54, 0x68, 0x64,  # "MThd"
        0x00, 0x00, 0x00, 0x06,  # Header length
        0x00, 0x00,              # Format 0
        0x00, 0x01,              # 1 track
        0x00, 0x60,              # 96 ticks per quarter note
        
        # Track chunk
        0x4D, 0x54, 0x72, 0x6B,  # "MTrk"
        0x00, 0x00, 0x00, 0x0B,  # Track length
        
        # Note on C4
        0x00,                    # Delta time 0
        0x90, 0x3C, 0x40,       # Note on, note 60 (C4), velocity 64
        
        # Note off C4
        0x60,                    # Delta time 96
        0x80, 0x3C, 0x40,       # Note off, note 60, velocity 64
        
        # End of track
        0x00, 0xFF, 0x2F, 0x00  # End of track meta event
    ])
    
    try:
        with open(output_path, 'wb') as f:
            f.write(midi_data)
        print(f"Created simple test MIDI file: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to create test MIDI file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI-GPT diagnostic data")
    parser.add_argument("--midi-files", nargs="+", help="MIDI files to test")
    parser.add_argument("--checkpoint", help="Model checkpoint for sampling tests")
    parser.add_argument("--output", default="diagnostics.json", help="Output JSON file")
    parser.add_argument("--create-test-midi", action="store_true", 
                       help="Create a simple test MIDI file")
    
    args = parser.parse_args()
    
    # Handle MIDI files
    midi_files = args.midi_files or []
    
    if args.create_test_midi:
        test_midi_path = "simple_test.mid"
        if create_simple_midi_file(test_midi_path):
            midi_files.append(test_midi_path)
    
    if not midi_files:
        print("Warning: No MIDI files provided. Use --midi-files or --create-test-midi")
    
    # Validate files
    valid_midi_files = []
    for midi_file in midi_files:
        if os.path.exists(midi_file):
            valid_midi_files.append(midi_file)
        else:
            print(f"Warning: MIDI file not found: {midi_file}")
    
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint not found: {args.checkpoint}")
        args.checkpoint = None
    
    # Generate diagnostics
    success = generate_diagnostics(valid_midi_files, args.checkpoint, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Extract protobuf files from libraries/protobuf/src/ to proto/ directory
and create updated include wrapper if needed.
"""

import os
import shutil
from pathlib import Path

def main():
    """Extract protobuf files and create new directory structure."""
    
    # Define source and destination directories
    old_proto_dir = Path("libraries/protobuf/src")
    new_proto_dir = Path("proto")
    old_include_dir = Path("libraries/protobuf/include")
    new_include_dir = Path("include/proto")
    
    # Create new directories if they don't exist
    new_proto_dir.mkdir(exist_ok=True)
    new_include_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting protobuf definitions...")
    
    # List of proto files to extract
    proto_files = [
        "enum.proto",
        "midi.proto", 
        "midi_internal.proto",
        "track_type.proto",
        "feature_extraction.proto"
    ]
    
    # Copy proto files
    for proto_file in proto_files:
        src_file = old_proto_dir / proto_file
        dst_file = new_proto_dir / proto_file
        
        if src_file.exists():
            print(f"  Copying {src_file} -> {dst_file}")
            shutil.copy2(src_file, dst_file)
        else:
            print(f"  Warning: {src_file} not found!")
    
    # Check for include files that need to be preserved
    if old_include_dir.exists():
        include_files = list(old_include_dir.glob("*.h"))
        if include_files:
            print(f"Found {len(include_files)} include files to copy...")
            for include_file in include_files:
                dst_file = new_include_dir / include_file.name
                print(f"  Copying {include_file} -> {dst_file}")
                shutil.copy2(include_file, dst_file)
    
    # Create a proto_library.h wrapper for easier includes
    proto_wrapper_content = '''#pragma once

// Generated protobuf headers
#include "enum.pb.h"
#include "midi.pb.h"
#include "midi_internal.pb.h"
#include "track_type.pb.h"
#include "feature_extraction.pb.h"

// Convenience namespace alias
namespace midi_proto = midi;
'''
    
    wrapper_file = new_include_dir / "proto_library.h"
    print(f"Creating protobuf wrapper: {wrapper_file}")
    with open(wrapper_file, 'w') as f:
        f.write(proto_wrapper_content)
    
    print("\nProtobuf extraction complete!")
    print("Next steps:")
    print("1. Update C++ source files to use new include paths")
    print("2. Remove libraries/protobuf directory")
    print("3. Update CMakeLists.txt to use new proto directory")

if __name__ == "__main__":
    main()

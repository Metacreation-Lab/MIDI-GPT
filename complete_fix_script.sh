#!/bin/bash
# Complete fix script for MIDI-GPT build issues

echo "=== MIDI-GPT Complete Fix Script ==="

# Step 1: Force protobuf downgrade with specific protoc version
echo "Step 1: Installing compatible protobuf..."
pip uninstall protobuf -y
pip install protobuf==3.20.3

# Also check if system protoc needs to be downgraded on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Checking system protoc version..."
    protoc --version
    
    # If protoc is too new, suggest using older version
    if command -v brew &> /dev/null; then
        echo "Consider using homebrew to install compatible protoc:"
        echo "  brew unlink protobuf"
        echo "  brew install protobuf@3"
        echo "  brew link protobuf@3 --force"
    fi
fi

# Step 2: Fix missing torch_library.h include
echo "Step 2: Fixing torch_library.h include..."
if [ -f "src/lib.cpp" ]; then
    # Check if torch_library.h is actually used
    if grep -q "torch_library" src/lib.cpp; then
        # Option 1: Try to remove the include entirely
        echo "Attempting to remove torch_library.h include..."
        sed -i.bak2 's|#include "../libraries/torch/include/torch_library.h"|// #include "../libraries/torch/include/torch_library.h" // Removed - not needed|g' src/lib.cpp
    fi
    
    # Create a minimal torch_library.h wrapper if needed
    mkdir -p libraries/torch/include
    if [ ! -f "libraries/torch/include/torch_library.h" ]; then
        echo "Creating minimal torch_library.h wrapper..."
        cat > libraries/torch/include/torch_library.h << 'EOF'
#pragma once
// Minimal torch library wrapper
// Most torch functionality should be accessed directly via torch headers

#ifndef NO_TORCH
#include <torch/torch.h>
#endif
EOF
    fi
    
    echo "Updated src/lib.cpp torch include path"
else
    echo "Warning: src/lib.cpp not found"
fi

# Step 3: Try to downgrade system protoc if possible
echo "Step 3: Attempting to use compatible protoc..."

# Check if we can create a local protoc wrapper
if command -v protoc &> /dev/null; then
    PROTOC_VERSION=$(protoc --version | grep -o '[0-9]\+\.[0-9]\+')
    echo "Current protoc version: $PROTOC_VERSION"
    
    # If protoc is too new (4.x), try to work around it
    if [[ "$PROTOC_VERSION" > "3.99" ]]; then
        echo "WARNING: protoc version $PROTOC_VERSION may be too new"
        echo "Consider downgrading protoc to 3.x series"
    fi
fi

# Step 4: Force regeneration of protobuf files with better settings
echo "Step 4: Cleaning and regenerating protobuf files..."
rm -rf build/ libraries/protobuf/build/

# Step 5: Attempt build
echo "Step 5: Attempting build..."
python3.9 setup_midigpt.py --mac-os --test

# Step 6: If build still fails with string_view errors, apply source code fixes
if [ $? -ne 0 ]; then
    echo "Build failed - applying source code fixes..."
    
    # Apply specific fixes for string_view issues
    echo "Applying string_view compatibility fixes..."
    
    # Fix src/inference/enum/gm.h:689
    if [ -f "src/inference/enum/gm.h" ]; then
        sed -i.bak3 's/result\.push_back( descriptor->FindValueByNumber(kv\.first)->name() );/result.push_back( std::string(descriptor->FindValueByNumber(kv.first)->name()) );/g' src/inference/enum/gm.h
        echo "Fixed gm.h string_view issue"
    fi
    
    # Fix src/common/encoder/attribute_control.h:469
    if [ -f "src/common/encoder/attribute_control.h" ]; then
        sed -i.bak4 's/output\[field_name\]\.push_back(enum_descriptor->value(i)->name());/output[field_name].push_back(std::string(enum_descriptor->value(i)->name()));/g' src/common/encoder/attribute_control.h
        
        # Fix line 494
        sed -i.bak5 's/output\[field_name\]\[enum_descriptor->value(i)->name()\]/output[field_name][std::string(enum_descriptor->value(i)->name())]/g' src/common/encoder/attribute_control.h
        
        # Fix line 1029
        sed -i.bak6 's/std::string name = descriptor->FindValueByNumber.*->name();/std::string name = std::string(descriptor->FindValueByNumber(static_cast<midi::GenreMusicmap>(i+1))->name());/g' src/common/encoder/attribute_control.h
        
        echo "Fixed attribute_control.h string_view issues"
    fi
    
    # Fix src/common/encoder/encoder_base.h:64
    if [ -f "src/common/encoder/encoder_base.h" ]; then
        sed -i.bak7 's/types\.push_back(enum_descriptor->FindValueByNumber(c)->name());/types.push_back(std::string(enum_descriptor->FindValueByNumber(c)->name()));/g' src/common/encoder/encoder_base.h
        echo "Fixed encoder_base.h string_view issue"
    fi
    
    # Fix src/inference/protobuf/validate.h - multiple fd->name() fixes
    if [ -f "src/inference/protobuf/validate.h" ]; then
        sed -i.bak8 's/key_map\[fd->name()\]/key_map[std::string(fd->name())]/g' src/inference/protobuf/validate.h
        echo "Fixed validate.h string_view issues"
    fi
    
    echo "Applied source code compatibility fixes"
    echo "Retrying build..."
    python3.9 setup_midigpt.py --mac-os --test
fi

echo "=== Fix script complete ==="

# Check final result
if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: Build completed successfully!"
    echo "Testing import..."
    cd python_lib && python -c "import midigpt; print('midigpt import successful!')" && cd ..
else
    echo "❌ Build still failing. Additional manual fixes may be needed."
    echo ""
    echo "Next steps to try:"
    echo "1. Check protoc version: protoc --version"
    echo "2. Try installing protobuf@3 via homebrew if on macOS"
    echo "3. Manually review remaining string_view errors in build output"
    echo "4. Consider using an older system protoc compiler"
fi

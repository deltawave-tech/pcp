#!/bin/bash

# Quick build verification script
echo "🌙 Verifying PCP distributed training build..."

# Check if we're in the right directory
if [ ! -f "build.zig" ]; then
    echo "💣 Not in PCP project root. Navigate to the directory containing build.zig"
    exit 1
fi

# Build the project
echo "⛓️ Building..."
zig build

if [ $? -ne 0 ]; then
    echo "💣 Build failed. Check compilation errors above."
    exit 1
fi

# Verify the main_distributed executable exists
if [ ! -f "./zig-out/bin/main_distributed" ]; then
    echo "💣 main_distributed executable not found. Check build.zig configuration."
    exit 1
fi

# Test command-line parsing
echo "🐉 Testing command-line argument parsing..."
./zig-out/bin/main_distributed --help 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🌙 Help functionality works"
else
    echo "☣️  Help option not recognized (this may be expected)"
fi

echo ""
echo "🌑 Build verification complete!"
echo "   Executable: ./zig-out/bin/main_distributed"
echo "   Size: $(ls -lh ./zig-out/bin/main_distributed | awk '{print $5}')"
echo ""
echo "Ready to run distributed training test:"
echo "   ./test_distributed.sh"
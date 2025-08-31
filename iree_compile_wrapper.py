#!/usr/bin/env python3
"""
IREE Compiler Wrapper for PCP Project
This script provides a command-line interface to IREE compiler
that can be called from Zig subprocess execution.
"""

import sys
import os
import tempfile
import glob
from pathlib import Path
from iree.compiler.tools import compile_file

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 iree_compile_wrapper.py <input.mlir> <output.vmfb> <spirv_output_dir> [target_backends]", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] 
    spirv_output_dir = sys.argv[3]
    target_backends = sys.argv[4] if len(sys.argv) > 4 else "vulkan-spirv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory for SPIR-V files
    os.makedirs(spirv_output_dir, exist_ok=True)
    
    try:
        print(f"Compiling {input_file} -> {output_file} with target: {target_backends}")
        print(f"SPIR-V files will be dumped to: {spirv_output_dir}")
        
        # Use IREE compiler with SPIR-V dump enabled
        result = compile_file(
            input_file,
            target_backends=[target_backends],
            output_file=output_file,
            # Dump SPIR-V files to the specified directory using extra_args
            extra_args=[f"--iree-hal-dump-executable-files-to={spirv_output_dir}"]
        )
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✓ Successfully compiled to {output_file} ({file_size} bytes)")
        else:
            print("Error: Output file was not created", file=sys.stderr)
            sys.exit(1)
        
        # List any SPIR-V files that were generated
        spirv_files = glob.glob(os.path.join(spirv_output_dir, "*.spv"))
        if spirv_files:
            print(f"✓ Generated {len(spirv_files)} SPIR-V file(s):")
            for spv_file in spirv_files:
                spv_size = os.path.getsize(spv_file)
                print(f"  - {os.path.basename(spv_file)} ({spv_size} bytes)")
        else:
            print("Warning: No SPIR-V files were generated")
            
    except Exception as e:
        print(f"Error during IREE compilation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
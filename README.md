# PCP - Planetary Compute Protocol

A distributed tensor computation framework written in Zig, designed for training and inference of large neural networks across planetary-scale compute clusters. PCP uses MLIR (Multi-Level Intermediate Representation) for high-performance tensor operations and automatic differentiation.

## Features

- **MLIR-based computation**: Leverages MLIR for tensor operations with StableHLO dialect
- **Vector-Jacobian Product (VJP) autodiff**: Efficient reverse-mode automatic differentiation
- **Pure Zig implementation**: Zero external dependencies except MLIR/LLVM
- **Static tensor shapes**: Compile-time shape validation with Zig's comptime
- **Memory-safe operations**: Proper tensor lifecycle management
- **Cross-platform backend support**:
  - CPU (via MLIR execution)
  - Apple Silicon via Metal (WIP)
  - NVIDIA (planned via MLIR GPU dialect)
  - AMD (planned via MLIR GPU dialect)
- **Distributed training**: Actor model for planetary-scale computation (planned)

## Architecture

PCP transforms high-level tensor operations into optimized MLIR computation graphs:

```
Zig Tensor Ops → StableHLO MLIR → Optimized Execution
     ↓              ↓                    ↓
   VJP Rules → Gradient Graph → Automatic Differentiation
```

### Core Components

- **StableHLO Dialect Wrapper**: Clean Zig interface to MLIR operations
- **MLIRBuilder**: Constructs computation graphs using MLIR C API
- **VJP Autodiff Engine**: Graph-to-graph automatic differentiation
- **Tensor Abstraction**: High-level tensor operations over MLIR values

## Project Structure

```
src/
├── mlir.zig                    # Core MLIR wrapper types
├── mlir/
│   ├── c.zig                   # MLIR C API bindings
│   └── dialects/
│       └── stablehlo.zig       # StableHLO dialect wrapper
├── ops.zig                     # High-level tensor operations
├── autodiff.zig                # VJP-based automatic differentiation
├── tensor.zig                  # Tensor abstraction layer
├── backends/
│   └── metal.zig              # Metal backend for Apple Silicon
├── models/
│   └── gpt2.zig               # GPT-2 transformer implementation
└── examples/
    ├── mlir_test.zig          # MLIR integration tests
    ├── tensor_mlir_test.zig   # Tensor-MLIR bridge tests
    └── gpt2_training.zig      # GPT-2 training example
```

## MLIR Integration

PCP uses MLIR's StableHLO dialect for tensor operations:

```zig
// High-level tensor operation
const result = try ops.add(builder, tensor_a, tensor_b);

// Compiles to StableHLO MLIR:
// %result = stablehlo.add %tensor_a, %tensor_b : tensor<2x2xf32>
```

### Automatic Differentiation

VJP-based reverse-mode autodiff transforms forward graphs to gradient graphs:

```zig
// Forward: z = x * y + w
const xy = try ops.multiply(builder, x, y);
const z = try ops.add(builder, xy, w);

// Autodiff creates gradient graph:
// dz/dx = grad_z * y
// dz/dy = grad_z * x  
// dz/dw = grad_z
```

## Getting Started

### Prerequisites

- **Zig compiler**: 0.14.x or later
- **LLVM/MLIR**: Built with C API bindings enabled
- **For Metal backend**: macOS with Apple Silicon

### Building MLIR Support

PCP requires LLVM/MLIR with C API bindings. The build system automatically detects MLIR installations:

```bash
# macOS with Homebrew (basic support)
brew install llvm

# Or build from source for full MLIR support
git clone --depth=1 --branch=release/20.x https://github.com/llvm/llvm-project.git
mkdir llvm-build && cd llvm-build
cmake -G "Ninja" ../llvm-project/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DMLIR_ENABLE_BINDINGS_C=ON \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm-pcp
ninja && ninja install
```

### Building and Running

```bash
# Build the library
zig build

# Run unit tests
zig build test

# Test MLIR dialect wrappers
zig build test-dialects

# Run MLIR integration tests
zig build run-mlir-test

# Test tensor-MLIR bridge
zig build run-tensor-mlir-test

# Run autodiff tests
zig build run-autodiff-test

# Run GPT-2 training example
zig build run

# Run Metal backend tests (macOS only)
zig build run-metal-test
```

## Memory Management

- **MLIR-managed**: Computation graphs managed by MLIR context
- **Reference counting**: Efficient tensor reuse and cleanup
- **Safe tensor lifecycle**: Proper cleanup of intermediate values
- **Bounds checking**: Overflow protection for numerical stability

## API Example

```zig
const std = @import("std");
const pcp = @import("pcp");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{});
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create MLIR builder
    var builder = try pcp.ops.MLIRBuilder.init(allocator);
    defer builder.deinit();
    
    // Create tensors (shapes known at compile time)
    const shape = &[_]usize{2, 2};
    var a = try pcp.tensor.Tensor.zeros(allocator, shape, .f32, .cpu);
    var b = try pcp.tensor.Tensor.ones(allocator, shape, .f32, .cpu);
    defer a.deinit();
    defer b.deinit();
    
    // Perform operations (compiled to MLIR)
    const c = try pcp.ops.add(&builder, a, b);
    const d = try pcp.ops.multiply(&builder, c, a);
    
    // Automatic differentiation
    var autodiff = pcp.autodiff.AutoDiff.init(allocator, &builder);
    const grad_fn = try autodiff.grad(d.operation);
    
    std.debug.print("Computation graph built successfully!\n", .{});
}
```

## Roadmap

**Core Infrastructure**
- [x] MLIR C API bindings
- [x] StableHLO dialect wrapper
- [x] VJP-based automatic differentiation
- [x] Tensor abstraction over MLIR values
- [x] MLIRBuilder for graph construction

**Operations & Models**
- [x] Basic tensor operations (add, multiply, matmul, ReLU)
- [x] GPT-2 transformer implementation
- [ ] Complete operation set (conv2d, batch_norm, etc.)
- [ ] Pre-trained model loading

**Backends & Optimization**
- [x] CPU execution via MLIR
- [ ] Metal backend completion
- [ ] MLIR optimization passes
- [ ] JIT compilation
- [ ] GPU dialect integration

**Distributed Training**
- [ ] Actor-based compute nodes
- [ ] Gradient aggregation protocols
- [ ] Fault-tolerant training
- [ ] Decentralized optimization algorithms

## Contributing

PCP uses MLIR for high-performance tensor computation. When contributing:

1. **Operations**: Add new ops to `src/mlir/dialects/stablehlo.zig`
2. **VJP Rules**: Update `src/autodiff.zig` with gradient rules
3. **Testing**: Ensure MLIR integration tests pass
4. **Documentation**: Update examples with new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.
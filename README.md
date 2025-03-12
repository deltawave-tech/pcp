# PCP - Planetary Compute Protocol

A distributed tensor computation framework written in Zig, designed for training and inference of large neural networks across planetary-scale compute clusters.

## Features

- Pure Zig implementation with no external dependencies
- Static tensor shapes checked at compile-time with Zig's comptime
- Compile-time operation plan generation for optimized execution
- Automatic gradient calculation with comptime-derived rules
- Memory-safe tensor operations with proper cleanup
- Embedding layers with correct gradient flow
- Reference-counted tensor management
- Cross-platform backend support
  - CPU computing with SIMD optimizations
  - Apple Silicon via Metal (WIP)
  - NVIDIA via CUDA (planned)
  - AMD via ROCm (planned)
- Distributed training using actor model (planned)
- Fault-tolerant decentralized optimization (planned)

## Project Structure

- `src/tensor.zig` - Core tensor implementation
- `src/ops.zig` - Tensor operations with comptime plan generation
- `src/autodiff.zig` - Automatic differentiation with gradient rules
- `src/plan_example.zig` - Example of using the comptime plan system
- `src/backends/` - Hardware-specific backends
  - `metal.zig` - Metal backend for Apple Silicon
- `src/models/` - Neural network model implementations
  - `gpt2.zig` - GPT-2 transformer model
- `src/examples/` - Example applications
  - `gpt2_training.zig` - Training a mini GPT-2 model
  - `shakespeare_training.zig` - Training on Shakespeare text
  - `autodiff_test.zig` - Memory leak tests and benchmarks

## Memory Management

One of PCP's core strengths is its careful memory management:

- All tensors and computational graph nodes track their own memory
- Reference counting for efficient tensor reuse and proper cleanup
- Proper cleanup of intermediate tensors during backward pass
- Robust gradient flow through embedding layers and complex operations
- Safe handling of tensor references throughout the computation graph
- Bounds checking and overflow protection for numerical stability

## Comptime Plan System

PCP uses Zig's compile-time features to optimize tensor operations:

- Operations are defined as comptime-generated Plans
- Plans encapsulate shape information and operation logic
- Backend-specific optimizations can be applied at compile time
- Gradient rules are automatically applied to forward operations
- Error handling is streamlined with errdefer for cleanup
- Plans can be composed to build complex operations

> **Note**: We're migrating from a legacy Node-based computation graph approach to the new comptime Plan-based system. The legacy system is marked as deprecated and will be removed in the future.

## Getting Started

### Prerequisites

- Zig compiler (0.14.x)
- For Metal backend: macOS with an Apple Silicon or compatible GPU

### Zig Version Compatibility

This project is compatible with Zig 0.14.x. The following changes were made to ensure compatibility:

1. Updated all code to use the Zig 0.14 `std.Random` module for random number generation
2. Standardized pointer casting with a helper function:
   ```zig
   fn ptrCastHelper(comptime T: type, ptr: anytype) T {
       // We still need to use alignCast for pointers that require higher alignment
       return @ptrCast(@alignCast(ptr));
   }
   ```

### Building and Running

```bash
# Build the library
zig build

# Run tests
zig build test

# Run benchmark and memory tests
zig build run-autodiff-test

# Run the Shakespeare training example
zig build run-shakespeare

# Run the GPT-2 training example
zig build run-gpt2

# Run the comptime plan examples (recommended)
zig build run-comptime-examples
```

## Performance

The framework is designed with performance in mind:

- Fast tensor operations on the CPU
- Efficient memory usage with proper cleanup of temporary tensors
- Benchmarks show tensor operations complete in milliseconds
- Metal GPU backend (WIP) will further accelerate computations

## Roadmap

- [x] Core tensor implementation
- [x] Basic operations (add, matmul, etc.)
- [x] Autodiff engine with memory management
- [x] Basic model definition (GPT-2)
- [x] Memory leak fixes in backward operations
- [x] Training examples with text generation
- [x] Embedding layers with proper gradient flow
- [x] Robust bounds checking and safety mechanisms
- [x] Comptime Plan-based operation design
- [x] Automatic gradient rule generation with comptime
- [ ] Complete migration from legacy Node-based system to Plan-based approach
- [ ] Update all examples to use the Plan-based API
- [ ] Complete Metal backend implementation
- [ ] Actor-based distributed training
- [ ] Decentralized optimization algorithms
- [ ] CUDA and ROCm backends

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# PCP - Planetary Compute Protocol

A distributed tensor computation framework written in Zig, designed for training and inference of large neural networks across planetary-scale compute clusters.

## Features

- Pure Zig implementation with no external dependencies
- Static tensor shapes checked at compile-time with Zig's comptime
- Automatic differentiation using dynamic computation graphs
- Memory-safe tensor operations with proper cleanup
- Cross-platform backend support
  - CPU computing with SIMD optimizations
  - Apple Silicon via Metal (WIP)
  - NVIDIA via CUDA (planned)
  - AMD via ROCm (planned)
- Distributed training using actor model (planned)
- Fault-tolerant decentralized optimization (planned)

## Project Structure

- `src/tensor.zig` - Core tensor implementation
- `src/ops.zig` - Tensor operations (add, matmul, etc.)
- `src/autodiff.zig` - Automatic differentiation engine
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
- Proper cleanup of intermediate tensors during backward pass
- No memory leaks in the automatic differentiation engine
- Safe handling of tensor references throughout the computation graph

## Getting Started

### Prerequisites

- Zig compiler (0.11.0 or newer)
- For Metal backend: macOS with an Apple Silicon or compatible GPU

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
- [x] Autodiff engine with proper memory management
- [x] Basic model definition (GPT-2)
- [x] Memory leak fixes in backward operations
- [x] Training examples with text generation
- [ ] Complete Metal backend implementation
- [ ] Reference counting system for tensor management
- [ ] Actor-based distributed training
- [ ] Decentralized optimization algorithms
- [ ] CUDA and ROCm backends

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# PCP - Planetary Compute Protocol

A distributed tensor computation framework written in Zig, designed for training and inference of large neural networks across planetary-scale compute clusters.

## Features

- Pure Zig implementation with no external dependencies
- Static tensor shapes checked at compile-time with Zig's comptime
- Automatic differentiation using dynamic computation graphs
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
  - `gpt2_training.zig` - Example of training a small GPT-2 model

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

# Run the GPT-2 training example
zig build run-gpt2
```

## Roadmap

- [x] Core tensor implementation
- [x] Basic operations (add, matmul, etc.)
- [x] Autodiff engine
- [x] Basic model definition (GPT-2)
- [ ] Complete Metal backend implementation
- [ ] Actor-based distributed training
- [ ] Decentralized optimization algorithms
- [ ] CUDA and ROCm backends

## License

This project is licensed under the MIT License - see the LICENSE file for details.
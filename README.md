# PCP

PCP is a Zig runtime for distributed tensor workloads built on MLIR and IREE.
A gateway exposes HTTP APIs, owns embedded services, and runs a shared
worker-fabric endpoint. Workers register their backend and optional hardware
target, receive program and data assignments, and execute IREE VMFBs on local
devices. Workload code lives under `src/workloads`; protocol coordination lives
under `src/protocol`, `src/nodes`, and `src/algorithms`.

## Architecture

The runtime topology is:

```text
Federation hub -> Gateway -> Workers
```

- `--gateway` starts the site control plane, gateway API, embedded services,
  and worker-fabric TCP endpoint.
- `--federation-hub` starts the cross-gateway coordination API. The current hub
  handles gateway registration, capability placement, forwarding, global job
  lookup/cancel, and graph replication metadata.
- `--worker` starts one worker process and connects it to a gateway worker
  fabric.
- `--node-manager` starts multiple supervised workers on one node.

Gateway and node-manager processes expose Kubernetes probes at `/healthz` and
`/readyz`, plus Prometheus text metrics at `/metrics`.

Supported runtime backends are `cpu`, `cuda`, `rocm`, `metal`, and `vulkan`.
Workers may also provide a target architecture such as `sm_80`, `sm_90a`,
`gfx942`, or `gfx1101`.

## Build

### Cachix cache

The project caches build artifacts and heavy dependencies, including IREE, in
the `pcp` Cachix cache. Configure it from your user profile before the first
build:

```sh
nix profile install nixpkgs#cachix
cachix use pcp --mode user-nixconf
nix profile add github:deltawave-tech/pcp
```

On multi-user Nix installs, the daemon only accepts custom substituters from
trusted users. If Nix reports that `https://pcp.cachix.org` is ignored, add your
user to `trusted-users` in `/etc/nix/nix.conf` and restart the daemon after any
active builds finish.

### Build commands

Use Nix for the repository build:

```sh
nix build
./result/bin/pcp --help
```

For development:

```sh
nix develop
zig build
```

Build the container image tarball with:

```sh
nix build .#pcp-docker
```

The image entrypoint is `pcp`; pass normal runtime args such as `--gateway`,
`--node-manager`, or `--federation-hub`. Mount model artifacts at `/models` and
config files at `/etc/pcp`.

Python is used by exporters, tokenizer helpers, and smoke tests:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Documentation

Detailed runtime, gateway, training, inference, RL, federation, and API
documentation lives in `DOCUMENTATION.md`. Qwen implementation milestones live
in `docs/qwen3-implementation.md`.

## Topology Simulator

The topology simulator in `src/testing/protocol_topology` is a Zig-native
bounded protocol-analysis layer. It models PCP contracts as finite chromatic
complexes, execution carriers, task carriers, and decision maps, then checks the
distributed-computing topology condition `delta o Xi subset Delta`.

It now includes:

- finite simplicial complexes, subcomplexes, vertex maps, carrier maps, and
  carried decision-map validation
- generated and extracted bounded protocol complexes for workload, message
  dispatch, gateway scheduling, reservation lifecycle, federation placement,
  Decoupled DiLoCo, and RDA merge surfaces
- native decision-map search plus Z3-backed SMT-LIB solving with native
  rechecks
- barycentric and standard chromatic subdivision validation for bounded
  protocol-power comparisons
- connectivity, Betti numbers, induced homology, cycle witnesses, and small
  fundamental-group diagnostics
- PCP-named obstruction reports for carrier nonexistence, disconnected task
  outputs, local-view incompatibility, cancellation holes, stale gateways,
  duplicate quorums, lease-owner conflicts, and global rollback gaps
- coverage metadata, trace shrinking, replay fixtures, DOT export, and stable
  report signatures

Run it directly:

```sh
nix develop -c zig test src/testing/protocol_topology/root.zig
```

## Tests

Common checks:

```sh
nix build
zig test src/unit_tests.zig
zig build test
zig test src/testing/protocol_topology/root.zig
```

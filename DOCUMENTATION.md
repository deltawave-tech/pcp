# PCP Documentation

PCP is a distributed MLIR/IREE runtime for training, inference, and RL
workloads. The public runtime surface is a gateway, optional federation hub, and
workers. The gateway owns service APIs and orchestration. Workers own device
execution. Workload-specific code lives under `src/workloads`; protocol and
coordination code lives under `src/protocol`, `src/nodes`, and `src/algorithms`.

## Process Topology

```text
federation hub
  |
gateway API
  |
embedded services: training, inference, RL
  |
shared worker fabric
  |
workers and node-manager launched workers
```

The gateway config schema is in `src/nodes/gateway/config.zig`.

- `graph_backend` is `memory` or `neo4j`.
- `worker_fabric` sets the worker TCP host and port.
- `api.token_env` and `api.internal_token_env` set gateway auth.
- `federation.enabled`, `federation.upstream`, `federation.token_env`, and
  `federation.heartbeat_interval_ms` configure the gateway-to-hub client.
- `controllers.training`, `controllers.inference`, `controllers.rl` enable embedded services.
- Controller scheduling fields are `workers`, `worker_class`, `target_arch`,
  `reserved_workers`, and `max_workers`.

The worker registration path carries backend and target metadata. Scheduling
policies match worker class and target before a controller leases a worker.

### Gateway Operation

`experiments/gateway_local.json` is a minimal gateway config. Embedded services
are enabled through the `controllers` object:

```json
{
  "gateway_id": "lab-alpha-gateway",
  "lab_id": "lab-alpha",
  "graph_backend": "memory",
  "api": {
    "token_env": "PCP_GATEWAY_API_TOKEN",
    "internal_token_env": "PCP_GATEWAY_INTERNAL_TOKEN"
  },
  "worker_fabric": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "controllers": {
    "training": {
      "enabled": true,
      "config_path": "experiments/nanochat_u16_smoke_32.json",
      "service_id": "training-main",
      "workers": 2,
      "worker_class": "cuda",
      "target_arch": "sm_80",
      "api": {
        "host": "127.0.0.1",
        "port": 18100
      }
    }
  }
}
```

Start a gateway:

```sh
PCP_GATEWAY_API_TOKEN=dev \
PCP_GATEWAY_INTERNAL_TOKEN=dev \
./result/bin/pcp \
  --gateway \
  --gateway-config gateway.json \
  --gateway-host 0.0.0.0 \
  --gateway-port 18010 \
  --control-host 0.0.0.0 \
  --control-port 8080
```

Start workers:

```sh
./result/bin/pcp --worker --connect <gateway-ip>:8080 --backend cuda --target sm_80
./result/bin/pcp --node-manager --host <gateway-ip> --port 8080 --scale 8 --backend cuda --target sm_90a
```

### Kubernetes Runtime Environment

PCP treats Kubernetes as the owner of pod lifecycle, resources, secrets, and
mounted storage. Runtime knobs are therefore env-driven:

- `WORKER_MAX_CONCURRENCY` limits in-flight worker task handling. Local
  development defaults to `1`; production mode requires it.
- `STATE_DIR` is the root for stateful runtime files. Production defaults to
  `/var/lib/pcp`; local development defaults to `/tmp/pcp-dev`. Current
  subpaths are `compiler/` for MLIR/IREE compiler temporaries, `compiled/` for
  cached VMFBs, `grpo/` for GRPO snapshots, and `workers/` for worker-local
  materializations.
- `PCP_PROBE_PORT` controls the gateway probe listener. It defaults to `8081`
  and is independent of the user-facing gateway API port.
- `PCP_API_TOKEN_FILE`, `PCP_INTERNAL_TOKEN_FILE`, and
  `PCP_FEDERATION_HUB_TOKEN_FILE` point to mounted secret files. File-backed
  tokens take precedence over token env vars and are re-read during gateway
  authorization, so projected-secret rotation is picked up without restarting
  the pod. If a config names a custom token env var, `<ENV>_FILE` is also
  honored.

Production mode is enabled by `PCP_ENV=production`, `PCP_MODE=production`,
`PCP_PRODUCTION=true`, or the Kubernetes-provided `KUBERNETES_SERVICE_HOST`.
Production gateway boot requires configured tokens. Production worker boot
requires `WORKER_MAX_CONCURRENCY`.

PCP has no embedded Postgres worker-side dependency. A repository audit for
`postgres`, `psql`, `5432`, `libpq`, and Postgres env names is clean; graph
storage is `memory` or external Neo4j through gateway config.

## Backend Runtime

Backend selection lives in `src/backends/selection.zig`.

| PCP backend | IREE compile target | IREE runtime driver |
| --- | --- | --- |
| `cpu` | `llvm-cpu` | `local-sync` |
| `cuda` | `cuda` | `cuda` |
| `rocm` | `rocm` | `hip` |
| `metal` | `metal-spirv` | `metal` |
| `vulkan` | `vulkan-spirv` | `vulkan` |

Workers are started with `--backend` and optional `--target`. The target is used
for IREE compilation flags and for controller scheduling.

## Training

Regular training configs are parsed by `src/workloads/training/config.zig`.
Configs support the current block shape:

- `artifacts`: model paths and bundle/model identifiers.
- `dataset`: data path, split directory, tokenizer, and sampling.
- `distributed`: aggregation, outer loop, local-step, dtype, backend, target,
  seed, and accumulation settings.
- `decoupled_diloco`: fragment count, sync interval, quorum, grace, merge,
  compression, and recovery settings.
- `outputs`: checkpoints, resume settings, output directories, and WandB
  settings.

Legacy flat fields are still normalized for compatibility. The effective
required regular-training fields are:

- `model_path`
- `data_path`
- `tokenizer`
- `sampling`
- `learning_rate`
- `tau`
- `outer_loop_steps`
- `nesterov_momentum`
- `max_epochs`
- `dtype`

Optional fields include `effective_batch_size`, `use_in_graph_accumulation`,
`checkpoint_dir`, `should_resume`, `aggregation_strategy`, Decoupled DiLoCo
settings, and WandB settings.

Example regular training config:

```json
{
  "artifacts": {
    "model_path": "models/nanochat_smoke_32.mlir"
  },
  "dataset": {
    "path": "data/fineweb_edu_2shards_1m.u16",
    "tokenizer": "u16",
    "sampling": "random"
  },
  "distributed": {
    "aggregation_strategy": "decoupled_diloco",
    "learning_rate": 0.0006,
    "local_steps_per_round": 2,
    "outer_loop_steps": 3,
    "outer_momentum": 0.9,
    "max_epochs": 10,
    "dtype": "f32"
  },
  "decoupled_diloco": {
    "num_fragments": 24,
    "sync_interval_h": 24,
    "overlap_tau": 2,
    "min_quorum": 1,
    "merge_strategy": "avg_embedding_rda_model",
    "fragment_strategy": "balanced_tensor",
    "learner_alpha": 0.0,
    "adaptive_grace_enabled": true
  },
  "outputs": {
    "wandb_project": "pcp-distributed",
    "wandb_run_name": "nanochat-u16-smoke-32"
  }
}
```

The gateway training controller loads model MLIR, builds a training graph, and
compiles VMFBs through IREE. Workers execute local inner loops. The gateway
aggregates returned deltas and applies the host-side Nesterov outer update for
regular DiLoCo jobs.

When `aggregation_strategy` is `decoupled_diloco`, regular training is
normalized through `src/protocol/decoupled_job.zig` and
`src/workloads/training/workload.zig`. The gateway starts a Decoupled DiLoCo
syncer, fragments parameters, enforces `min_quorum`, applies grace-window rules,
uses event-tape recovery metadata, and exchanges fragment pull/ready messages
with workers through the shared worker fabric.

Data is assigned through `src/data/assignment.zig`. The assignment format
supports local paths, URI paths, byte ranges, record ranges, record shards, and
materialized blobs.

## Inference

Inference configs are parsed by `src/workloads/inference/config.zig`.

Key fields:

- `model_id`
- `pool_name`
- `generation_vmfb_path`
- `generation_mlir_path`
- `weights_path`
- `tokenizer_source`
- `tokenizer_path`
- `num_gen_data_inputs`
- `max_context_tokens`
- `default_max_output_tokens`
- `worker_backend`
- `worker_target_arch`
- `api_token_env`

The embedded inference controller registers a service with the gateway. The
gateway proxy endpoint is:

```text
POST /v1/inference/chat/completions
```

Qwen inference uses `tools/qwen_tokenizer_server.py` through the tokenizer
adapter and renders Qwen instruct chat prompts before worker execution.

## RL

RL configs use `grpo_config` in `src/workloads/training/config.zig`. The
embedded RL service uses Qwen generation artifacts for rollouts and a GRPO
training MLIR program for updates. The gateway endpoint is:

```text
POST /v1/rl/jobs
```

## Gateway HTTP Surface

Gateway endpoints in `src/nodes/gateway/api.zig`:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/capabilities`
- `GET /v1/services`
- `GET /v1/services/<executor_id>`
- `POST /v1/services/register`
- `POST /v1/training/jobs`
- `POST /v1/rl/jobs`
- `POST /v1/inference/chat/completions`
- `POST /v1/inference/query/chat/completions`
- `GET /v1/jobs`
- `GET /v1/jobs/<job_id>`
- `POST /v1/jobs/<job_id>/cancel`
- `GET /v1/federation/status`
- `GET /v1/federation/peers`
- `GET /v1/federation/services`
- `GET /v1/federation/replication`
- `POST /v1/federation/connect`
- `GET /v1/graph/status`
- `GET /v1/graph/policies`
- `PUT /v1/graph/policies/<namespace_id>`
- `POST /v1/graph/mutate`
- `POST /v1/graph/query`

The gateway also starts a probe-only listener on `PCP_PROBE_PORT` for
`/healthz`, `/readyz`, and `/metrics`, so charts can keep probes off the
user-facing Service.

Federation Hub endpoints in `src/nodes/federation_hub/api.zig`:

- `GET /healthz`
- `GET /readyz`
- `GET /v1/controller`
- `POST /v1/training/jobs`
- `POST /v1/rl/jobs`
- `GET /v1/jobs`
- `GET /v1/jobs/<global_job_id>`
- `POST /v1/jobs/<global_job_id>/cancel`
- `POST /v1/federation/connect`
- `GET /v1/federation/peers`
- `GET /v1/federation/services`
- `POST /v1/federation/mutations`
- `GET /v1/global-graph/status`
- `GET /v1/global-graph/replication`
- `POST /v1/global-graph/query`
- `GET /v1/graph/policies`

The current Federation Hub places and forwards global jobs to compatible
gateways. It tracks gateway registrations, service capabilities, replication
metadata, and global job records. Full multi-gateway contract decomposition is
not implemented yet.

Gateway internal federation endpoints in `src/nodes/gateway/api.zig`:

- `POST /v1/internal/events`
- `POST /v1/internal/federation/training/jobs`
- `POST /v1/internal/federation/training/reservations`
- `POST /v1/internal/federation/rl/jobs`
- `POST /v1/internal/federation/rl/reservations`
- `POST /v1/internal/federation/reservations/<reservation_id>/commit`
- `POST /v1/internal/federation/reservations/<reservation_id>/release`
- `GET /v1/internal/federation/jobs/<job_id>`
- `POST /v1/internal/federation/jobs/<job_id>/cancel`

Controller-local endpoints in `src/nodes/gateway/control_plane/api.zig`:

- `GET /healthz`
- `GET /readyz`
- `GET /v1/controller`
- `GET|POST /v1/job`
- `GET|POST /v1/jobs/current`
- `POST /v1/job/cancel`
- `POST /v1/jobs/current/cancel`
- `GET /v1/workers`
- `GET /v1/metrics`

Node-manager probe endpoints in `src/nodes/node_manager.zig`:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Prometheus Metrics

Gateway and node-manager expose Prometheus text format on unauthenticated
`GET /metrics` endpoints. Counters are monotonically increasing process-local
values. Gauges are point-in-time values at scrape time.

Gateway metrics:

- `pcp_gateway_info{gateway_id,lab_id,graph_backend}`: gauge, value `1`.
- `pcp_gateway_draining`: gauge, `1` after SIGTERM/SIGINT drain starts.
- `pcp_gateway_registered_services`: gauge, service count.
- `pcp_gateway_service_workers{service_id,executor_id,service_type,worker_class,target_arch,state}`:
  gauge, worker count by `state` of `connected`, `ready`, or `available`.
- `pcp_gateway_service_reserved_workers{service_id,executor_id,service_type,worker_class,target_arch}`:
  gauge, reserved worker count.
- `pcp_gateway_service_max_workers{service_id,executor_id,service_type,worker_class,target_arch}`:
  gauge, configured max workers; `-1` means unset.
- `pcp_gateway_service_job_status{service_id,executor_id,service_type,worker_class,target_arch,status}`:
  gauge, value `1` for the latest service job status label.
- `pcp_gateway_federation_connected`: gauge, `1` when connected to a hub.
- `pcp_gateway_federation_peers`: gauge, visible federation peer count.
- `pcp_gateway_federation_pending_mutations`: gauge, graph mutation count.
- `pcp_gateway_federation_replication_lag`: gauge, local sequence lag.
- `pcp_gateway_api_requests_total`: counter, handled HTTP request count.
- `pcp_gateway_api_errors_total`: counter, request error-path count.
- `pcp_gateway_api_request_duration_seconds_count`: counter, duration sample count.
- `pcp_gateway_api_request_duration_seconds_sum`: counter, request duration sum in seconds.

Node-manager metrics:

- `pcp_node_manager_info{backend,target_arch}`: gauge, value `1`.
- `pcp_node_manager_expected_supervisors{backend,target_arch}`: gauge, configured supervisor count.
- `pcp_node_manager_launched_supervisors{backend,target_arch}`: gauge, launched monitor count.
- `pcp_node_manager_running_supervisors{backend,target_arch}`: gauge, tracked supervisor child processes.
- `pcp_node_manager_ready{backend,target_arch}`: gauge, `1` when readiness conditions pass.

## Container Image

The Nix package `pcp-docker` builds a self-contained runtime image:

```sh
nix build .#pcp-docker
```

The image includes:

- the wrapped `pcp` binary and IREE runtime closure
- a Python runtime with `transformers` and `wandb`
- `/app/tools/qwen_tokenizer_server.py`
- `/app/tools/wandb_adapter.py`
- conventional directories `/models`, `/etc/pcp`, and `/var/lib/pcp`

The image entrypoint is `pcp-entrypoint`, which changes to `/app` and execs
the wrapped PCP binary. Chart commands can pass the normal PCP CLI args
directly:

```text
--gateway --gateway-config /etc/pcp/gateway.json
--node-manager --scale 8 --backend cuda --host <gateway-worker-fabric-host>
--federation-hub --api-host 0.0.0.0 --api-port 19010
```

`/models` is the conventional mount path for model assets: weights, VMFBs,
tokenizer directories, and exported contracts/metadata.
PCP does not prescribe how those files arrive; use an init container, mounted
volume, or model-registry sync process. Config files should be mounted under
`/etc/pcp` and referenced explicitly from CLI flags. Stateful runtime files and
caches should live under `STATE_DIR`.

The image sets:

- `PCP_MODEL_ROOT=/models`
- `PCP_CONFIG_ROOT=/etc/pcp`
- `STATE_DIR=/var/lib/pcp`
- `TRANSFORMERS_CACHE=/var/lib/pcp/hf-cache`
- `HF_HOME=/var/lib/pcp/hf-home`

## Public Copy

The public `deltawave-tech/pcp` copy is materialized from `pcp-internal` using
`docs/materialize-public-copy.sh` and checked with `docs/verify-public-cut.sh`.
Cut the public copy after each release candidate and after major deployability
or API-boundary changes that downstream charts need to consume. Routine private
experiments do not require a public cut.

## Topology Simulator

The topology simulator is the bounded protocol-analysis layer in
`src/testing/protocol_topology`. It models PCP control-plane behavior as finite
chromatic complexes and checks that reachable protocol views admit legal
decisions under the declared task carrier. It is a real bounded simulator for
PCP protocol design and implementation checks: it builds or extracts finite
topological objects, validates carrier conditions, searches for legal decisions,
and explains bounded impossibility cases.

Core objects:

- `I`: input complex for legal starting configurations.
- `P`: protocol complex for reachable final local views.
- `O`: output complex for legal decisions.
- `Xi: I -> P`: execution carrier.
- `Delta: I -> O`: task carrier.
- `delta: P -> O`: color-preserving decision map.
- Correctness check: `delta o Xi subset Delta`.

Implemented capabilities:

- Finite chromatic complex kernel with face closure, subcomplex operations,
  skeletons, boundaries, stars, links, joins, and duplicate-color rejection.
- Simplicial maps and carrier maps with monotone, strict, rigid, and chromatic
  validation.
- Generated protocol complexes from bounded traces, with execution carriers,
  trace witnesses, partial-order reduction, seeded exploration, and shrinking.
- Extraction adapters for production-like workload, message-registry,
  scheduling, reservation, federation, Decoupled DiLoCo, and RDA behavior.
- Carried decision-map validation, native finite decision-map search, minimal
  unsat cores, SMT-LIB export, and Z3-backed solving with imported models
  rechecked by the native topology engine.
- Barycentric and standard chromatic subdivisions with independent
  protocol-power validators, boundary and link checks, round/view metadata, and
  bounded iterated-subdivision equivalence.
- Connectivity, path witnesses, Betti numbers over finite fields, induced
  homology ranks, bounded source/image cycle witnesses, and small
  fundamental-group presentations.
- PCP task catalog, cross-layer carrier composition, carrier-refinement checks,
  counterexample reports, replay fixtures, DOT export, coverage summaries, and
  CI artifact signatures.
- Case-driven obstruction diagnostics for carrier nonexistence, disconnected
  task outputs, hole preservation or creation failures, incompatible local-view
  identifications, and solver unsat cases.

Modeled protocol surfaces:

- Workload normalization for regular training.
- Message registry dispatch and fail-closed handler ownership.
- Federation placement with gateway compatibility, staleness, loss, and registry
  ranking.
- Gateway scheduling with lease owners, `reserved_workers`, `max_workers`,
  cross-owner capacity preservation, queued jobs, and cancellation cleanup.
- Decoupled DiLoCo fragments with quorum, duplicate queued responses,
  context/payload mismatch, unique sender collection, vector-clock freshness,
  stale updates, cancellation, resume compatibility, and event-tape replay.
- Merge oracles for direct average, token-weighted average, and RDA
  direction/norm carriers.
- Multi-gateway global-job task states for commit, rollback, cancel, capacity
  deferral, and a bounded global rollback-gap obstruction under partial gateway
  loss. This path is still design-only until the corresponding production
  surface stabilizes.

Coverage and evidence:

- Each task definition records the production surface, coverage paths, evidence
  level, and whether the path is design-only.
- Evidence levels distinguish symbolic carriers, production fixtures, bounded
  models, generated execution carriers, solver checks, and smoke replays.
- Coverage summaries make it visible when a production path is implemented,
  generated, solver-checked, smoke-backed, symbolic, or design-only.

Failure explanations:

- Carrier escape reports include the input simplex, offending output simplex,
  decided image, allowed image, replay seed, and replay events when available.
- Obstruction reports include the input face, protocol carrier image, output
  carrier image, failed invariant, PCP witness term, Betti values, induced
  ranks, solver core vertices, or concrete homology cycle witness where
  available.
- PCP witness terms currently include partition, stale gateway, duplicate
  quorum, cancellation hole, global rollback gap, and lease-owner conflict.

Scope boundary:

- The simulator is bounded and finite. It is not an unbounded theorem prover for
  arbitrary distributed systems.
- Unsupported dimensions, solver timeouts, and incomplete extraction data are
  reported explicitly instead of treated as proofs.
- Solver answers are accepted only after native topology validation.

Run the simulator directly:

```sh
nix develop -c zig test src/testing/protocol_topology/root.zig
```

It is also included in the aggregate unit target:

```sh
nix develop -c zig test src/unit_tests.zig
```

## Verification

Canonical build:

```sh
nix build
```

Core Zig checks:

```sh
zig test src/unit_tests.zig
zig build test
zig test src/testing/protocol_topology/root.zig
```

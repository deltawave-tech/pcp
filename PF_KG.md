# Protocol Foundations: Remaining KG + Federation Scope

## Status

The following foundations are already implemented in PCP:

- local `gateway` process role
- unified gateway API for local inference, RL, training, and graph access
- service registration plus controller proxying
- local graph API with in-memory and Neo4j backends
- service-to-gateway event ingestion
- `global-controller` role
- gateway registration and heartbeat
- gateway-to-global mutation replication with replay from last acked sequence
- namespace-level graph policy storage and policy-aware replication control
- gateway/global policy inspection APIs
- gateway-visible federated query modes: `local`, `global`, and `local_plus_global`

This document now tracks only the missing pieces.

## Remaining Goal

Finish the extension from "working local gateway plus replicated global graph" into the intended multi-lab agent-facing system.

The remaining work falls into four buckets:

1. richer global graph semantics
2. EVE-facing composed operations
3. deferred inference extensions
4. deferred autoresearch extensions

## Missing Pieces

## 1. Richer Global Graph Semantics

What is still missing:

- canonical identity resolution across gateways
- local-to-global entity linking
- identity-aware federated query semantics beyond the current raw ID-based local/global composition
- conflict handling rules when multiple gateways contribute overlapping facts
- explicit replication/share state inspection beyond last-acked sequence

Minimum concrete deliverables:

- canonical ID model for cross-lab entities
- global-controller-side identity resolution rules
- conflict policy for entity upsert, relation upsert, and observation append
- better replication inspection APIs for operator use

What just landed:

- the gateway can now answer `local`, `global`, and `local_plus_global` graph queries through one request surface
- `local_plus_global` currently merges by raw object ID only
- this is sufficient for first federated retrieval, but not yet for cross-lab aliasing or canonical entity identity
- the gateway now also exposes a first composed query-then-infer path that retrieves federated graph context and prepends it to an inference request

Recommended query modes:

- `local`
- `local_plus_global`
- `global`

## 2. EVE-Facing Integration Surface

The gateway now exposes the building blocks, but not the higher-level composed interface.

What is still missing:

- graph-linked job launch flows for RL and training
- richer experiment/artifact linkage in graph provenance
- stable gateway-facing endpoints designed around agent workflows rather than just thin proxying
- stronger shaping of retrieved graph context before it is injected into prompts
- graph-linked recording of composed inference requests and outputs

Minimum concrete deliverables:

- gateway endpoint that launches RL or training linked to graph entities
- graph conventions for experiments, runs, checkpoints, rewards, and outputs
- graph-linked conventions for composed inference requests, retrieved context, and outputs

Acceptance target:

- an EVE-like client can use one gateway to:
  - search local plus global knowledge
  - run inference with retrieved context
  - start RL/training runs linked to graph objects
  - observe resulting outputs and checkpoints through graph records

## 3. Inference Extensions Still Missing

The current inference surface is useful, but it is not yet the intended full interface.

Still missing:

- `tool_calls`
- broader OpenAI-surface compatibility beyond current chat completions behavior
- gateway-level inference cancellation and session APIs
- stronger cache observability surfaced through the gateway
- artifact exposure for VMFB, MLIR, weights, and tokenizer snapshots
- any TurboQuant-backed KV optimization

## 4. Autoresearch Layer Still Missing

Nothing from the autoresearch layer is implemented yet.

Still missing:

- experiment manifests
- remote cluster orchestration
- artifact staging and result collection
- log and metric streaming across remote runs
- budget/safety controls for provider, region, GPU-hours, and destructive actions

This should remain separate from the core gateway/global-controller graph work until the policy and composed gateway surfaces are stable.

## Recommended Next Build Order

1. add better replication/share inspection APIs for operators
2. add canonical identity and local-to-global linking rules for federated query results
3. add graph-linked recording and better context shaping for the query-then-infer flow
4. add graph-linked RL/training launch flows on top of the same job and graph conventions
5. only then expand broader inference parity and autoresearch scope

## Recent Fix

- inference service registration freshness at the gateway is now correct for the composed query-then-infer flow
- the smoke now verifies that `/v1/services` reports the inference service as `health_status:"ok"` with `ready_worker_count:1` once the CUDA worker is ready

## Suggested Remaining Repo Areas

- `src/gateway/api.zig`
- `src/global_controller/identity.zig`
- `src/global_controller/api.zig`
- `src/federation/protocol.zig`

## Design Constraints

- keep service controllers service-focused
- keep graph authority at gateway/global-controller layers
- continue replicating typed mutations, not raw backend internals
- keep Neo4j behind the existing graph abstraction
- do not turn worker-local graph state into the primary source of truth

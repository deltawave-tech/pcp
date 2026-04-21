Current State

  Inference is still the strongest controller API surface. The inference controller
  starts an HTTP server plus the worker control plane, authorizes requests with a
  bearer token, exposes GET /healthz, GET /readyz, GET /v1/models, and POST
  /v1/chat/completions, and routes requests to ready workers.

  Training and RL are no longer pure CLI-only processes. They now expose a shared
  controller/operator HTTP surface through the control-plane server:

  - GET /v1/controller
  - GET /v1/job
  - GET /v1/workers
  - GET /v1/metrics
  - POST /v1/job/cancel

  The gateway already proxies a subset of this into:

  - POST /v1/training/jobs
  - POST /v1/rl/jobs
  - GET /v1/jobs
  - GET /v1/jobs/:id
  - POST /v1/jobs/:id/cancel

  The gateway now also has a first composed inference endpoint:

  - POST /v1/inference/query/chat/completions

  That endpoint performs federated graph retrieval first, injects the retrieved graph
  JSON as prompt context, and then calls the existing inference service.

  Assessment

  They are not unified enough yet for a seamless external orchestration service.

  What is usable now:

  - Inference can already be consumed by an outside service as a normal HTTP API.
  - Inference has basic readiness/health and some runtime metrics via readyz.
  - Inference has session affinity and worker-side KV reuse, which is a good base for
    multi-agent workloads. See src/inference/router.zig:10 and src/nodes/controllers/
    inference_shepherd.zig:722.
  - Training and RL have a basic controller API for status, metrics, worker listing,
    and cancellation.
  - The gateway has a minimal shared job lookup/cancel surface across inference, RL,
    and training services.
  - The gateway now has a first agent-facing composed retrieval-plus-inference path.

  What is not unified:

  - Training and RL still do not expose API-driven job creation, resume, pause,
    checkpoint listing, or log/event streaming.
  - Inference still has a strong data-plane API, but not a full deployment/control
    plane.
  - There is no shared resource model across training, RL, and inference jobs.

  In other words, today PCP exposes:

  - a usable inference serving plane
  - but only a process/CLI execution plane for training and RL

  That is not enough for an autoresearch-style agent that should create, monitor,
  pause, resume, compare, and manage experiments programmatically.

  Main Gaps

  1. No unified job abstraction
     There is no first-class concept of job, run, experiment, deployment, or session
     group shared across controller types. Training has a local run_id and checkpoint
     directory logic in src/pcp.zig:551, but that is filesystem-driven, not API-driven.
  2. No control-plane API for training or RL
     This is now only partially true. An external service can get status, fetch
     metrics, list workers, and cancel a running training or RL controller. What is
     still missing is create/resume/pause/checkpoint/log/event functionality over an
     API.
  3. Weak operational observability
     Inference exposes only minimal metrics. Training/RL metrics are local monitoring
     state and WandB side effects, not a controller API. There is no unified machine-
     readable status model for:

  - pending
  - starting
  - waiting_for_workers
  - running
  - paused
  - checkpointing
  - failed
  - completed
  - cancelling

  4. No lifecycle operations
     There is no controller API for:

  - graceful cancel
  - pause
  - resume by run id
  - scale worker pool expectations
  - drain a worker
  - reload model
  - switch inference deployment version
  - snapshot state on demand

  5. No artifact/config registry surface
     Configs, VMFBs, MLIR, weights, checkpoints, prompts, and tokenizers are all file-
     path based. That is fine internally, but an external orchestration service needs
     stable identifiers and API-visible metadata for them.
  6. Inference routing is too primitive for production
     Current routing is session-affinity first, otherwise “first ready worker,”
     effectively from hash-map iteration order. See src/inference/router.zig:10 and
     src/nodes/controllers/inference_shepherd.zig:26. That is not production-grade
     scheduling.
  7. No multitenancy or policy layer
     Inference has a single bearer token check in src/nodes/controllers/
     inference_shepherd.zig:515. There is no notion of tenant, quota, namespace,
     priority, or authorization scopes. Training/RL have no API auth model at all.
  8. No idempotent external operations
     A production control plane needs request ids / idempotency keys for job creation,
     resume, cancellation, deployment updates, and checkpoint actions.
  9. No unified queue/admission model
     Inference queues requests when no worker is ready, but training and RL do not
     expose a scheduler queue. There is no cluster-wide admission control or resource
     reservation across job types.
  10. No structured worker inventory API
     The system internally knows worker backend/status/address in src/nodes/
     controllers/shepherd.zig:31, but this is not exposed as a proper service API for
     operators.

  What I’d Add

  For production-grade external orchestration, I would add a unified controller API
  above all three modes.

  Core resource model:

  - clusters
  - workers
  - models
  - artifacts
  - deployments
  - jobs

  Job types:

  - training
  - rl
  - inference_deployment
  - maybe later batch_inference

  Minimum API surface:

  - POST /v1/jobs
  - GET /v1/jobs
  - GET /v1/jobs/:id
  - POST /v1/jobs/:id/cancel
  - POST /v1/jobs/:id/pause
  - POST /v1/jobs/:id/resume
  - GET /v1/jobs/:id/events
  - GET /v1/jobs/:id/metrics
  - GET /v1/jobs/:id/logs
  - GET /v1/jobs/:id/checkpoints

  For inference deployments:

  - POST /v1/deployments
  - GET /v1/deployments/:id
  - PATCH /v1/deployments/:id
  - POST /v1/deployments/:id/scale
  - POST /v1/deployments/:id/drain
  - GET /v1/deployments/:id/workers

  For cluster operations:

  - GET /v1/workers
  - GET /v1/workers/:id
  - POST /v1/workers/:id/drain
  - POST /v1/workers/:id/undrain

  Specific Improvements By Mode

  Inference:

  - Better worker selection: least-loaded, queue depth, latency-aware, or pool-based
    round robin.
  - Explicit request ids and cancellation.
  - Richer metrics: per-model, per-worker, queue latency, end-to-end latency, tokens/
    sec.
  - Deployment lifecycle: load/unload/swap model versions without process-level
    choreography.
  - Session introspection and eviction APIs.

  Training:

  - External job creation instead of local process bootstrap only.
  - API-visible run ids, configs, checkpoints, and recovery state.
  - Structured progress metrics and state transitions.
  - Pause/resume/cancel endpoints.
  - Artifact references instead of raw path strings only.

  RL:

  - Same as training, plus:
  - prompt dataset management
  - rollout stats
  - reward metrics
  - policy/version checkpoints
  - evaluation hooks

  My Bottom-Line Opinion

  Inference is already a decent starting point for external service integration.

  Training and RL are now basic operator APIs, but they are not yet external
  orchestration APIs.

  If the goal is “an autoresearch agent can use PCP as a cluster substrate,” the
  missing piece is not mainly model logic. It is a unified control plane:

  - common job model
  - common lifecycle
  - common observability
  - common auth/policy
  - common artifact references
  - better scheduling semantics

  The cleanest path is:

  1. Keep the existing inference HTTP API as the serving data plane.
  2. Keep expanding the gateway as the single composed/agent-facing surface.
  3. Add a new unified control-plane API for training, RL, and inference deployments.
  4. Make all controllers report into the same job/status/event model.

  Current issue to note:

  - inference service registration freshness is not fully correct at the gateway yet
  - in the composed query-then-infer smoke, the request succeeded through a ready CUDA
    worker, but `/v1/services` still showed the inference service as
    `health_status:"starting"` and `ready_worker_count:0`
  - that suggests registration is not being refreshed after worker readiness changes

  If you want, I can turn this into a concrete design proposal next:

  - proposed endpoint schema
  - resource JSON shapes
  - controller state machine
  - which existing internals can be reused vs what needs refactoring first.

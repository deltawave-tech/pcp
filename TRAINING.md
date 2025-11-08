# Training Infrastructure Architecture

## Overview

This document describes the architecture for running distributed training across heterogeneous platforms (Mac on Scaleway, Linux AMD64 with Nvidia GPU on GCP).

## High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│         GitHub Actions Workflow (Manual Trigger)      │
│         Environment: "training" (provides locking)    │
└────┬──────────────────────────────┬──────────────────┘
     │                               │
     │ 1. Start GCP VM              │ 6. Collect results
     │    (gcloud start)             │    Stop GCP VM
     │                               │    (gcloud stop)
     ▼                               ▼
┌──────────────────┐         ┌─────────────────────┐
│ GCP Linux + GPU  │         │  Mac (Scaleway)     │
│ (Worker Node)    │◀──TCP───│  (Head/Shepherd)    │
│                  │ :PORT   │                     │
│ • Cachix binary  │         │ • Cachix binary     │
│ • Auto-starts    │         │ • Coordinates       │
│ • Exits on done  │         │ • Exits on done     │
└──────────────────┘         └─────────────────────┘
     │                               │
     └───────────┬───────────────────┘
                 ▼
         ┌──────────────────┐
         │  Results Storage │
         │  (GCS Bucket)    │
         └──────────────────┘
```

## Components

### 1. Infrastructure as Code (Pulumi + Python)

**Repository structure:**
```
infra/
  ├── __main__.py          # Main Pulumi program (logic for all stacks)
  ├── Pulumi.yaml          # Project config
  ├── Pulumi.tiny.yaml     # Tiny stack config (2 nodes) - CURRENT
  ├── Pulumi.small.yaml    # Small stack config (4 nodes) - FUTURE
  ├── Pulumi.large.yaml    # Large stack config (8 nodes) - FUTURE
  ├── Pulumi.huge.yaml     # Huge stack config (16 nodes) - FUTURE
  └── requirements.txt     # pulumi, pulumi-gcp
```

**Managed resources:**
- GCP Compute Instance(s) with GPU (persistent, stopped when not in use)
- Firewall rules (allow TCP port for worker→shepherd communication)
- Startup scripts (pulls cachix binary, starts worker on boot)
- GCS bucket for results storage
- Outputs: VM external IPs, connection details

**Note:** VMs are created once and persist. They are started/stopped per training run, not created/destroyed.

**State Management:**
- Pulumi Service (pulumi.com) for remote state storage
- Automatic state locking and encryption
- State shared between local development and GitHub Actions
- Free tier sufficient for this use case

### 2. GitHub Actions Workflow

**Trigger:** `workflow_dispatch` (manual button in GitHub UI)

**Workflow Inputs:**
- `stack` (choice: tiny/small/large/huge, default: tiny) - Environment to use
- `keep_vm_alive` (boolean, default: false) - Keep VMs running after training
- `training_config` (string, optional) - Additional training parameters

**Environment:** `training`
- Provides concurrency control (only one training run at a time)
- Stores secrets (GCP credentials, SSH keys for Mac)
- Optional: Require approval before run starts

**Workflow Steps:**
1. Checkout code
2. Setup Pulumi CLI
3. Select stack: `pulumi stack select <stack>`
4. Ensure infrastructure exists (`pulumi up` - idempotent)
5. Start GCP VMs: `gcloud compute instances start <vm-names>`
6. Wait for VMs to boot
7. SSH to Mac (Scaleway): Start shepherd process
8. GCP VMs auto-start workers (via startup script)
9. Wait for all processes to exit (training complete)
10. Collect results from GCS bucket
11. Stop GCP VMs: `gcloud compute instances stop <vm-names>` (unless keep_vm_alive=true)

### 3. Training Execution Flow

**GCP VM (Worker Node):**
```bash
# Startup script (runs on boot)
# 1. Install/update cachix
# 2. Pull training binary from cachix
# 3. Start worker: ./binary worker --head-node=<MAC_IP>:<PORT>
# 4. On exit, write completion marker to GCS
```

**Mac (Shepherd/Head Node):**
```bash
# Triggered via SSH from GitHub Actions
# 1. Pull training binary from cachix
# 2. Start shepherd: ./binary shepherd --port=<PORT>
# 3. On exit, write results to GCS
```

**Network Communication:**
- Worker connects to shepherd via TCP
- Worker initiates connection to shepherd's IP:PORT
- No external coordination needed beyond hostname/port

### 4. Results & Artifacts

**Current:**
- Both nodes write outputs to GCS bucket:
  - Training logs
  - Model checkpoints (if applicable)
  - Metrics/results JSON
  - Completion markers

**Future:**
- Full wandb integration for real-time monitoring
- Automatic metric reporting

### 5. VM Lifecycle Management

**Default Behavior:**
- VM stops automatically after training completes
- Cost optimization: Only pay for compute during training

**Keep-Alive Mode:**
- Workflow input `keep_vm_alive=true` skips shutdown step
- Useful for debugging or consecutive runs

**Safety Mechanism:**
- GCP VM configured with auto-stop after N hours (metadata/scheduled action)
- Prevents runaway costs if shutdown fails

### 6. Concurrency & Locking

**GitHub Actions Environment:**
- Environment name: `training`
- Concurrency configuration:
  ```yaml
  concurrency:
    group: training
    cancel-in-progress: false  # Queue runs, don't cancel
  ```
- If multiple runs triggered, they queue and execute sequentially
- Prevents resource conflicts between training runs

## Environments (Pulumi Stacks)

The infrastructure supports multiple isolated environments via Pulumi stacks. Each stack maintains independent state and resources, allowing different scales of training deployments.

### Current Environment

**Tiny (2 nodes)** - `pulumi stack select tiny`
- 1 Shepherd node (Mac on Scaleway)
- 1 Worker node (GCP Linux + GPU)
- Purpose: Initial development and testing
- Status: **Currently implemented**

### Future Environments

The following environments are planned for future expansion:

**Small (4 nodes)** - `pulumi stack select small`
- 1 Shepherd node
- 3 Worker nodes
- Purpose: Medium-scale training experiments
- Status: **Planned**

**Large (8 nodes)** - `pulumi stack select large`
- 1 Shepherd node
- 7 Worker nodes
- Purpose: Large-scale distributed training
- Status: **Planned**

**Huge (16 nodes)** - `pulumi stack select huge`
- 1 Shepherd node
- 15 Worker nodes
- Purpose: Maximum-scale training runs
- Status: **Planned**

### Stack Configuration

Each stack has its own configuration file (`Pulumi.<stack>.yaml`) defining:
- Number of worker nodes
- Machine types (CPU/GPU specifications)
- GPU types (e.g., T4, V100, A100)
- Network configuration
- Cost optimization settings

**Example:** `Pulumi.tiny.yaml`
```yaml
config:
  pcp:worker_count: 1
  pcp:machine_type: "n1-standard-4"
  pcp:gpu_type: "nvidia-tesla-t4"
  pcp:shepherd_host: "<MAC_IP>"
  pcp:tcp_port: "9000"
```

### Stack Benefits

- **Isolation:** Each environment has separate resources and state
- **Scalability:** Easy to add more environments as needs grow
- **Cost Control:** All VMs stopped when idle; only pay for active training
- **Flexibility:** Different configurations per environment (dev uses cheaper GPUs)
- **Safety:** Changes to one environment don't affect others

## Key Architectural Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| IaC Tool | Pulumi (Python) | Team preference, programmable infrastructure |
| State Management | Pulumi Service | Zero setup, free tier, automatic locking |
| Environments | Pulumi Stacks | Isolated environments at different scales |
| Trigger Mechanism | GitHub workflow_dispatch | Semi-automatic, accessible to team |
| Locking | GH Actions Environment | Simple, built-in concurrency control |
| Mac Access | SSH from GH Actions | Mac is long-lived, just execute commands |
| GCP VM Lifecycle | Persistent (start/stop) | Fast startup, cost-effective |
| Results Storage | GCS Bucket | Cloud-native, accessible from both nodes |
| Binary Distribution | Cachix | Current system, works well |
| Containerization | Future Phase | Plan to adopt, not blocking initial setup |

## Platform Support

**Current:**
- Mac (Scaleway) - Shepherd node
- Linux AMD64 + Nvidia GPU (GCP) - Worker node

**Future:**
- Additional worker nodes on different architectures
- Heterogeneous training across multiple cloud providers
- ARM64 support

## Future Enhancements

### Short-term
- Containerize binaries (Docker/OCI images)
- Complete wandb integration
- Auto-stop safety mechanisms
- Detailed logging and monitoring

### Medium-term
- Support for multiple worker nodes
- Advanced queuing system
- Auto-trigger on git tags (with approval)
- Cost tracking and reporting

### Long-term
- Multi-cloud support (AWS, Azure, etc.)
- Spot/preemptible instance support
- Distributed training across >2 nodes
- Auto-scaling based on workload

## Prerequisites

### To Implement:
1. GCP project with Compute Engine API enabled
2. GCP service account with appropriate permissions
3. SSH access to Mac (Scaleway)
4. Pulumi account (free tier sufficient) at https://app.pulumi.com
5. GitHub repository secrets:
   - `GCP_CREDENTIALS` - Service account JSON
   - `MAC_SSH_KEY` - Private key for Mac access
   - `MAC_HOST` - Mac hostname/IP
   - `PULUMI_ACCESS_TOKEN` - Token from Pulumi Service
6. GCS bucket for results storage

### Binary Requirements:
- Training binary available in cachix
- Binary supports `shepherd` and `worker` modes
- Worker can connect to shepherd via `--head-node=<HOST>:<PORT>`
- Both exit cleanly on training completion

## Open Questions

1. Specific GCP machine type for GPU training?
2. Exact TCP port for shepherd/worker communication?
3. Auto-stop timeout value (hours)?
4. Results retention policy in GCS?
5. Approval requirement for training environment?

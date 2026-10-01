# MLflow on RHOAI 3.5 — Demo Application

A hands-on demo covering all key MLflow features available in Red Hat OpenShift AI (RHOAI) Self-Managed 3.5, including live Nemotron LLM inference tracing and RAG pipeline evaluation.

## What This Demo Covers

| Notebook | Audience | Topic |
|----------|----------|-------|
| [00_cluster_setup.ipynb](notebooks/00_cluster_setup.ipynb) | **Admin** | Enable MLflow Operator, deploy Minimal server (SQLite + PVC), RBAC |
| [01_setup_authentication.ipynb](notebooks/01_setup_authentication.ipynb) | Developer | SDK install, auth modes, workspace connectivity |
| [02_experiment_tracking.ipynb](notebooks/02_experiment_tracking.ipynb) | Developer | Experiments, runs, params, metrics, artifacts, autolog |
| [03_hyperparameter_tuning.ipynb](notebooks/03_hyperparameter_tuning.ipynb) | Developer | Nested runs, grid search, programmatic run comparison |
| [04_genai_tracing.ipynb](notebooks/04_genai_tracing.ipynb) | Developer | LLM/GenAI tracing, custom spans, sessions, trace archival |
| [05_model_registry.ipynb](notebooks/05_model_registry.ipynb) | Developer | Model registration, versioning, aliases, loading |
| [06_nemotron_tracing.ipynb](notebooks/06_nemotron_tracing.ipynb) | Developer | Live Nemotron LLM tracing — single completions, multi-turn sessions, temperature sweep |
| [07_nemotron_rag_eval.ipynb](notebooks/07_nemotron_rag_eval.ipynb) | Developer | RAG pipeline with TF-IDF retrieval, systematic prompt × top_k evaluation, artifact logging |

> **Notebooks 01–05** run fully offline with synthetic data — no LLM endpoint required.
> **Notebooks 06–07** call the Nemotron InferenceService deployed in the cluster.

## Prerequisites

### RHOAI Cluster Requirements
- RHOAI 3.5 installed
- `oc` CLI access with cluster-admin (for notebook 00 only)
- An OpenShift project (namespace) to use as your MLflow workspace
- Refer to [ai-accelerator](https://github.com/redhat-ai-services/ai-accelerator) for installing RHOAI 

> If your cluster already has a running `MLflowTrackingServer`, skip notebook 00 and start at notebook 01.

### Nemotron InferenceService (Notebooks 06–07)

A Nemotron model must be deployed as an RHOAI InferenceService before running notebooks 06 or 07.

```bash
# Verify the InferenceService is ready
oc get inferenceservice -n <your-namespace> | grep nemotron

# Get the route
oc get route -n <your-namespace> | grep nemotron

# Confirm the model name
curl -sk -H "Authorization: Bearer $(oc whoami --show-token)" \
  https://<nemotron-route>/v1/models | jq '.data[].id'
```

Update `LLM_BASE_URL` and `MODEL_ID` in the first cell of each notebook to match your deployment. The endpoint probe cell (Section 1) will auto-detect the model name and API protocol (OpenAI-compatible or KServe v2) at runtime.

### Deployment Modes

| Mode | Database | Artifacts | Replicas | Use case |
|------|----------|-----------|----------|----------|
| **Minimal** | SQLite (on PVC) | PVC (local) | 1 | Dev, test, demo — no external dependencies |
| **Production** | External PostgreSQL | S3-compatible store | 2+ | Production workloads, HA, Trace Archival |

Notebook 00 deploys the **Minimal** mode. The production CR is included in that notebook as reference.

### Workbench Setup (Recommended Path)

The easiest way to run this demo is inside an RHOAI Workbench with automatic MLflow integration:

1. Create a workbench in your RHOAI project
2. In the workbench settings, enable the MLflow integration (annotation `opendatahub.io/mlflow-instance`)
3. RHOAI automatically injects:
   - `MLFLOW_TRACKING_URI`
   - `MLFLOW_TRACKING_AUTH=kubernetes-namespaced`
   - `MLFLOW_K8S_INTEGRATION=true`
4. Clone this repo into the workbench and run the notebooks in order

### Manual Setup (Outside Workbench)

```bash
pip install "mlflow[kubernetes]>=3.11"

export MLFLOW_TRACKING_URI="https://<your-mlflow-route>/workspaces/<namespace>"
export MLFLOW_TRACKING_AUTH="kubernetes-namespaced"
# OR use a token:
# export MLFLOW_TRACKING_TOKEN="<your-token>"
# export MLFLOW_WORKSPACE="<namespace>"
```

### Disconnected / Air-Gapped Clusters

Standard RHOAI workbench images do not include the MLflow SDK, and there is no internet egress to install it at runtime. Build a custom workbench image:

```bash
# 1. On a connected machine — download wheels
pip download "mlflow[kubernetes]>=3.11" "packaging>=23" \
    --platform manylinux_2_28_x86_64 \
    --python-version 311 \
    --only-binary=:all: \
    -d ./wheels/

# 2. Containerfile
# FROM registry.redhat.io/rhoai/odh-generic-data-science-notebook-rhel9:v3.5
# USER 0
# COPY wheels/ /tmp/wheels/
# RUN pip install --no-index --find-links=/tmp/wheels \
#       "mlflow[kubernetes]>=3.11" "packaging>=23" && \
#     rm -rf /tmp/wheels/
# USER 1001

# 3. Build, push, and register in RHOAI Dashboard > Settings > Notebook images
```

Notebooks 06–07 require only `requests` and `scikit-learn`, which are pre-installed in the standard data science workbench image. The custom image above is sufficient for all seven notebooks.

## Key RHOAI 3.5 MLflow Facts

- **Server version:** 3.13.0  |  **Minimum SDK:** `mlflow[kubernetes]>=3.11`
- **Auth model:** Kubernetes RBAC via `SelfSubjectAccessReview` — every API call is authorized
- **1:1 mapping:** One OpenShift project namespace = one MLflow workspace
- **New in 3.5:** Trace Archival (moves span payloads from PostgreSQL → S3 via CronJob)
- **GenAI workflow:** Dashboard supports GenAI mode with Traces/Sessions tabs alongside classic Model Training mode
- **MLflow 3.x breaking changes:** `artifact_path` → `name` in `log_model`; skops default serialisation requires explicit trust or `SERIALIZATION_FORMAT_CLOUDPICKLE`; `mlflow.set_trace_tags` does not exist — use `span.set_attribute` instead

## RHOAI Dashboard Navigation

After running the notebooks, explore the results at:

```
Develop & train > Experiments (MLflow)
```

Features available in the dashboard:
- Browse experiments and runs
- Interactive metric charts (line, bar)
- System metrics (CPU / memory)
- **Compare runs** with parallel coordinates, scatter, box, and contour plots
- "Show differences only" toggle for run comparison
- Cross-experiment run comparison
- GenAI workflow: Overview, Traces, Sessions tabs
- "Start Demo" button for sample data

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│              OpenShift Project (namespace)                         │
│                                                                    │
│  ┌──────────────┐    ┌────────────────────────────────────────┐   │
│  │   Workbench  │───▶│  MLflow Tracking Server (3.13.0)       │   │
│  │  (Notebook)  │    │                                        │   │
│  └──────┬───────┘    │  ┌──────────┐  ┌────────────────────┐ │   │
│         │            │  │  DB      │  │   S3 Artifact      │ │   │
│         │            │  │ (PG /    │  │   Store            │ │   │
│  ┌──────▼───────┐    │  │  SQLite) │  │                    │ │   │
│  │ RHOAI        │    │  └──────────┘  └────────────────────┘ │   │
│  │ Dashboard    │    └────────────────────────────────────────┘   │
│  └──────────────┘                                                  │
│         │                                                          │
│  ┌──────▼──────────────────────────────────────────────────────┐  │
│  │  Nemotron InferenceService (KServe / vLLM)                   │  │
│  │  Route: /v1/chat/completions  or  /v2/models/{model}/infer  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Kubernetes RBAC (SelfSubjectAccessReview) gates every            │
│  MLflow API call and every Nemotron inference call                │
└────────────────────────────────────────────────────────────────────┘
```

## Running the Demo

```bash
# 0. [Admin] Deploy MLflow Minimal server — run once per namespace
jupyter notebook notebooks/00_cluster_setup.ipynb

# 1. Verify SDK and connectivity
jupyter notebook notebooks/01_setup_authentication.ipynb

# 2. Core experiment tracking
jupyter notebook notebooks/02_experiment_tracking.ipynb

# 3. Hyperparameter tuning with run comparison
jupyter notebook notebooks/03_hyperparameter_tuning.ipynb

# 4. GenAI / LLM tracing (mock LLM — no external endpoint needed)
jupyter notebook notebooks/04_genai_tracing.ipynb

# 5. Model registry
jupyter notebook notebooks/05_model_registry.ipynb

# 6. Nemotron live inference tracing (requires Nemotron InferenceService)
jupyter notebook notebooks/06_nemotron_tracing.ipynb

# 7. Nemotron RAG pipeline + systematic evaluation (requires Nemotron InferenceService)
jupyter notebook notebooks/07_nemotron_rag_eval.ipynb
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 Forbidden` on MLflow API | Missing RBAC | Ensure workbench annotation or `mlflow-view` ClusterRole binding |
| `Workspace not found` | Wrong namespace | Check `MLFLOW_WORKSPACE` matches your OpenShift project name |
| `JSON decode error` | SDK version mismatch | Upgrade: `pip install "mlflow[kubernetes]>=3.11"` |
| Artifact upload fails | S3 not configured | Use `MLflowConfig` CR + `mlflow-artifact-connection` secret |
| Notebook won't start | Missing annotation webhook | Add `opendatahub.io/mlflow-instance` only when notebook is stopped |
| `UntrustedTypesFoundException` | MLflow 3.13 skops default | Add `serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE` to `log_model` |
| `AttributeError: set_trace_tags` | Function removed in MLflow 3.13 | Use `span.set_attribute("mlflow.trace.session_id", ...)` inside `mlflow.start_span` |
| `404 Not Found` on Nemotron route | Wrong model name in request body | Run the endpoint probe cell (Section 1 of nb06/nb07); check model name with `oc get inferenceservice` |
| `401/403` on Nemotron calls | Invalid or missing bearer token | Verify `OC_TOKEN` env var or SA token at `/var/run/secrets/kubernetes.io/serviceaccount/token` |
| `None` content in Nemotron response | Model returned `content: null` (e.g. reasoning step) | Already handled: `choice['message'].get('content') or ''` in all generate functions |
| Nemotron endpoint probe: both 404 | Route path differs from standard | Run `oc get route -n <namespace>` and update `LLM_BASE_URL` manually |

# Runner Wrapper

`runner_wrapper/` is a copyable HTTP wrapper for turning a model repo into a SceneGenDeployBench runner image. In this repo it is a test runner. In a model repo, keep `server.py` and replace the test logic in `adapter.py`.

For AI agents adapting a copied folder, read `AGENTS.md`; it is the self-contained contract note.

## What To Build

One image implements one role:

- `generator`: consumes dataset files and returns reusable generated artifacts.
- `evaluator`: consumes dataset files and/or generated artifacts, then returns scalar metrics and optional report/preview/log artifacts.

Do not build a hybrid runner. The catalog `kind` must be `generator` or `evaluator`; the orchestrator injects matching `RUNNER_TYPE` during real runs.

## Files

```text
runner_wrapper/
  AGENTS.md                         # adaptation instructions for agents
  adapter.py                        # replace this in model repos
  job_logging.py                    # tee model output to Docker and runner.log
  measurements.py                   # lightweight container resource metrics
  server.py                         # stable HTTP runner API
  Dockerfile                        # builds from repo root
  localtest.sh                      # local build/run/smoke helper
  docs/                             # runner-facing input notes
  examples/
```

Create these in the target model repo:

- `runner_wrapper/config/runners/<runner>.yaml`
- `.dockerignore`
- `.github/workflows/runner-image.yaml`

## Upstream Subtree

Add the upstream once in the target model repo:

```bash
git remote add deploybench https://github.com/CashewHero/SceneGenDeployBench.git
git fetch deploybench subtree/runner_wrapper
git subtree add --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

Pull later upstream updates into `runner_wrapper/`:

```bash
git fetch deploybench subtree/runner_wrapper
git subtree pull --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

## HTTP API

Endpoints:

- `GET /status`
- `POST /run-job`
- `POST /shutdown`

States:

- `starting`, `idle`, `running`, `finished`, `failed`, `shutting_down`

`POST /run-job` accepts or rejects a job immediately. If accepted, work happens in the background. Poll `GET /status` until `state` is `finished` or `failed`; then read `result`.

## Adapter Contract

`adapter.py` must expose:

```python
def run_job(job_request: dict) -> dict: ...
```

Use:

- `sample.data`: original dataset sample used by the source job
- `sample.output`: reusable artifacts returned by the selected source job
- `sample.references`: other samples from the same dataset subset when the evaluator requires references
- `sample.metadata`: inherited dataset/subset/sample metadata
- `job.parameters`: runner-specific catalog defaults merged with per-job `--set key=value` overrides
- [Camera pose inputs](docs/camera_pose.md): how runners should read `camera_pose`
- [Camera trajectory inputs](docs/camera_trajectory.md): how runners should read `camera_trajectory`
- `runtime.output_dir`: durable output root
- `runtime.model_cache_dir`: reusable model assets for this runner
- `runtime.temp_dir`: scratch root
- `runtime.device`: requested device, such as `cuda:0`
- `config.inputs`: the normalized catalog input requirements for `data`, `output`, and `references`

Return:

```json
{
  "status": "completed",
  "started_at": "2026-04-18T10:00:00Z",
  "completed_at": "2026-04-18T10:07:31Z",
  "metrics": [],
  "artifacts": [],
  "failure": null
}
```

Rules:

- validate required keys against `sample.data`, `sample.output`, and each item in `sample.references`
- write job outputs only under `runtime.output_dir`
- use `runtime.model_cache_dir` only for reusable model assets
- write `runner.log` directly in `runtime.output_dir` and flush progress while the job runs
- write `metrics.json` directly in `runtime.output_dir`
- write output files directly in `runtime.output_dir`, for example `3DGS.ply`
- return artifact paths relative to `runtime.output_dir`
- put evaluator scores in `metrics`
- put files in `artifacts`
- never write directly to PostgreSQL

## Measurements

Report these standard per-job metrics when available:

```text
resources.cpu_time_ms
resources.peak_memory_bytes
resources.disk_read_bytes
resources.disk_write_bytes
resources.disk_read_ops
resources.disk_write_ops
resources.input_total_bytes
resources.output_total_bytes
resources.gpu_peak_memory_bytes
gpu.device_memory_total_bytes
performance.wall_time_ms
```

Optional when easy and honest:

```text
model.estimated_ops
model.inference_steps
gpu.energy_joules
gpu.compute_time_ms
```

`measurements.py` provides `ResourceMonitor` for the standard container metrics. Omit metrics that cannot be measured; do not return guessed zeroes.

## Data Type Alignment

This is the main compatibility rule:

```text
catalog inputs.data -> sample.data
catalog inputs.output -> sample.output
catalog inputs.references -> sample.references[].data
generator artifact data_type -> future evaluator sample.output key
```

Generator outputs are reusable only when `artifact_type` is `model_output`, `generated_output`, or `output`. Set `data_type` to the semantic key a downstream evaluator will require, for example `scene`, `mesh`, `image`, `depth`, or `point_cloud`.

Each output may include an `inputs` list recording exactly what produced it. Use `path` for file or directory inputs and `value` for structured or scalar inputs:

```json
{
  "data_type": "scene",
  "path": "scene.ply",
  "inputs": [
    {"source": "sample.data", "data_type": "image", "path": "/data/datasets/example.png"},
    {"source": "sample.data", "data_type": "camera_pose", "value": {"position": [0, 0, 0]}}
  ]
}
```

The same entries appear under `output_files` at the top of `metrics.json`. Outputs that use different inputs should carry different `inputs` lists.

Return `runner.log`, `metrics.json`, and output files in `result.artifacts`. The orchestrator records only returned artifacts; it does not scan the output directory.

## Build And Smoke

Build from the model repo root:

```bash
docker build -f runner_wrapper/Dockerfile -t my-model-runner .
```

Use the helper:

```bash
runner_wrapper/localtest.sh build
runner_wrapper/localtest.sh smoke
```

For Pano2Room, full smoke also needs the three checkpoint files mounted with
`RUNNER_WEIGHTS_DIR`; the helper mounts it at
`/data/model_cache/pano2room/checkpoints`. Stable Diffusion inpainting
must either be available via `HF_TOKEN` or supplied as a container-visible local
model path through `PANO2ROOM_HF_STABLE_DIFFUSION_MODEL`.

For the bundled test adapter only:

```bash
TEST_RUNNER_MIN_SECONDS=0 TEST_RUNNER_MAX_SECONDS=0 runner_wrapper/localtest.sh smoke
```

Manual run:

```bash
docker run --rm -p 58090:58090 \
  -e RUNNER_NAME=my-generator \
  -e RUNNER_TYPE=generator \
  -e RUNNER_VERSION=0.1.0 \
  -v "$PWD/data:/data" \
  my-model-runner
```

Submit `runner_wrapper/examples/generator_job_request.json` or `runner_wrapper/examples/evaluator_job_request.json` to `POST /run-job`, then poll `GET /status`.

## Runner Catalog

Create one catalog config:

```bash
mkdir -p runner_wrapper/config/runners
```

Start from the matching example:

```text
runner_wrapper/examples/generator_runner_catalog.example.yaml
runner_wrapper/examples/evaluator_runner_catalog.example.yaml
```

Set `runner`, `version`, `kind`, `inputs`, `launcher.image`, and `launcher.endpoint.port`. Add `launcher.env` for runner-specific runtime config, such as model mode, checkpoint selector, thresholds, backend flags, API endpoints, cache locations, or weight/config paths. Add `launcher.env_passthrough` for optional host/orchestrator variables such as tokens. Copy the finished YAML into the orchestrator repo's `config/runners/` directory.

If an env value is a path, it must be visible inside the runner container. For Docker runs, shared paths usually come from mounted dataset/output directories.

## GitHub Actions

Install the image-build template in the target model repo:

```bash
mkdir -p .github/workflows
cp runner_wrapper/examples/github-workflows/build-runner-image.yaml \
  .github/workflows/runner-image.yaml
```

The workflow derives the image name from the repository name. The target repo should be named `SceneGenDeployBench-<model>`. It builds from repo root with `runner_wrapper/Dockerfile`, pushes to GHCR on branch/tag pushes, and builds pull requests without pushing.

## Environment

- `RUNNER_PORT=58090`
- `RUNNER_NAME`
- `RUNNER_TYPE`
- `RUNNER_VERSION`
- `RUNNER_CONTRACT_VERSION=1`
- `RUNNER_ADAPTER=runner_wrapper.adapter:run_job`
- `RUNNER_LOG_LEVEL=INFO`
- `RUNNER_IDLE_TIMEOUT_SECONDS=900`
- `RUNNER_STARTUP_TIMEOUT_SECONDS=60`

Timeout guards:

- Startup: `RUNNER_STARTUP_TIMEOUT_SECONDS` is normally `scheduling.startup_timeout_minutes + 1 minute`; the wrapper logs `runner_startup_timeout` and exits when reached.
- Job: each accepted request includes `job.timeout_seconds`; if the model is still running after `job.timeout_seconds + 60`, the wrapper logs `runner_job_timeout`, terminates the job process, and marks the job failed.

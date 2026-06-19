# Runner Adaptation

This folder is meant to be copied into a model repository. Assume no other
SceneGenDeployBench files are available. Your task is to wrap the existing
model as one HTTP runner image that the orchestrator can launch.

Wrap the model repo; do not redesign it. Prefer changes in `runner_wrapper/`,
Docker/package wiring, config, and small launch scripts. Patch original model
source only when unavoidable, and document why.

## Choose One Role

Build exactly one role per image:

- `generator`: consumes dataset inputs and returns reusable generated outputs.
- `evaluator`: consumes dataset inputs and/or generated outputs, then returns
  scalar metrics plus optional reports, previews, or logs.

Set the catalog `kind` to the chosen role. In orchestrated runs, `RUNNER_TYPE`
is injected from that catalog value. For manual runs, set `RUNNER_TYPE` to the
same value. Do not implement a hybrid runner.

## Files To Edit

Usually edit or create only:

- `runner_wrapper/adapter.py`
- `runner_wrapper/measurements.py`
- `runner_wrapper/Dockerfile`
- `runner_wrapper/config/runners/<runner>.yaml`
- `.dockerignore`
- `.github/workflows/runner-image.yaml`, copied from the template

Keep `runner_wrapper/server.py` stable unless the HTTP contract changes.

## Orchestrator Contract

The orchestrator owns scheduling, retries, persistence, and database writes.
The runner executes one request at a time and reports the terminal result
through `GET /status`.

HTTP endpoints exposed by `server.py`:

- `GET /status`
- `POST /run-job`
- `POST /shutdown`

Runner states:

- `starting`, `idle`, `running`, `finished`, `failed`, `shutting_down`

`POST /run-job` accepts a JSON request, starts work in a background thread, and
returns `accepted: true` when the runner is available. Poll `GET /status` until
`state` is `finished` or `failed`; then read `result`.

## Adapter Contract

`adapter.py` must expose:

```python
def run_job(job_request: dict) -> dict:
    ...
```

Use these request fields:

- `job.job_id`, `job.batch_id`, `job.timeout_seconds`
- `sample.data`: mapping from semantic data type to readable file path
- `sample.metadata`: optional dataset or upstream-run metadata
- `runtime.output_dir`: durable output root for this job
- `runtime.temp_dir`: scratch space for this job
- `runtime.device`: requested device string, for example `cuda:0`
- `config.required_data_types`: catalog-required `sample.data` keys

Adapter rules:

- validate every `config.required_data_types` key exists in `sample.data`
- read inputs only from `sample.data` paths or model assets in the image
- write durable files only under `runtime.output_dir`
- use `runtime.temp_dir` for scratch files
- do not write to PostgreSQL or assume an orchestrator source tree is present
- return artifact paths relative to `runtime.output_dir`
- return `status: "completed"` or `status: "failed"`

Return shape:

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

For handled failures, return `status: "failed"` and:

```json
{
  "code": "MODEL_ERROR",
  "message": "short reason",
  "retryable": false,
  "stage": "adapter"
}
```

Uncaught exceptions are converted by `server.py` into runner failures.

## Line-Up Rule

Names must line up across the catalog, request, and artifacts:

1. Catalog `inputs.required` becomes `config.required_data_types`.
2. Each required input must be present as a key in `sample.data`.
3. A generator's reusable artifact `data_type` becomes a future evaluator's
   `sample.data` key.
4. An evaluator catalog must require the same `data_type` keys it expects.

Choose semantic data types from the model and benchmark domain, not from local
variable names. Good examples: `image`, `depth`, `camera_pose`, `scene`,
`mesh`, `point_cloud`, `caption`. Keep them stable across versions unless the
contract intentionally changes.

## Outputs

Generator reusable outputs must be artifacts with one of these types:

- `model_output`
- `generated_output`
- `output`

Reusable output artifact fields:

- `artifact_type`: one of the reusable types above
- `data_type`: semantic key for downstream evaluators
- `path`: path relative to `runtime.output_dir`
- `format`: file format when known, for example `glb`, `obj`, `png`, `json`
- `metadata`: optional small JSON object

Evaluator scores belong in `metrics`; reports, previews, summaries, and logs
belong in `artifacts` with non-reusable types such as `report`, `preview`,
`diagnostic`, `metric_summary`, or `job_log`.

Metric fields:

- `namespace`: group, for example `quality`, `geometry`, `performance`
- `name`: stable metric name
- `type`: `float`, `integer`, `boolean`, or `string`
- `value`: scalar value
- `unit`: optional unit or scale
- `source`: `runner`, `model`, or `evaluator`

The orchestrator records only artifacts and metrics returned in the result. It
does not scan the output directory.

## Measurements

Use `ResourceMonitor` from `runner_wrapper.measurements` around the job body
and report these standard per-job metrics when available:

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

Optional when the wrapped repo exposes them cleanly:

```text
model.estimated_ops
model.inference_steps
gpu.energy_joules
gpu.compute_time_ms
```

Omit metrics that cannot be measured. Do not return guessed zeroes. Keep
model-specific quality/evaluator scores separate from these resource metrics.

## Catalog

Create one catalog YAML under `runner_wrapper/config/runners/` using the
matching example as a starting point:

```bash
mkdir -p runner_wrapper/config/runners
```

```text
runner_wrapper/examples/generator_runner_catalog.example.yaml
runner_wrapper/examples/evaluator_runner_catalog.example.yaml
```

Set:

- `runner`: stable runner name
- `version`: runner contract/image version
- `kind`: `generator` or `evaluator`
- `inputs.required` and `inputs.optional`: semantic data type keys
- `launcher.image`: image tag the orchestrator can pull or run
- `launcher.endpoint.port`: container port used by `RUNNER_PORT`
- `launcher.env`: optional runner-specific runtime config, such as model mode,
  checkpoint selector, thresholds, backend flags, credentials, API endpoints,
  cache locations, or weight/config paths

The orchestrator injects `RUNNER_NAME`, `RUNNER_TYPE`, `RUNNER_VERSION`,
`RUNNER_CONTRACT_VERSION`, and `RUNNER_PORT` from the catalog.
`RUNNER_ADAPTER` defaults to `runner_wrapper.adapter:run_job`; set it in
`launcher.env` only when the callable lives somewhere else.
If an env value is a path, make sure that path exists in the image, is created
or downloaded by startup code, or is provided by deployment-specific mounting.

## Build And Smoke

Adaptation workflow:

1. Inspect the model entry points, dependencies, weights, inputs, and outputs.
2. Choose exactly one role and the stable input/output data type names.
3. Replace the test adapter logic in `runner_wrapper/adapter.py`.
4. Update `runner_wrapper/Dockerfile` for dependencies, package install, model
   assets, and weight handling.
5. Add `.dockerignore` entries for datasets, outputs, caches, and local weights
   that should not be baked into the image.
6. Create one catalog YAML in `runner_wrapper/config/runners/`.
7. Build and smoke test.

Build from the model repo root:

```bash
docker build -f runner_wrapper/Dockerfile -t my-model-runner .
```

Use the helper:

```bash
runner_wrapper/localtest.sh build
runner_wrapper/localtest.sh smoke
```

Manual run:

```bash
docker run --rm -p 8080:8080 \
  -e RUNNER_NAME=my-model \
  -e RUNNER_TYPE=generator \
  -e RUNNER_VERSION=0.1.0 \
  -v "$PWD/data:/data" \
  my-model-runner
```

Submit a smoke request and poll status:

```bash
curl -sS -X POST http://127.0.0.1:8080/run-job \
  -H 'Content-Type: application/json' \
  --data @runner_wrapper/examples/generator_job_request.json

curl -sS http://127.0.0.1:8080/status
```

Install the image-build workflow:

```bash
mkdir -p .github/workflows
cp runner_wrapper/examples/github-workflows/build-runner-image.yaml \
  .github/workflows/runner-image.yaml
```

## Final Checklist

- one image implements one role only
- `GET /status` returns `state: "idle"` before a job
- `POST /run-job` returns `accepted: true`
- terminal `result.status` is `completed` or `failed`
- catalog `kind`, image, version, port, and input keys are correct
- adapter validates `config.required_data_types`
- all durable outputs are listed in `artifacts`
- generator reusable artifacts use stable `data_type` keys
- evaluator metrics are scalar and stable
- standard resource measurements are reported when available
- artifact paths are relative to `runtime.output_dir`
- `.github/workflows/runner-image.yaml` builds the image
- `runner_wrapper/config/runners/<runner>.yaml` is ready to copy into the
  orchestrator repo's `config/runners/` directory

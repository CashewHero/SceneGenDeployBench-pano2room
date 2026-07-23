# Runner Adaptation Instructions

These instructions are for coding agents adapting this wrapper inside a model repository. Read `README.md` for the human-facing project overview and commands; do not duplicate those sections here.

## Scope And Boundaries

Verify the repository naming requirement documented in the README before editing. Stop and notify the user if it is not satisfied.

Wrap the existing model rather than redesigning it. Prefer changes in `runner_wrapper/`, dependency and Docker wiring, catalog configuration, and small launch scripts. Modify original model code only when integration cannot be achieved cleanly from the wrapper, and explain why in the handoff.

Follow the README's one-role-per-image rule. Do not add runtime switches that create a hybrid runner.

Reuse `job_logging.py` and `measurements.py`. Keep `server.py` stable unless the shared HTTP contract itself changes.

Do not assume that an orchestrator source tree exists in the target repository. Do not write to PostgreSQL. Do not put private credentials, datasets, caches, or local model weights in the image or repository.

## Inspect Before Editing

Identify:

1. the real inference entry point;
2. required Python, system, CUDA, and model dependencies;
3. how weights are found or downloaded;
4. accepted model inputs and their coordinate/projection assumptions;
5. generated files or evaluator scores;
6. the smallest realistic smoke input.

Choose stable semantic data types from the benchmark domain, such as `image`, `depth`, `camera_pose`, `camera_trajectory`, `3dgs`, `mesh`, `scene`, or `point_cloud`. Do not use model-local variable names as contract types.

## Adaptation Sequence

1. Choose one runner role and its semantic input and output types.
2. Replace the bundled test logic in `adapter.py` with the smallest model-specific integration.
3. Update `Dockerfile` and the repository dependencies for the model. Copy `examples/dockerignore.example` to the repository root as `.dockerignore` if an equivalent file is not already present.
4. Update the matching request example into a realistic smoke request. Copy the matching catalog example to `runner_wrapper/config/runners/<runner>.yaml`, then make both agree with the adapter.
5. Add a short note to the model repository's main README naming the runner role, semantic inputs/outputs, and where to find the wrapper instructions.
6. Add the image workflow using the README command when the repository will publish through GitHub Actions.
7. Run the unit, build, and smoke checks before handoff.

The bundled adapter and examples are contract fixtures, not a complete model implementation. The test adapter supports both fixture roles only so this shared wrapper can be tested; a model adapter must keep one role. Replace placeholder names, image tags, paths, data types, generated files, and evaluator metrics. Preserve the request, result, server, and measurement contracts described below.

## Request Contract

`adapter.py` exposes:

```python
def run_job(job_request: dict) -> dict:
    ...
```

The request contains only:

- `contract_version`
- `job`
- `inputs`
- `runtime`

Relevant job fields:

- `job.job_id`
- `job.batch_id`
- `job.job_type`
- `job.primary_sample`: the input sample that owns the job
- `job.primary_sample_metadata`: inherited dataset metadata, omitted when empty
- `job.source_job_id`: upstream job identity when applicable
- `job.attempt`
- `job.timeout_seconds`
- `job.parameters`: catalog defaults merged with per-job overrides

The orchestrator validates catalog requirements before dispatch. A runner request does not contain the catalog contract or a root `config` object. Adapter validation should cover operational assumptions such as readable files, supported formats, and model constraints.

### Shared Server Lifecycle

Do not replace the shared lifecycle when adapting a model. `server.py` exposes:

- `GET /status`
- `POST /run-job`
- `POST /shutdown`

States are `starting`, `idle`, `running`, `finished`, `failed`, and `shutting_down`. The server accepts one job at a time and binds to the first accepted `job.batch_id`; another batch is rejected until the runner container is replaced.

`job.timeout_seconds` is required. The adapter runs in a child process, and the server terminates it if it is still running after `job.timeout_seconds + 60` seconds. The startup watchdog uses `RUNNER_STARTUP_TIMEOUT_SECONDS`, normally the catalog startup timeout plus one minute.

### Inputs

Every input role has the same shape:

```text
inputs -> role -> sample_id -> data_type -> data
```

Roles:

- `inputs.data`: original dataset samples
- `inputs.output`: reusable files produced by the selected upstream runner, normally consumed by an evaluator
- `inputs.references`: additional dataset or runner-output samples selected with `--reference`

`inputs.output` is an input to the current job. It does not describe files being produced by the current runner.

File and directory data are absolute paths. Structured types such as `camera_pose` remain JSON values. Empty roles are omitted. Read `docs/camera_pose.md` and `docs/camera_trajectory.md` when those types are used.

### Runtime And Environment

`runtime.output_dir` is the only per-job runtime path. Write every durable job file below it.

Runner containers receive shared roots through:

- `PATH_DATASETS`
- `PATH_MODEL_CACHE`
- `PATH_OUTPUT`

Use `PATH_MODEL_CACHE` for reusable downloaded assets. Derive temporary job space from `job.job_id` under `/tmp`. Device access and runner-specific paths or flags belong in container/catalog environment configuration.

The orchestrator also injects `RUNNER_PORT`, `RUNNER_NAME`, `RUNNER_TYPE`, `RUNNER_VERSION`, `RUNNER_CONTRACT_VERSION`, and `RUNNER_STARTUP_TIMEOUT_SECONDS`. The image configures `RUNNER_ADAPTER` and `RUNNER_IDLE_TIMEOUT_SECONDS`; the latter shuts down a non-running server after it stops receiving status traffic for that interval. Use these values through `server.py`; do not hardcode a second identity or port in the adapter.

## Result Contract

Return a terminal result:

```json
{
  "status": "completed",
  "started_at": "2026-04-18T10:00:00Z",
  "completed_at": "2026-04-18T10:07:31Z",
  "output_files": {
    "sample-1": {
      "3dgs": "3DGS.ply"
    }
  },
  "metrics": [],
  "artifacts": [
    {"artifact_type": "job_log", "path": "runner.log"},
    {"artifact_type": "metric_summary", "path": "metrics.json"}
  ],
  "failure": null
}
```

Rules:

- `output_files` contains only reusable outputs from the current runner.
- Its shape is `sample_id -> data_type -> relative path`.
- Construct the mapping once, write the same mapping near the top of `metrics.json`, and return it in the result.
- Omit `output_files` when no reusable files were produced.
- `artifacts` contains administrative or diagnostic files, not reusable model outputs.
- Artifact and output paths are relative to `runtime.output_dir`.
- `metrics` contains evaluator scores and standard measurements.
- Do not scan the output directory to infer the result; report files explicitly.

Keep the output folder flat and readable where practical: `runner.log`, `metrics.json`, and actual output files.

Write `metrics.json` for human inspection with this top-level order:

1. `inputs`, copied from the normalized request inputs;
2. `output_files`, when present, using the same object returned in the result;
3. non-empty `parameters`;
4. evaluator `metrics` and standard `resource_metrics`, when present.

For handled failures, return the normal terminal result shape with `status: "failed"` and a populated `failure` field:

```json
{
  "status": "failed",
  "started_at": "2026-04-18T10:00:00Z",
  "completed_at": "2026-04-18T10:00:05Z",
  "metrics": [],
  "artifacts": [
    {"artifact_type": "job_log", "path": "runner.log"}
  ],
  "failure": {
    "code": "MODEL_ERROR",
    "message": "short reason",
    "retryable": false,
    "stage": "adapter"
  }
}
```

Uncaught exceptions are converted by `server.py` into runner failures.

## Metrics And Logging

Use `ResourceMonitor` around the model job and report available standard measurements:

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

Optional model measurements may include `model.estimated_ops`, `model.inference_steps`, `gpu.energy_joules`, or `gpu.compute_time_ms`. Omit values that cannot be measured; never report guessed zeroes.

Evaluator metric entries use stable `namespace`, `name`, `type`, `value`, optional `unit`, and `source` fields. Metric `type` is `float`, `integer`, `boolean`, or `string`; `source` is normally `runner`, `model`, or `evaluator`. Keep evaluator quality scores separate from resource metrics.

Use `tee_job_output` so model stdout and stderr reach both Docker logs and `runner.log`. Preserve exceptions and useful progress while avoiding high-frequency progress-bar noise.

## Catalog Alignment

Create one YAML under `runner_wrapper/config/runners/` from the matching example. This is the runner's distributable catalog; a deployment copies it into its active runner-config directory. Do not make the model repository depend on an orchestrator checkout. Ensure these fields match the adapter:

- required identity: `runner`, `version`, `display_name`, and `kind`
- version selection: `latest` and `contract_version`
- `inputs.data`
- `inputs.output`
- `inputs.references`
- `job_parameters`
- `launcher.driver` and `launcher.compat_version`
- `launcher.image` for the Docker driver
- `launcher.endpoint.port`
- `launcher.env` and `launcher.env_passthrough`
- optional Docker settings such as `launcher.gpus` and `launcher.user`
- `scheduling.max_batch_size`, `max_attempts`, `job_timeout_minutes`, and `startup_timeout_minutes`

Catalog input config example:

```yaml
inputs:
  data:
    required_sample:
      required_datatype: [image]
      optional_datatype: [camera_pose]
  output:
    required_sample:
      required_datatype: [3dgs]
  references:
    optional_sample:
      required_datatype: [image, camera_pose]
```

Set `kind` explicitly to one supported role. Mark exactly one version of a runner name as `latest` when multiple versions are present. Keep `contract_version` aligned with the wrapper server.

Names must line up exactly:

```text
catalog inputs.data       -> request inputs.data
catalog inputs.output     -> request inputs.output
catalog inputs.references -> request inputs.references
producer output_files     -> downstream evaluator inputs.output
```

The semantic data type reported by a producer must be the type required by its consumer.
The catalog describes inputs; a producer reports the outputs it actually created through each job result's `output_files` mapping.

## Docker Adaptation

Keep heavyweight or mutable weights in the shared model cache unless licensing or reproducibility requires image-bundled public assets. Make automatic downloads concurrency-safe and deterministic. Document required tokens without committing them.

Use the catalog `launcher.env` for model mode, checkpoint selection, thresholds, backend flags, and paths. Use `env_passthrough` only for values supplied by deployment, such as credentials.

The HTTP process must become ready without loading per-job inputs. Load expensive reusable model state at startup only when that model benefits from it and failures remain clear.

## Verification

Run the build and smoke commands documented in the README, then verify:

- the server reaches `idle`;
- a realistic small request is accepted;
- the job reaches `finished` or returns a useful failure;
- `runner.log` and `metrics.json` exist;
- reusable files appear only in `output_files` with correct semantic types;
- evaluator metrics are scalar and stable;
- paths work inside the container without host-only assumptions;
- the catalog image, port, role, types, and environment match the built image.

## Handoff

Report:

- the selected role and semantic input/output types;
- required model assets and environment variables;
- build and smoke results;
- any changes outside `runner_wrapper/` and why they were necessary;
- any remaining limitation that affects real benchmark runs.

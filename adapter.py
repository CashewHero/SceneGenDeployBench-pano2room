from __future__ import annotations

"""Default adapter implementation.

Replace this file or point RUNNER_ADAPTER at a different callable inside the
model repository.
"""

import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapter")


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def _safe_role(role: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in role)


def _normalize_sample_data(sample: dict[str, Any]) -> dict[str, str]:
    sample_data = sample.get("data")
    if isinstance(sample_data, dict) and sample_data:
        normalized: dict[str, str] = {}
        for data_type, raw_path in sample_data.items():
            data_key = str(data_type).strip()
            if not data_key:
                raise ValueError("sample data type names must be non-empty")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"sample data path for {data_key} must be a non-empty string")
            normalized[data_key] = raw_path.strip()
        return normalized

    normalized = {}
    for index, item in enumerate(sample.get("inputs", [])):
        if not isinstance(item, dict):
            raise ValueError("legacy sample inputs must be objects")
        data_type = str(item.get("role", f"input_{index}")).strip()
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample input path for {data_type} must be a non-empty string")
        normalized[data_type] = raw_path.strip()
    return normalized


def _validate_required_data_types(
    sample_data: dict[str, str],
    required_data_types: list[str],
    job_id: str,
) -> None:
    missing_data_types = [data_type for data_type in required_data_types if data_type not in sample_data]
    if not missing_data_types:
        return

    logger.error(
        event_message(
            "adapter_input_validation_failed",
            job_id=job_id,
            missing_data_types=missing_data_types,
        )
    )
    raise ValueError(f"sample missing required data types: {', '.join(missing_data_types)}")


def _copy_inputs(sample_data: dict[str, str], output_root: Path, *, model_outputs: bool) -> list[dict[str, Any]]:
    output_dir = output_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for index, (data_type, raw_path) in enumerate(sample_data.items()):
        src_path = Path(raw_path)
        if not src_path.exists() or not src_path.is_file():
            raise FileNotFoundError(f"input file not found: {src_path}")

        suffix = src_path.suffix
        dst_name = f"input_{index:02d}_{_safe_role(data_type)}{suffix}"
        dst_path = output_dir / dst_name
        shutil.copy2(src_path, dst_path)

        artifacts.append(
            {
                "artifact_type": "model_output" if model_outputs else "diagnostic",
                "role": "primary" if model_outputs and index == 0 else data_type,
                "data_type": data_type,
                "path": str(dst_path.relative_to(output_root)),
                "format": suffix.lstrip(".") or "bin",
                "size_bytes": dst_path.stat().st_size,
                "metadata": {
                    "source_path": str(src_path),
                },
            }
        )

    return artifacts


def _sleep_range_seconds() -> int:
    min_seconds = int(os.getenv("TEST_RUNNER_MIN_SECONDS", "360"))
    max_seconds = int(os.getenv("TEST_RUNNER_MAX_SECONDS", "720"))
    if min_seconds < 0 or max_seconds < min_seconds:
        raise ValueError("invalid TEST_RUNNER_MIN_SECONDS / TEST_RUNNER_MAX_SECONDS")
    return random.randint(min_seconds, max_seconds)


def _write_summary(metrics_dir: Path, summary: dict[str, Any]) -> None:
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _is_evaluator_mode() -> bool:
    runner_type = os.getenv("RUNNER_TYPE", "generator").strip().lower()
    mode = os.getenv("TEST_RUNNER_MODE", "").strip().lower()
    return runner_type == "evaluator" or mode == "evaluator"


def _random_evaluation_metrics() -> list[dict[str, Any]]:
    return [
        {
            "namespace": "quality",
            "name": "test_quality_score",
            "type": "float",
            "value": round(random.uniform(0.0, 1.0), 6),
            "unit": "score",
            "source": "evaluator",
        },
        {
            "namespace": "quality",
            "name": "test_geometry_error",
            "type": "float",
            "value": round(random.uniform(0.0, 0.25), 6),
            "unit": "normalized_error",
            "source": "evaluator",
        },
    ]


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    job = job_request["job"]
    runtime = job_request["runtime"]
    sample = job_request["sample"]
    sample_data = _normalize_sample_data(sample)
    output_root = Path(runtime["output_dir"])
    monitor = ResourceMonitor(sample_data=sample_data, output_dir=output_root)
    monitor.start()

    try:
        logger.info(
            event_message(
                "adapter_run_started",
                job_id=job["job_id"],
                batch_id=job.get("batch_id"),
                output_dir=runtime["output_dir"],
                input_data_types=sorted(sample_data),
            )
        )

        required_data_types = job_request.get("config", {}).get("required_data_types", [])
        _validate_required_data_types(sample_data, required_data_types, job["job_id"])

        output_root.mkdir(parents=True, exist_ok=True)

        logs_dir = output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        sleep_seconds = _sleep_range_seconds()
        logger.info(event_message("adapter_sleeping", job_id=job["job_id"], sleep_seconds=sleep_seconds))
        with (logs_dir / "runner.log").open("a", encoding="utf-8") as handle:
            handle.write(f"test runner sleeping for {sleep_seconds} seconds\n")

        time.sleep(sleep_seconds)
        evaluator_mode = _is_evaluator_mode()
        artifacts = _copy_inputs(sample_data, output_root, model_outputs=not evaluator_mode)
        logger.info(
            event_message(
                "adapter_inputs_copied",
                job_id=job["job_id"],
                copied_input_count=len(artifacts),
            )
        )

        copied_input_count = len(artifacts)
        evaluation_metrics = _random_evaluation_metrics() if evaluator_mode else []

        _write_summary(
            metrics_dir,
            {
                "sleep_seconds": sleep_seconds,
                "copied_input_count": copied_input_count,
                "evaluation_metrics": evaluation_metrics,
            },
        )

        resource_metrics = monitor.stop()
        metrics = resource_metrics + evaluation_metrics
        completed_at = time.time()
        wall_time_ms = round((completed_at - started_at) * 1000, 3)

        logger.info(
            event_message(
                "adapter_run_completed",
                job_id=job["job_id"],
                wall_time_ms=wall_time_ms,
                copied_input_count=copied_input_count,
            )
        )

        return {
            "status": "completed",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
            "metrics": metrics,
            "artifacts": artifacts
            + [
                {
                    "artifact_type": "job_log",
                    "role": "stdout",
                    "path": "logs/runner.log",
                    "format": "text",
                },
                {
                    "artifact_type": "metric_summary",
                    "role": "summary",
                    "path": "metrics/summary.json",
                    "format": "json",
                },
            ],
            "failure": None,
        }
    except Exception:
        monitor.stop()
        raise

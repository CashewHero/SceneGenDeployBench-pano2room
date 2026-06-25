from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapter")

RUNNER_NAME = "pano2room"
OUTPUT_FILENAME = "3DGS.ply"
DEFAULT_CHECKPOINT_DIR = "/models/pano2room/checkpoints"
DEFAULT_CAMERA_TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "input" / "Camera_Trajectory"
CAMERA_TRAJECTORY_DATA_KEYS = ("camera_trajectory", "camera_trajectory_dir")

CHECKPOINT_DEFAULTS = {
    "PANO2ROOM_LAMA_CONFIG_PATH": "big-lama-config.yaml",
    "PANO2ROOM_LAMA_CKPT_PATH": "big-lama.ckpt",
    "PANO2ROOM_OMNIDATA_DEPTH_CKPT_PATH": "omnidata_dpt_depth_v2.ckpt",
    "PANO2ROOM_OMNIDATA_NORMAL_CKPT_PATH": "omnidata_dpt_normal_v2.ckpt",
}

PANO2ROOM_WEIGHT_DOWNLOADS = {
    "PANO2ROOM_LAMA_CKPT_PATH": "https://drive.google.com/uc?id=1H5CHOsm_yAxZI9a5hv9tyZmh5CMjJxap",
    "PANO2ROOM_OMNIDATA_DEPTH_CKPT_PATH": "https://drive.google.com/uc?id=18S9ycwHi07hzPdLORsAQFFORTeovdo4E",
    "PANO2ROOM_OMNIDATA_NORMAL_CKPT_PATH": "https://drive.google.com/uc?id=1gMBrl51AZZr6ANy8d77KFXb7oiYzMDjw",
}

CONFIG_ENV_KEYS = {
    "checkpoint_dir": "PANO2ROOM_CHECKPOINT_DIR",
    "lama_config_path": "PANO2ROOM_LAMA_CONFIG_PATH",
    "lama_checkpoint_path": "PANO2ROOM_LAMA_CKPT_PATH",
    "depth_checkpoint_path": "PANO2ROOM_OMNIDATA_DEPTH_CKPT_PATH",
    "normal_checkpoint_path": "PANO2ROOM_OMNIDATA_NORMAL_CKPT_PATH",
    "stable_diffusion_model_path": "PANO2ROOM_SD_MODEL_PATH",
    "sdft_weights_dir": "PANO2ROOM_SDFT_WEIGHTS_DIR",
    "auto_download_weights": "PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS",
}


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def utc_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _normalize_sample_data(sample: dict[str, Any]) -> dict[str, Any]:
    sample_data = sample.get("data")
    if isinstance(sample_data, dict) and sample_data:
        normalized: dict[str, Any] = {}
        for data_type, raw_value in sample_data.items():
            data_key = str(data_type).strip()
            if not data_key:
                raise ValueError("sample data type names must be non-empty")
            if data_key == "camera_pose" and isinstance(raw_value, dict):
                normalized[data_key] = dict(raw_value)
                continue
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"sample data path for {data_key} must be a non-empty string")
            normalized[data_key] = raw_value.strip()
        return normalized

    normalized: dict[str, str] = {}
    for data_type, raw_path in sample_data.items():
        data_key = str(data_type).strip()
        if not data_key:
            raise ValueError("sample data type names must be non-empty")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample data path for {data_key} must be a non-empty string")
        normalized[data_key] = raw_path.strip()
    return normalized


def _validate_required_data_types(
    sample_data: dict[str, Any],
    required_data_types: list[str],
    job_id: str,
) -> None:
    missing_data_types = [data_type for data_type in required_data_types if data_type not in sample_data]
    if missing_data_types:
        raise ValueError(f"sample missing required data types: {', '.join(missing_data_types)}")


def _config_string(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _configure_model_paths(config: dict[str, Any]) -> None:
    checkpoint_dir_from_config = _config_string(config, "checkpoint_dir")
    checkpoint_dir = checkpoint_dir_from_config or os.getenv("PANO2ROOM_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    os.environ["PANO2ROOM_CHECKPOINT_DIR"] = checkpoint_dir

    explicit_env_keys: set[str] = set()
    for config_key, env_key in CONFIG_ENV_KEYS.items():
        value = _config_string(config, config_key)
        if value:
            os.environ[env_key] = value
            explicit_env_keys.add(env_key)

    checkpoint_root = Path(os.environ["PANO2ROOM_CHECKPOINT_DIR"])
    for env_key, filename in CHECKPOINT_DEFAULTS.items():
        if env_key not in explicit_env_keys and (checkpoint_dir_from_config or env_key not in os.environ):
            os.environ[env_key] = str(checkpoint_root / filename)



def _required_local_paths() -> list[Path]:
    paths = [Path(os.environ[env_key]) for env_key in CHECKPOINT_DEFAULTS]
    sd_model_path = os.getenv("PANO2ROOM_SD_MODEL_PATH")
    if sd_model_path and Path(sd_model_path).is_absolute():
        paths.append(Path(sd_model_path))
    return paths


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _repo_lama_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoints" / "big-lama-config.yaml"


def _copy_lama_config_if_needed(target_path: Path) -> None:
    if target_path.exists():
        return
    source_path = _repo_lama_config_path()
    if not source_path.is_file():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _download_with_gdown(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "gdown", url, "-O", str(output_path)],
            check=True,
        )
        return

    downloaded = gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError(f"gdown failed to download {url} to {output_path}")


def _copy_inputs(sample_data: dict[str, Any], output_root: Path, *, model_outputs: bool) -> list[dict[str, Any]]:
    output_dir = output_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for index, (data_type, raw_path) in enumerate(sample_data.items()):
        if not isinstance(raw_path, str):
            continue
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
    if not missing_downloads:
        return

    if not _truthy_env("PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS"):
        return

    for env_key in missing_downloads:
        output_path = Path(os.environ[env_key])
        logger.info(
            event_message(
                "pano2room_weight_download_started",
                env_key=env_key,
                output_path=str(output_path),
            )
        )
        _download_with_gdown(PANO2ROOM_WEIGHT_DOWNLOADS[env_key], output_path)
        logger.info(
            event_message(
                "pano2room_weight_download_finished",
                env_key=env_key,
                output_path=str(output_path),
                size_bytes=output_path.stat().st_size if output_path.exists() else None,
            )
        )


def _validate_local_paths(paths: list[Path]) -> None:
    _ensure_pano2room_weights()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        hint = " Set PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS=1 to download Pano2Room checkpoints, or mount/configure the weight paths."
        raise FileNotFoundError("missing Pano2Room weight path(s): " + ", ".join(missing) + hint)


def _resolve_camera_trajectory_dir(sample_data: dict[str, str]) -> Path:
    for data_key in CAMERA_TRAJECTORY_DATA_KEYS:
        raw_path = sample_data.get(data_key)
        if raw_path:
            trajectory_dir = Path(raw_path)
            if not trajectory_dir.is_dir():
                raise FileNotFoundError(f"camera trajectory directory not found: {trajectory_dir}")
            return trajectory_dir

    if not DEFAULT_CAMERA_TRAJECTORY_DIR.is_dir():
        raise FileNotFoundError(f"default camera trajectory directory not found: {DEFAULT_CAMERA_TRAJECTORY_DIR}")
    return DEFAULT_CAMERA_TRAJECTORY_DIR


def _artifact(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "artifact_type": "model_output",
        "role": "primary",
        "data_type": "3dgs",
        "path": str(path.relative_to(output_root)),
        "format": "ply",
        "size_bytes": path.stat().st_size,
        "metadata": {"runner": RUNNER_NAME},
    }


def _failure_result(
    *,
    started_at: float,
    completed_at: float,
    code: str,
    message: str,
    metrics: list[dict[str, Any]],
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "started_at": utc_time(started_at),
        "completed_at": utc_time(completed_at),
        "metrics": metrics,
        "artifacts": [],
        "failure": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "stage": "adapter",
            "traceback": traceback.format_exc(),
        },
    }


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    monitor: ResourceMonitor | None = None
    resource_metrics: list[dict[str, Any]] = []

    try:
        job = job_request["job"]
        runtime = job_request["runtime"]
        sample_data = _normalize_sample_data(job_request["sample"])
        config = job_request.get("config", {})
        required_data_types = config.get("required_data_types", ["image"])
        _validate_required_data_types(sample_data, required_data_types)

        image_path = Path(sample_data["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"input image not found: {image_path}")

        requested_device = str(runtime.get("device", "cuda:0")).strip().lower()
        if requested_device and not requested_device.startswith("cuda"):
            raise RuntimeError(f"Pano2Room runner requires a CUDA device, got {requested_device}")

        output_root = Path(runtime["output_dir"])
        temp_root = Path(runtime["temp_dir"])
        run_dir = temp_root / RUNNER_NAME
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        _configure_model_paths(config)
        _validate_local_paths(_required_local_paths())

        import torch
        from pano2room import Pano2RoomPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Pano2Room runner requires CUDA")

        camera_trajectory_dir = str(_resolve_camera_trajectory_dir(sample_data))

        logger.info(
            event_message(
                "pano2room_run_started",
                job_id=job.get("job_id"),
                image_path=str(image_path),
                output_dir=str(output_root),
                temp_dir=str(run_dir),
                camera_trajectory_dir=camera_trajectory_dir,
            )
        )

        monitor = ResourceMonitor(sample_data=sample_data, output_dir=output_root)
        monitor.start()

        pipeline = Pano2RoomPipeline(
            image_path=str(image_path),
            save_path=str(run_dir),
            camera_trajectory_dir=camera_trajectory_dir,
            render_outputs=False,
        )
        produced_path = pipeline.run()
        source_ply = Path(produced_path) if produced_path else run_dir / OUTPUT_FILENAME
        if not source_ply.is_file():
            raise FileNotFoundError(f"Pano2Room did not produce {OUTPUT_FILENAME}: {source_ply}")

        output_ply = output_root / OUTPUT_FILENAME
        shutil.copy2(source_ply, output_ply)
        resource_metrics = monitor.stop()
        monitor = None
        completed_at = time.time()

        logger.info(
            event_message(
                "pano2room_run_completed",
                job_id=job.get("job_id"),
                output_ply=str(output_ply),
                wall_time_ms=round((completed_at - started_at) * 1000, 3),
            )
        )

        return {
            "status": "completed",
            "started_at": utc_time(started_at),
            "completed_at": utc_time(completed_at),
            "metrics": resource_metrics,
            "artifacts": [_artifact(output_ply, output_root)],
            "failure": None,
        }
    except Exception as exc:
        if monitor is not None:
            resource_metrics = monitor.stop()
        completed_at = time.time()
        logger.exception(event_message("pano2room_run_failed", error=str(exc)))
        return _failure_result(
            started_at=started_at,
            completed_at=completed_at,
            code="PANO2ROOM_RUN_FAILED",
            message=str(exc),
            metrics=resource_metrics,
        )

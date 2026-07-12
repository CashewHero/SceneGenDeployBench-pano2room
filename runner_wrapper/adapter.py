from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapter")

RUNNER_NAME = "pano2room"
OUTPUT_FILENAME = "3DGS.ply"
DEFAULT_MODEL_CACHE_DIR = "/data/model_cache/pano2room"
DEFAULT_CHECKPOINT_DIR = f"{DEFAULT_MODEL_CACHE_DIR}/checkpoints"
DEFAULT_CAMERA_TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "input" / "Camera_Trajectory"
CAMERA_TRAJECTORY_DATA_KEYS = ("camera_trajectory", "camera_trajectory_dir")

CHECKPOINT_DEFAULTS = {
    "PANO2ROOM_CHECKPOINT_LAMA_CONFIG": "big-lama-config.yaml",
    "PANO2ROOM_CHECKPOINT_LAMA_CKPT": "big-lama.ckpt",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_DEPTH": "omnidata_dpt_depth_v2.ckpt",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_NORMAL": "omnidata_dpt_normal_v2.ckpt",
    "PANO2ROOM_CHECKPOINT_SDFT_WEIGHTS_DIR": "SDFT_weights",
}

MODEL_ENV_ALIASES = {
    "PANO2ROOM_CHECKPOINT_LAMA_CONFIG": "PANO2ROOM_LAMA_CONFIG_PATH",
    "PANO2ROOM_CHECKPOINT_LAMA_CKPT": "PANO2ROOM_LAMA_CKPT_PATH",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_DEPTH": "PANO2ROOM_OMNIDATA_DEPTH_CKPT_PATH",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_NORMAL": "PANO2ROOM_OMNIDATA_NORMAL_CKPT_PATH",
    "PANO2ROOM_CHECKPOINT_SDFT_WEIGHTS_DIR": "PANO2ROOM_SDFT_WEIGHTS_DIR",
    "PANO2ROOM_HF_STABLE_DIFFUSION_MODEL": "PANO2ROOM_SD_MODEL_PATH",
}

PANO2ROOM_WEIGHT_DOWNLOADS = {
    "PANO2ROOM_CHECKPOINT_LAMA_CKPT": "https://drive.google.com/uc?id=1H5CHOsm_yAxZI9a5hv9tyZmh5CMjJxap",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_DEPTH": "https://drive.google.com/uc?id=18S9ycwHi07hzPdLORsAQFFORTeovdo4E",
    "PANO2ROOM_CHECKPOINT_OMNIDATA_NORMAL": "https://drive.google.com/uc?id=1gMBrl51AZZr6ANy8d77KFXb7oiYzMDjw",
}
PANO2ROOM_WEIGHT_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h"
    "?rlkey=a96g2incso7g53evzamzo0j0y&dl=1"
)

CONFIG_ENV_KEYS = {
    "checkpoint_dir": "PANO2ROOM_CHECKPOINT_DIR",
    "checkpoint_lama_config": "PANO2ROOM_CHECKPOINT_LAMA_CONFIG",
    "checkpoint_lama_ckpt": "PANO2ROOM_CHECKPOINT_LAMA_CKPT",
    "checkpoint_omnidata_depth": "PANO2ROOM_CHECKPOINT_OMNIDATA_DEPTH",
    "checkpoint_omnidata_normal": "PANO2ROOM_CHECKPOINT_OMNIDATA_NORMAL",
    "checkpoint_sdft_weights_dir": "PANO2ROOM_CHECKPOINT_SDFT_WEIGHTS_DIR",
    "hf_stable_diffusion_model": "PANO2ROOM_HF_STABLE_DIFFUSION_MODEL",
    "hf_home": "HF_HOME",
    "hf_token": "HF_TOKEN",
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

    normalized: dict[str, Any] = {}
    for index, item in enumerate(sample.get("inputs", [])):
        if not isinstance(item, dict):
            raise ValueError("legacy sample inputs must be objects")
        data_type = str(item.get("role", f"input_{index}")).strip()
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample input path for {data_type} must be a non-empty string")
        normalized[data_type] = raw_path.strip()
    return normalized


def _validate_required_data_types(sample_data: dict[str, Any], required_data_types: list[str]) -> None:
    missing_data_types = [data_type for data_type in required_data_types if data_type not in sample_data]
    if missing_data_types:
        raise ValueError(f"sample missing required data types: {', '.join(missing_data_types)}")


def _config_string(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _configure_model_paths(config: dict[str, Any], model_cache_dir: str | None = None) -> None:
    checkpoint_dir_from_config = _config_string(config, "checkpoint_dir")
    default_checkpoint_dir = str(Path(model_cache_dir or DEFAULT_MODEL_CACHE_DIR) / "checkpoints")
    checkpoint_dir = checkpoint_dir_from_config or os.getenv("PANO2ROOM_CHECKPOINT_DIR", default_checkpoint_dir)
    os.environ["PANO2ROOM_CHECKPOINT_DIR"] = checkpoint_dir

    explicit_env_keys: set[str] = set()
    for config_key, env_key in CONFIG_ENV_KEYS.items():
        value = _config_string(config, config_key)
        if value:
            os.environ[env_key] = value
            explicit_env_keys.add(env_key)

    checkpoint_root = Path(os.environ["PANO2ROOM_CHECKPOINT_DIR"])
    for env_key, filename in CHECKPOINT_DEFAULTS.items():
        if env_key not in explicit_env_keys and not os.getenv(env_key):
            os.environ[env_key] = str(checkpoint_root / filename)

    for public_env_key, model_env_key in MODEL_ENV_ALIASES.items():
        value = os.getenv(public_env_key)
        if value:
            os.environ[model_env_key] = value


def _required_local_paths() -> list[Path]:
    paths = [Path(os.environ[env_key]) for env_key in CHECKPOINT_DEFAULTS if env_key != "PANO2ROOM_CHECKPOINT_SDFT_WEIGHTS_DIR"]
    sd_model_path = os.getenv("PANO2ROOM_HF_STABLE_DIFFUSION_MODEL")
    if sd_model_path and Path(sd_model_path).is_absolute():
        paths.append(Path(sd_model_path))
    return paths


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _usable_path(path: Path) -> bool:
    return path.is_dir() or (path.is_file() and path.stat().st_size > 0)


def _repo_lama_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoints" / "big-lama-config.yaml"


@contextmanager
def _exclusive_path_lock(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _copy_lama_config_if_needed(target_path: Path) -> None:
    source_path = _repo_lama_config_path()
    if not source_path.is_file():
        return
    with _exclusive_path_lock(target_path):
        if _usable_path(target_path):
            return
        temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.part")
        try:
            shutil.copy2(source_path, temp_path)
            os.replace(temp_path, target_path)
        finally:
            temp_path.unlink(missing_ok=True)


def _download_with_gdown(url: str, output_path: Path) -> None:
    with _exclusive_path_lock(output_path):
        if _usable_path(output_path):
            return

        temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.part")
        try:
            try:
                import gdown
            except ImportError:
                subprocess.run([sys.executable, "-m", "gdown", url, "-O", str(temp_path)], check=True)
            else:
                try:
                    downloaded = gdown.download(url, str(temp_path), quiet=False, fuzzy=True)
                except TypeError:
                    downloaded = gdown.download(url, str(temp_path), quiet=False)
                if downloaded is None:
                    raise RuntimeError(f"gdown failed to download {url} to {output_path}")

            if not temp_path.is_file() or temp_path.stat().st_size <= 0:
                raise RuntimeError(f"download produced no data for {output_path}")
            os.replace(temp_path, output_path)
        finally:
            temp_path.unlink(missing_ok=True)


def _download_checkpoint_archive(output_paths: dict[str, Path]) -> None:
    checkpoint_root = Path(os.environ["PANO2ROOM_CHECKPOINT_DIR"])
    archive_lock_target = checkpoint_root / "pano2room-pretrained-checkpoints"
    with _exclusive_path_lock(archive_lock_target):
        missing_paths = {
            env_key: path
            for env_key, path in output_paths.items()
            if not _usable_path(path)
        }
        if not missing_paths:
            return

        archive_path = checkpoint_root / f".pano2room-checkpoints.{uuid.uuid4().hex}.zip.part"
        extracted_temp_paths: list[Path] = []
        try:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                PANO2ROOM_WEIGHT_ARCHIVE_URL,
                headers={"User-Agent": "SceneGenDeployBench-Pano2Room/0.1.0"},
            )
            print(
                "Google Drive checkpoint download failed; downloading official "
                f"Pano2Room checkpoint archive from Dropbox to {archive_path}",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file, length=1024 * 1024)

            if not _usable_path(archive_path):
                raise RuntimeError("Pano2Room checkpoint archive download produced no data")

            with zipfile.ZipFile(archive_path) as archive:
                members_by_name = {
                    Path(info.filename).name: info
                    for info in archive.infolist()
                    if not info.is_dir()
                }
                for env_key, output_path in missing_paths.items():
                    expected_name = CHECKPOINT_DEFAULTS[env_key]
                    member = members_by_name.get(expected_name)
                    if member is None:
                        raise RuntimeError(
                            f"official Pano2Room checkpoint archive is missing {expected_name}"
                        )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.part")
                    extracted_temp_paths.append(temp_path)
                    with archive.open(member) as source, temp_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    if not _usable_path(temp_path):
                        raise RuntimeError(f"checkpoint extraction produced no data for {output_path}")
                    os.replace(temp_path, output_path)
                    print(f"Installed Pano2Room checkpoint: {output_path}", flush=True)
        finally:
            archive_path.unlink(missing_ok=True)
            for temp_path in extracted_temp_paths:
                temp_path.unlink(missing_ok=True)


def _ensure_pano2room_weights() -> None:
    _copy_lama_config_if_needed(Path(os.environ["PANO2ROOM_CHECKPOINT_LAMA_CONFIG"]))

    missing_downloads = [
        env_key
        for env_key in PANO2ROOM_WEIGHT_DOWNLOADS
        if not _usable_path(Path(os.environ[env_key]))
    ]
    if not missing_downloads:
        return

    if not _truthy_env("PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS"):
        return

    output_paths = {
        env_key: Path(os.environ[env_key])
        for env_key in PANO2ROOM_WEIGHT_DOWNLOADS
    }
    try:
        for env_key in missing_downloads:
            output_path = output_paths[env_key]
            logger.info(event_message("pano2room_weight_download_started", env_key=env_key, output_path=str(output_path)))
            _download_with_gdown(PANO2ROOM_WEIGHT_DOWNLOADS[env_key], output_path)
            logger.info(
                event_message(
                    "pano2room_weight_download_finished",
                    env_key=env_key,
                    output_path=str(output_path),
                    size_bytes=output_path.stat().st_size if output_path.exists() else None,
                )
            )
    except Exception as drive_error:
        logger.warning(
            event_message(
                "pano2room_google_drive_download_failed",
                error=str(drive_error),
                fallback="official_dropbox_archive",
            )
        )
        try:
            _download_checkpoint_archive(output_paths)
        except Exception as archive_error:
            raise RuntimeError(
                "Pano2Room checkpoint auto-download failed from both Google Drive "
                f"and the official Dropbox archive: {archive_error}"
            ) from archive_error


def _write_metrics_file(metrics_path: Path, summary: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _validate_local_paths(paths: list[Path]) -> None:
    _ensure_pano2room_weights()
    missing = [str(path) for path in paths if not _usable_path(path)]
    if missing:
        hint = " Set PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS=1 to download Pano2Room checkpoints, or mount/configure the weight paths."
        raise FileNotFoundError("missing Pano2Room weight path(s): " + ", ".join(missing) + hint)


def _load_trajectory_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read camera_trajectory YAML files") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise ValueError(f"camera_trajectory must be a JSON/YAML file or pose directory: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"camera_trajectory file must contain an object: {path}")
    return data


def _as_float_list(value: Any, *, length: int, field: str) -> list[float]:
    if value is None:
        raise ValueError(f"missing {field}")
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} numbers")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{field} must contain only finite numbers")
    return numbers


def _quat_xyzw_to_rotation_matrix(quat: list[float]) -> list[list[float]]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("rotation_quaternion_xyzw must not be zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matrix_from_frame(frame: dict[str, Any]) -> list[list[float]]:
    position = _as_float_list(frame.get("position", [0.0, 0.0, 0.0]), length=3, field="position")
    quat = _as_float_list(
        frame.get("rotation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        length=4,
        field="rotation_quaternion_xyzw",
    )
    rotation = _quat_xyzw_to_rotation_matrix(quat)
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], position[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], position[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _invert_rigid_transform(matrix: list[list[float]]) -> list[list[float]]:
    rotation_t = [[matrix[col][row] for col in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    inverted_translation = [-sum(rotation_t[row][col] * translation[col] for col in range(3)) for row in range(3)]
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inverted_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inverted_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inverted_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_pose_file(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(f"{0.0 if value == 0 else value:.9g}" for value in row) + "\n")


def _materialize_camera_trajectory(trajectory_file: Path, output_dir: Path) -> Path:
    data = _load_trajectory_file(trajectory_file)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"camera_trajectory file must contain a non-empty frames list: {trajectory_file}")

    convention = str(data.get("convention") or "camera_to_world").strip().lower()
    if convention not in {"camera_to_world", "world_to_camera"}:
        raise ValueError(f"unsupported camera_trajectory convention: {convention}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_pose in output_dir.glob("camera_pose_frame*.txt"):
        stale_pose.unlink()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"camera_trajectory frame {index} must be an object")
        matrix = _matrix_from_frame(frame)
        if convention == "world_to_camera":
            matrix = _invert_rigid_transform(matrix)
        _write_pose_file(output_dir / f"camera_pose_frame{index:06d}.txt", matrix)
    return output_dir


def _resolve_camera_trajectory_dir(sample_data: dict[str, Any], run_dir: Path) -> Path:
    for data_key in CAMERA_TRAJECTORY_DATA_KEYS:
        raw_path = sample_data.get(data_key)
        if raw_path:
            trajectory_path = Path(str(raw_path))
            if trajectory_path.is_dir():
                return trajectory_path
            if trajectory_path.is_file():
                return _materialize_camera_trajectory(trajectory_path, run_dir / "camera_trajectory")
            raise FileNotFoundError(f"{data_key} path not found: {trajectory_path}")

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
    log_path: Path,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "started_at": utc_time(started_at),
        "completed_at": utc_time(completed_at),
        "metrics": metrics,
        "artifacts": [
            {
                "artifact_type": "job_log",
                "role": "stdout",
                "path": "runner.log",
                "format": "text",
                "size_bytes": log_path.stat().st_size,
            }
        ],
        "failure": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "stage": "adapter",
        },
    }


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    runtime = job_request["runtime"]
    output_root = Path(runtime["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "runner.log"
    with tee_job_output(log_path):
        return _run_job_logged(job_request, started_at, output_root, log_path)


def _run_job_logged(
    job_request: dict[str, Any],
    started_at: float,
    output_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    monitor: ResourceMonitor | None = None
    resource_metrics: list[dict[str, Any]] = []

    try:
        job = job_request["job"]
        runtime = job_request["runtime"]
        sample_data = _normalize_sample_data(job_request["sample"])
        config = job_request.get("config", {})
        required_data_types = config.get("required_data_types", ["image"])
        _validate_required_data_types(sample_data, required_data_types)

        image_path = Path(str(sample_data["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"input image not found: {image_path}")

        requested_device = str(runtime.get("device", "cuda:0")).strip().lower()
        if requested_device and not requested_device.startswith("cuda"):
            raise RuntimeError(f"Pano2Room runner requires a CUDA device, got {requested_device}")

        temp_root = Path(runtime["temp_dir"])
        run_dir = temp_root / RUNNER_NAME
        metrics_path = output_root / "metrics.json"
        print(f"pano2room job {job.get('job_id')} started", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        _configure_model_paths(config, _config_string(runtime, "model_cache_dir"))
        _validate_local_paths(_required_local_paths())
        camera_trajectory_dir = str(_resolve_camera_trajectory_dir(sample_data, run_dir))

        import torch
        from pano2room import Pano2RoomPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Pano2Room runner requires CUDA")

        logger.info(
            event_message(
                "pano2room_run_started",
                job_id=job.get("job_id"),
                image_path=str(image_path),
                output_dir=str(output_root),
                temp_dir=str(run_dir),
                camera_trajectory_dir=camera_trajectory_dir,
                timeout_seconds=job.get("timeout_seconds"),
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
        wall_time_ms = round((completed_at - started_at) * 1000, 3)
        print(f"pano2room job {job.get('job_id')} completed in {wall_time_ms} ms", flush=True)
        _write_metrics_file(
            metrics_path,
            {
                "metrics": resource_metrics,
                "resource_metrics": resource_metrics,
                "output_files": [
                    {
                        "path": OUTPUT_FILENAME,
                        "format": "ply",
                        "size_bytes": output_ply.stat().st_size,
                    }
                ],
            },
        )

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
            "artifacts": [
                _artifact(output_ply, output_root),
                {
                    "artifact_type": "job_log",
                    "role": "stdout",
                    "path": "runner.log",
                    "format": "text",
                },
                {
                    "artifact_type": "metric_summary",
                    "role": "summary",
                    "path": "metrics.json",
                    "format": "json",
                },
            ],
            "failure": None,
        }
    except Exception as exc:
        if monitor is not None:
            resource_metrics = monitor.stop()
        completed_at = time.time()
        print(f"pano2room job failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return _failure_result(
            started_at=started_at,
            completed_at=completed_at,
            code="PANO2ROOM_RUN_FAILED",
            message="Pano2Room failed; see runner.log",
            metrics=resource_metrics,
            log_path=log_path,
        )

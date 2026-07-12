from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runner_wrapper.adapter import (
    CHECKPOINT_DEFAULTS,
    PANO2ROOM_WEIGHT_DOWNLOADS,
    _configure_model_paths,
    _download_checkpoint_archive,
    _download_with_gdown,
)


class AdapterCacheTests(unittest.TestCase):
    def test_runtime_cache_dir_sets_checkpoint_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _configure_model_paths({}, "/data/model_cache/custom")
            self.assertEqual(
                os.environ["PANO2ROOM_CHECKPOINT_DIR"],
                "/data/model_cache/custom/checkpoints",
            )
            self.assertEqual(
                os.environ["PANO2ROOM_CHECKPOINT_LAMA_CKPT"],
                "/data/model_cache/custom/checkpoints/big-lama.ckpt",
            )

    def test_concurrent_download_is_published_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "weights.ckpt"
            output_path.write_bytes(b"")
            call_count = 0
            count_lock = threading.Lock()

            def download(_: str, destination: str, **__: object) -> str:
                nonlocal call_count
                with count_lock:
                    call_count += 1
                time.sleep(0.05)
                Path(destination).write_bytes(b"complete-weights")
                return destination

            fake_gdown = SimpleNamespace(download=download)
            errors: list[Exception] = []

            def worker() -> None:
                try:
                    _download_with_gdown("https://example.invalid/weights", output_path)
                except Exception as exc:  # pragma: no cover - assertion reports details
                    errors.append(exc)

            with patch.dict(sys.modules, {"gdown": fake_gdown}):
                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)

            self.assertEqual(errors, [])
            self.assertEqual(call_count, 1)
            self.assertEqual(output_path.read_bytes(), b"complete-weights")
            self.assertFalse(any(path.name.endswith(".part") for path in output_path.parent.iterdir()))

    def test_failed_download_does_not_publish_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "weights.ckpt"

            def download(_: str, destination: str, **__: object) -> str:
                Path(destination).write_bytes(b"")
                return destination

            with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
                with self.assertRaisesRegex(RuntimeError, "produced no data"):
                    _download_with_gdown("https://example.invalid/weights", output_path)

            self.assertFalse(output_path.exists())
            self.assertFalse(any(path.name.endswith(".part") for path in output_path.parent.iterdir()))

    def test_checkpoint_archive_extracts_required_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            output_paths = {
                env_key: checkpoint_dir / CHECKPOINT_DEFAULTS[env_key]
                for env_key in PANO2ROOM_WEIGHT_DOWNLOADS
            }
            archive_bytes = BytesIO()
            with zipfile.ZipFile(archive_bytes, "w") as archive:
                for env_key, output_path in output_paths.items():
                    archive.writestr(f"pre_checkpoints/{output_path.name}", env_key.encode())

            class FakeResponse(BytesIO):
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

            with (
                patch.dict(os.environ, {"PANO2ROOM_CHECKPOINT_DIR": str(checkpoint_dir)}),
                patch("runner_wrapper.adapter.urllib.request.urlopen", return_value=FakeResponse(archive_bytes.getvalue())),
            ):
                _download_checkpoint_archive(output_paths)

            for env_key, output_path in output_paths.items():
                self.assertEqual(output_path.read_bytes(), env_key.encode())
            self.assertFalse(any(path.name.endswith(".part") for path in checkpoint_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()

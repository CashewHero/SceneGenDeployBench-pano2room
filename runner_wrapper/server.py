from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import Any, Callable

logger = logging.getLogger("runner_wrapper.server")


def configure_logging() -> None:
    level_name = os.getenv("RUNNER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_run_job_handler(target: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "RUNNER_ADAPTER must use the format 'module.path:function_name', "
            f"received {target!r}"
        )

    module = import_module(module_name)
    handler = getattr(module, attribute_name)
    if not callable(handler):
        raise TypeError(f"configured adapter target is not callable: {target}")
    return handler


@dataclass(frozen=True)
class RunnerSettings:
    port: int
    runner_name: str
    runner_type: str
    runner_version: str
    contract_version: int
    idle_timeout_seconds: int
    startup_timeout_seconds: float
    adapter_target: str

    @classmethod
    def from_env(cls) -> "RunnerSettings":
        return cls(
            port=int(os.getenv("RUNNER_PORT", "58090")),
            runner_name=os.getenv("RUNNER_NAME", "test-runner"),
            runner_type=os.getenv("RUNNER_TYPE", "generator"),
            runner_version=os.getenv("RUNNER_VERSION", "0.1.0"),
            contract_version=int(os.getenv("RUNNER_CONTRACT_VERSION", "1")),
            idle_timeout_seconds=int(os.getenv("RUNNER_IDLE_TIMEOUT_SECONDS", "900")),
            startup_timeout_seconds=float(os.getenv("RUNNER_STARTUP_TIMEOUT_SECONDS", "60")),
            adapter_target=os.getenv("RUNNER_ADAPTER", "runner_wrapper.adapter:run_job"),
        )


def make_status_payload(runner: "Runner", accepted: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "online": True,
        "runner_name": runner.settings.runner_name,
        "runner_type": runner.settings.runner_type,
        "runner_version": runner.settings.runner_version,
        "contract_version": runner.settings.contract_version,
        "batch_id": runner.batch_id,
        "state": runner.state,
        "current_job_id": runner.current_job_id,
        "updated_at": runner.updated_at,
        "result": runner.result if runner.state in ("finished", "failed") else None,
    }
    if accepted is not None:
        payload["accepted"] = accepted
    return payload


def build_failure_result(exc: Exception) -> dict[str, Any]:
    completed_at = utc_now()
    return {
        "status": "failed",
        "started_at": completed_at,
        "completed_at": completed_at,
        "metrics": [],
        "artifacts": [],
        "failure": {
            "code": "RUNNER_INTERNAL_ERROR",
            "message": str(exc),
            "retryable": False,
            "stage": "runner",
            "traceback": traceback.format_exc(),
        },
    }


@dataclass
class Runner:
    settings: RunnerSettings
    run_job_handler: Callable[[dict[str, Any]], dict[str, Any]]
    state: str = "starting"
    batch_id: str | None = None
    current_job_id: str | None = None
    result: dict[str, Any] | None = None
    updated_at: str = field(default_factory=utc_now)
    last_status_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_state(
        self,
        state: str,
        current_job_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        previous_state = self.state
        previous_job_id = self.current_job_id
        self.state = state
        self.current_job_id = current_job_id
        self.result = result
        self.updated_at = utc_now()
        if previous_state != state or previous_job_id != current_job_id:
            logger.info(
                event_message(
                    "runner_state_changed",
                    previous_state=previous_state,
                    state=state,
                    job_id=current_job_id,
                )
            )

    def mark_ready(self) -> None:
        with self.lock:
            self.set_state("idle")

    def touch_status(self) -> None:
        with self.lock:
            self.last_status_at = time.time()
            self.updated_at = utc_now()

    def submit_job(self, job_request: dict[str, Any]) -> bool:
        job_id = job_request.get("job", {}).get("job_id")
        batch_id = str(job_request.get("job", {}).get("batch_id") or "").strip()
        with self.lock:
            if not batch_id:
                logger.warning(event_message("job_rejected", job_id=job_id, state=self.state, reason="missing_batch_id"))
                return False
            if self.state not in ("idle", "finished", "failed"):
                logger.warning(event_message("job_rejected", job_id=job_id, state=self.state, batch_id=batch_id))
                return False
            if self.batch_id and self.batch_id != batch_id:
                logger.warning(
                    event_message(
                        "job_rejected",
                        job_id=job_id,
                        state=self.state,
                        batch_id=batch_id,
                        bound_batch_id=self.batch_id,
                        reason="batch_mismatch",
                    )
                )
                return False
            if self.batch_id is None:
                self.batch_id = batch_id
            self.set_state("running", current_job_id=job_id, result=None)
            logger.info(
                event_message(
                    "job_accepted",
                    job_id=job_id,
                    batch_id=batch_id,
                )
            )

        worker = threading.Thread(target=self._run_job_thread, args=(job_request,), daemon=True)
        worker.start()
        return True

    def _run_job_thread(self, job_request: dict[str, Any]) -> None:
        job_id = job_request.get("job", {}).get("job_id")
        logger.info(
            event_message(
                "job_execution_started",
                job_id=job_id,
                output_dir=job_request.get("runtime", {}).get("output_dir"),
            )
        )
        try:
            result = self.run_job_handler(job_request)
            with self.lock:
                final_state = "finished" if result.get("status") == "completed" else "failed"
                self.set_state(final_state, current_job_id=job_id, result=result)
                logger.info(
                    event_message(
                        "job_execution_finished",
                        job_id=job_id,
                        state=final_state,
                        result_status=result.get("status"),
                        artifact_count=len(result.get("artifacts", [])),
                        metric_count=len(result.get("metrics", [])),
                    )
                )
        except Exception as exc:
            failure = build_failure_result(exc)
            with self.lock:
                self.set_state("failed", current_job_id=job_id, result=failure)
            logger.exception(event_message("job_execution_failed", job_id=job_id, error=str(exc)))

    def request_shutdown(self) -> bool:
        with self.lock:
            interrupted_job_id = self.current_job_id if self.state == "running" else None
            self.set_state("shutting_down")
            logger.info(
                event_message(
                    "shutdown_requested",
                    state=self.state,
                    interrupted_job_id=interrupted_job_id,
                )
            )
            return True


class RunnerHandler(BaseHTTPRequestHandler):
    server_version = "RunnerWrapper/1.0"

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path != "/status":
            logger.warning(event_message("http_not_found", method="GET", path=self.path))
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        self.server.runner.touch_status()
        with self.server.runner.lock:
            payload = make_status_payload(self.server.runner)
        self._send_json(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        if self.path == "/run-job":
            self._handle_run_job()
            return
        if self.path == "/shutdown":
            self._handle_shutdown()
            return

        logger.warning(event_message("http_not_found", method="POST", path=self.path))
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _handle_run_job(self) -> None:
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            logger.warning(event_message("invalid_json", path=self.path))
            self._send_json(HTTPStatus.BAD_REQUEST, {"accepted": False, "error": "invalid json"})
            return

        accepted = self.server.runner.submit_job(payload)
        with self.server.runner.lock:
            response = make_status_payload(self.server.runner, accepted=accepted)
        self._send_json(HTTPStatus.OK, response)

    def _handle_shutdown(self) -> None:
        accepted = self.server.runner.request_shutdown()
        with self.server.runner.lock:
            response = make_status_payload(self.server.runner, accepted=accepted)
        self._send_json(HTTPStatus.OK, response)
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args: object) -> None:
        return


class RunnerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[RunnerHandler], runner: Runner):
        super().__init__(server_address, handler_class)
        self.runner = runner


def idle_shutdown_loop(server: RunnerHTTPServer) -> None:
    while True:
        time.sleep(5)
        with server.runner.lock:
            if server.runner.state == "running":
                continue

            idle_for = time.time() - server.runner.last_status_at
            if (
                server.runner.state != "shutting_down"
                and idle_for >= server.runner.settings.idle_timeout_seconds
            ):
                logger.info(
                    event_message(
                        "idle_shutdown_triggered",
                        idle_seconds=round(idle_for, 3),
                    )
                )
                server.runner.set_state("shutting_down")

            if server.runner.state == "shutting_down":
                break

    logger.info(event_message("server_shutdown"))
    server.shutdown()


def start_startup_timeout_watchdog(settings: RunnerSettings) -> threading.Event:
    ready_event = threading.Event()
    if settings.startup_timeout_seconds <= 0:
        raise ValueError("RUNNER_STARTUP_TIMEOUT_SECONDS must be greater than 0")

    def watch_startup() -> None:
        if ready_event.wait(settings.startup_timeout_seconds):
            return
        logger.error(
            event_message(
                "runner_startup_timeout",
                timeout_seconds=settings.startup_timeout_seconds,
                adapter_target=settings.adapter_target,
            )
        )
        os._exit(124)

    threading.Thread(target=watch_startup, daemon=True).start()
    return ready_event


def main() -> None:
    configure_logging()
    settings = RunnerSettings.from_env()
    startup_ready = start_startup_timeout_watchdog(settings)
    run_job_handler = load_run_job_handler(settings.adapter_target)
    logger.info(
        event_message(
            "runner_starting",
            port=settings.port,
            runner_name=settings.runner_name,
            runner_version=settings.runner_version,
            idle_timeout_seconds=settings.idle_timeout_seconds,
            startup_timeout_seconds=settings.startup_timeout_seconds,
            adapter_target=settings.adapter_target,
        )
    )

    runner = Runner(settings=settings, run_job_handler=run_job_handler)
    server = RunnerHTTPServer(("0.0.0.0", settings.port), RunnerHandler, runner)
    runner.mark_ready()
    startup_ready.set()

    shutdown_thread = threading.Thread(target=idle_shutdown_loop, args=(server,), daemon=True)
    shutdown_thread.start()

    try:
        server.serve_forever()
    finally:
        logger.info(event_message("server_closed"))
        server.server_close()


if __name__ == "__main__":
    main()

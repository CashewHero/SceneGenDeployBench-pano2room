#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

safe_name() {
  tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g'
}

repo_name="$(basename "${REPO_ROOT}" | safe_name)"

IMAGE="${RUNNER_IMAGE:-scenegendeploybench-pano2room:local}"
CONTAINER="${RUNNER_CONTAINER:-pano2room-runner-localtest}"
HOST_PORT="${RUNNER_HOST_PORT:-58090}"
DATA_DIR="${RUNNER_DATA_DIR:-${REPO_ROOT}/data}"
RUNNER_NAME="${RUNNER_NAME:-pano2room}"
RUNNER_TYPE="${RUNNER_TYPE:-generator}"
RUNNER_VERSION="${RUNNER_VERSION:-0.1.0}"
RUNNER_ADAPTER="${RUNNER_ADAPTER:-runner_wrapper.adapter:run_job}"
RUNNER_WEIGHTS_DIR="${RUNNER_WEIGHTS_DIR:-}"
RUNNER_DOCKER_GPUS="${RUNNER_DOCKER_GPUS:-all}"
RUNNER_STARTUP_TIMEOUT_SECONDS="${RUNNER_STARTUP_TIMEOUT_SECONDS:-300}"
REQUEST_FILE="${RUNNER_REQUEST_FILE:-${SCRIPT_DIR}/examples/${RUNNER_TYPE}_job_request.json}"

usage() {
  cat <<EOF
Usage:
  runner_wrapper/localtest.sh test
  runner_wrapper/localtest.sh build
  runner_wrapper/localtest.sh run
  runner_wrapper/localtest.sh smoke
  runner_wrapper/localtest.sh status
  runner_wrapper/localtest.sh logs
  runner_wrapper/localtest.sh down

Environment:
  RUNNER_IMAGE=${IMAGE}
  RUNNER_CONTAINER=${CONTAINER}
  RUNNER_HOST_PORT=${HOST_PORT}
  RUNNER_TYPE=${RUNNER_TYPE}
  RUNNER_NAME=${RUNNER_NAME}
  RUNNER_VERSION=${RUNNER_VERSION}
  RUNNER_ADAPTER=${RUNNER_ADAPTER}
  RUNNER_REQUEST_FILE=${REQUEST_FILE}
  RUNNER_DATA_DIR=${DATA_DIR}
  RUNNER_WEIGHTS_DIR=${RUNNER_WEIGHTS_DIR}
  RUNNER_DOCKER_GPUS=${RUNNER_DOCKER_GPUS}
  RUNNER_STARTUP_TIMEOUT_SECONDS=${RUNNER_STARTUP_TIMEOUT_SECONDS}
  PANO2ROOM_HF_STABLE_DIFFUSION_MODEL=${PANO2ROOM_HF_STABLE_DIFFUSION_MODEL:-}
  HF_TOKEN=${HF_TOKEN:+<set>}
  HF_HOME=${HF_HOME:-}

Set RUNNER_WEIGHTS_DIR to the host directory containing the mounted Pano2Room weights. It is mounted at /data/model_cache/pano2room/checkpoints.
Set PANO2ROOM_HF_STABLE_DIFFUSION_MODEL to a Hugging Face model id or container-visible local Stable Diffusion inpainting model directory, or provide HF_TOKEN for Hugging Face access.
EOF
}

require_tools() {
  command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
  command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
}

run_tests() {
  PYTHONPATH="${REPO_ROOT}" \
    python3 -m unittest discover -s "${SCRIPT_DIR}/tests" -v
}

build_image() {
  run_tests
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${REPO_ROOT}"
}

prepare_data() {
  mkdir -p \
    "${DATA_DIR}/datasets/smoke" \
    "${DATA_DIR}/model_cache/pano2room" \
    "${DATA_DIR}/output/pano2room@0.1.0/smoke-dataset/sample-1"

  if [[ ! -f "${DATA_DIR}/datasets/smoke/image.png" ]]; then
    if [[ -f "${REPO_ROOT}/input/input_panorama.png" ]]; then
      cp "${REPO_ROOT}/input/input_panorama.png" "${DATA_DIR}/datasets/smoke/image.png"
    elif [[ -f "${REPO_ROOT}/demo/input_panorama.png" ]]; then
      cp "${REPO_ROOT}/demo/input_panorama.png" "${DATA_DIR}/datasets/smoke/image.png"
    else
      printf 'smoke input\n' > "${DATA_DIR}/datasets/smoke/image.png"
    fi
  fi
}

run_container() {
  prepare_data
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  local env_args=(
    -e "RUNNER_PORT=58090"
    -e "RUNNER_NAME=${RUNNER_NAME}"
    -e "RUNNER_TYPE=${RUNNER_TYPE}"
    -e "RUNNER_VERSION=${RUNNER_VERSION}"
    -e "RUNNER_CONTRACT_VERSION=1"
    -e "RUNNER_ADAPTER=${RUNNER_ADAPTER}"
    -e "RUNNER_STARTUP_TIMEOUT_SECONDS=${RUNNER_STARTUP_TIMEOUT_SECONDS}"
    -e "PATH_DATASETS=/data/datasets"
    -e "PATH_MODEL_CACHE=/data/model_cache"
    -e "PATH_OUTPUT=/data/output"
  )

  if [[ -n "${RUNNER_LOG_LEVEL:-}" ]]; then
    env_args+=(-e "RUNNER_LOG_LEVEL=${RUNNER_LOG_LEVEL}")
  fi
  for env_name in \
    PANO2ROOM_CHECKPOINT_DIR \
    PANO2ROOM_CHECKPOINT_LAMA_CONFIG \
    PANO2ROOM_CHECKPOINT_LAMA_CKPT \
    PANO2ROOM_CHECKPOINT_OMNIDATA_DEPTH \
    PANO2ROOM_CHECKPOINT_OMNIDATA_NORMAL \
    PANO2ROOM_HF_STABLE_DIFFUSION_MODEL \
    HF_HOME \
    HF_TOKEN \
    PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS \
    PANO2ROOM_PANO_WIDTH \
    PANO2ROOM_PANO_HEIGHT \
    PANO2ROOM_RENDER_WIDTH \
    PANO2ROOM_RENDER_HEIGHT \
    PANO2ROOM_CHECKPOINT_SDFT_WEIGHTS_DIR; do
    if [[ -n "${!env_name:-}" ]]; then
      env_args+=(-e "${env_name}=${!env_name}")
    fi
  done

  local volume_args=(-v "${DATA_DIR}:/data")
  if [[ -n "${RUNNER_WEIGHTS_DIR}" ]]; then
    volume_args+=(-v "${RUNNER_WEIGHTS_DIR}:/data/model_cache/pano2room/checkpoints")
  fi

  local gpu_args=()
  if [[ -n "${RUNNER_DOCKER_GPUS}" && "${RUNNER_DOCKER_GPUS}" != "none" ]]; then
    gpu_args+=(--gpus "${RUNNER_DOCKER_GPUS}")
  fi

  docker run -d \
    --name "${CONTAINER}" \
    "${gpu_args[@]}" \
    -p "${HOST_PORT}:58090" \
    "${env_args[@]}" \
    "${volume_args[@]}" \
    "${IMAGE}" >/dev/null

  wait_ready
  echo "runner available at http://127.0.0.1:${HOST_PORT}"
}

wait_ready() {
  local attempt max_attempts
  max_attempts="${RUNNER_STARTUP_TIMEOUT_SECONDS}"
  for attempt in $(seq 1 "${max_attempts}"); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "runner did not become ready before startup timeout (${max_attempts}s)" >&2
  docker logs "${CONTAINER}" >&2 || true
  exit 1
}

submit_request() {
  [[ -f "${REQUEST_FILE}" ]] || { echo "missing request file: ${REQUEST_FILE}" >&2; exit 1; }
  curl -fsS \
    -X POST "http://127.0.0.1:${HOST_PORT}/run-job" \
    -H 'Content-Type: application/json' \
    --data @"${REQUEST_FILE}"
  echo
}

status_json() {
  curl -fsS "http://127.0.0.1:${HOST_PORT}/status"
}

status_field() {
  python3 -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1]) or "")' "$1"
}

ceil_positive_seconds() {
  python3 -c 'import math, sys
value = float(sys.argv[1])
name = sys.argv[2]
if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"{name} must be greater than 0")
print(math.ceil(value))' "$1" "$2"
}

request_timeout_seconds() {
  python3 -c 'import json, math, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    request = json.load(handle)
value = float(request.get("job", {}).get("timeout_seconds") or 3600)
if not math.isfinite(value) or value <= 0:
    raise SystemExit("job.timeout_seconds must be greater than 0")
print(math.ceil(value))' "${REQUEST_FILE}"
}

poll_terminal() {
  local attempt state max_attempts
  max_attempts="$(( $(request_timeout_seconds) + 60 ))"
  for attempt in $(seq 1 "${max_attempts}"); do
    state="$(status_json | status_field state)"
    case "${state}" in
      finished)
        status_json
        echo
        return 0
        ;;
      failed)
        status_json
        echo
        return 1
        ;;
    esac
    sleep 1
  done

  echo "runner job did not finish before local poll timeout (${max_attempts}s)" >&2
  return 1
}

smoke() {
  build_image
  run_container
  submit_request
  poll_terminal
}

main() {
  command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
  if [[ "${1:-}" == "test" ]]; then
    run_tests
    return
  fi

  require_tools
  RUNNER_STARTUP_TIMEOUT_SECONDS="$(ceil_positive_seconds "${RUNNER_STARTUP_TIMEOUT_SECONDS}" RUNNER_STARTUP_TIMEOUT_SECONDS)"
  case "${1:-smoke}" in
    build)
      build_image
      ;;
    run)
      build_image
      run_container
      ;;
    smoke)
      smoke
      ;;
    status)
      status_json
      echo
      ;;
    logs)
      docker logs -f "${CONTAINER}"
      ;;
    down)
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "unknown command: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"

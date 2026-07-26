FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS builder

WORKDIR /app/model_repo

# Build from the repository root:
# docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-pano2room:0.1.2 .
COPY requirements.txt requirements_git.txt /app/model_repo/

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ninja-build && \
    rm -rf /var/lib/apt/lists/*

ARG TCNN_CUDA_ARCHITECTURES=75;80;86;89
ENV TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}
ARG TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9+PTX
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FORCE_CUDA=1
ENV VIRTUAL_ENV=/opt/runner
ENV PATH=/opt/runner/bin:${PATH}

RUN python -m venv --system-site-packages ${VIRTUAL_ENV}

ARG INSTALL_REQUIREMENTS=1
RUN if [ "$INSTALL_REQUIREMENTS" = "1" ] && [ -f requirements_git.txt ]; then \
      pip install --no-cache-dir -r requirements_git.txt; \
    fi
RUN if [ "$INSTALL_REQUIREMENTS" = "1" ] && [ -f requirements.txt ]; then \
      grep -vE '^(--extra-index-url|torch[[:space:]]*$|torchvision[[:space:]]*$|triton[[:space:]]*$|xformers[[:space:]]*$)' requirements.txt > /tmp/requirements.txt; \
      pip install --no-cache-dir --retries 2 --timeout 60 -r /tmp/requirements.txt; \
    fi

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app/model_repo

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/runner
ENV PATH=/opt/runner/bin:${PATH}
COPY --from=builder /opt/runner /opt/runner
COPY . /app/model_repo

ARG INSTALL_EDITABLE=0
RUN if [ "$INSTALL_EDITABLE" = "1" ] && { [ -f pyproject.toml ] || [ -f setup.py ]; }; then \
      pip install --no-cache-dir -e .; \
    fi

RUN python -c "import pytorch3d; from diff_gaussian_rasterization import GaussianRasterizer; from simple_knn._C import distCUDA2"

EXPOSE 58090

ENV PYTHONPATH=/app/model_repo:/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNER_PORT=58090
ENV RUNNER_NAME=pano2room
ENV RUNNER_TYPE=generator
ENV RUNNER_VERSION=0.1.2
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_IDLE_TIMEOUT_SECONDS=900
ENV RUNNER_ADAPTER=runner_wrapper.adapter:run_job
ENV PANO2ROOM_AUTO_DOWNLOAD_WEIGHTS=1

CMD ["python", "-m", "runner_wrapper.server"]

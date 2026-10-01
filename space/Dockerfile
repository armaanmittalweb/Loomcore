# syntax=docker/dockerfile:1.7
#
# Loomcore live: the Hugging Face Space image (sdk: docker, app_port 7860).
#
# The Space repository holds only this file and README.md; the image clones
# the Loomcore source at LOOMCORE_REF and builds the real runtime from it:
#
#   source  clone the repo (a GITHUB_TOKEN build secret is used if present,
#           for a private repo)
#   models  prepare MobileNetV2 + bert_tiny, FP32 and INT8, with
#           scripts/prepare_all_models.py (CPU-only torch; this stage only)
#   build   configure, build and test exactly as .github/workflows/build.yml's
#           Linux job does (ctest must pass or the build fails), then run the
#           binding and server tests against the real models
#   final   python:3.11-slim with the built libraries, the bindings, the
#           models and the server; no compiler, no torch
#
# Build locally from anywhere (the build context is unused):
#   docker build -t loomcore-space -f space/Dockerfile space
#   docker run --rm -p 7860:7860 loomcore-space

ARG LOOMCORE_REPO=https://github.com/armaanmittalweb/loomcore.git
ARG LOOMCORE_REF=live
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS source
ARG LOOMCORE_REPO
ARG LOOMCORE_REF
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=false \
    set -eu; \
    url="$LOOMCORE_REPO"; \
    if [ -s /run/secrets/GITHUB_TOKEN ]; then \
      url="$(printf '%s' "$LOOMCORE_REPO" | sed "s#https://#https://x-access-token:$(cat /run/secrets/GITHUB_TOKEN)@#")"; \
    fi; \
    git clone --depth 1 --branch "$LOOMCORE_REF" "$url" /opt/loomcore; \
    git -C /opt/loomcore rev-parse --short HEAD > /opt/loomcore/.git-sha; \
    rm -rf /opt/loomcore/.git

# ---------------------------------------------------------------------------
# Stage 1: models.
FROM ${PYTHON_IMAGE} AS models
COPY --from=source /opt/loomcore /opt/loomcore
WORKDIR /opt/loomcore
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --no-cache-dir -r space/requirements-models.txt
RUN python scripts/prepare_all_models.py \
 && rm -f models/mobilenetv2.preprocessed.onnx \
 && ls -la models

# ---------------------------------------------------------------------------
# Stage 2: build and test. The RUN steps marked "CI:" are the Linux job of
# .github/workflows/build.yml, step for step.
FROM ${PYTHON_IMAGE} AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=source /opt/loomcore /opt/loomcore
WORKDIR /opt/loomcore
# CI: Install Python deps for bindings (build-time only)
RUN pip install --no-cache-dir pybind11 numpy
# CI: Install Ninja (Linux)
RUN apt-get update && apt-get install -y ninja-build && rm -rf /var/lib/apt/lists/*
# CI: Configure (Linux)
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
# CI: Build
RUN cmake --build build --config RelWithDebInfo --parallel
# CI: Test
RUN ctest --test-dir build -C RelWithDebInfo --output-on-failure
# Beyond CI: the Python binding tests (fixture graphs) and the server tests
# (real runtime, real models, every endpoint), so a broken server never ships.
COPY --from=models /opt/loomcore/models /opt/loomcore/models
RUN pip install --no-cache-dir -r space/requirements.txt pytest==9.1.1 httpx==0.28.1 \
 && LOOMCORE_BENCH_RESULTS=/tmp/loomcore-bench python -m pytest -q -p no:cacheprovider bindings/python/tests space/tests
# What the final stage takes: the libraries without their debug info.
RUN mkdir -p /opt/stage/bin \
 && cp -P build/bin/*.so* build/bin/loomcore_bench /opt/stage/bin/ \
 && strip --strip-debug /opt/stage/bin/libloomcore_core.so /opt/stage/bin/_loomcore*.so /opt/stage/bin/loomcore_bench \
 && ls -la /opt/stage/bin

# ---------------------------------------------------------------------------
# Final stage.
FROM ${PYTHON_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user
COPY --from=source /opt/loomcore/space/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# The same layout as the build tree, so the bindings' own lookup
# (LOOMCORE_BUILD_DIR/bin) and the benchmark's compiled-in project root
# (/opt/loomcore, where it reads models/ and writes benchmarks/results) hold.
COPY --from=build /opt/stage/bin /opt/loomcore/build/bin
COPY --from=source /opt/loomcore/bindings/python/loomcore /opt/loomcore/bindings/python/loomcore
COPY --from=source /opt/loomcore/assets/imagenet_labels.txt /opt/loomcore/assets/imagenet_labels.txt
COPY --from=source /opt/loomcore/space/server.py /opt/loomcore/space/server.py
COPY --from=source /opt/loomcore/space/samples /opt/loomcore/space/samples
COPY --from=source /opt/loomcore/.git-sha /opt/loomcore/.git-sha
COPY --from=models /opt/loomcore/models /opt/loomcore/models
RUN mkdir -p /opt/loomcore/benchmarks \
 && ln -s /tmp/loomcore-bench /opt/loomcore/benchmarks/results \
 && rm -f /opt/loomcore/space/samples/fetch_samples.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    LOOMCORE_BUILD_DIR=/opt/loomcore/build \
    LOOMCORE_BENCH_RESULTS=/tmp/loomcore-bench \
    TMPDIR=/tmp
USER user
WORKDIR /tmp
EXPOSE 7860
CMD ["python", "/opt/loomcore/space/server.py"]

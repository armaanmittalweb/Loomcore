"""Serves the live runtime on Modal: the image CI publishes to GHCR, scaled to zero when idle.

    modal secret create loomcore LOOMCORE_PROXY_KEY=<same value as the proxy Worker's PROXY_KEY>
    LOOMCORE_IMAGE_TAG=<commit sha> modal deploy deploy/modal/loomcore_modal.py

The tag defaults to `latest`; pass the commit CI built so a redeploy never reuses a stale pull.

One container at most: the server keeps its locks, rate limits and runtime in memory.
"""

import os

import modal

image = modal.Image.from_registry(f"ghcr.io/armaanmittalweb/loomcore-runtime:{os.environ.get('LOOMCORE_IMAGE_TAG', 'latest')}").env(
    {"LOOMCORE_BUILD_DIR": "/opt/loomcore/build", "LOOMCORE_BENCH_RESULTS": "/tmp/loomcore-bench", "PYTHONUNBUFFERED": "1",
     # gVisor shows the host: name the box and its allotment (cpu=2.0 is 2 physical cores, 4 logical).
     "LOOMCORE_CPU_NAME": "Modal container, x86-64", "LOOMCORE_CPUS": "4"}
)
app = modal.App("loomcore")


@app.function(image=image, cpu=2.0, memory=4096, max_containers=1, scaledown_window=300, timeout=300,
              secrets=[modal.Secret.from_name("loomcore")])
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label="loomcore-runtime")
def web():
    import sys

    sys.path.insert(0, "/opt/loomcore/space")
    from server import create_app

    return create_app()

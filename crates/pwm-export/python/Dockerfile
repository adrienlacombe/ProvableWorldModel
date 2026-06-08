# Real le-wm export: download the pretrained quentinll/lewm-pusht checkpoint,
# quantize the full 192-dim V0 subgraph, and write the prover bundles to /shared.
#
#   docker compose --profile real up --build export predictor-real
#
# Multi-GB image (PyTorch) and a ~70 MB checkpoint download; not the default demo.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch numpy pillow
WORKDIR /app
COPY . /app
CMD ["sh", "scripts/docker_export_real.sh"]

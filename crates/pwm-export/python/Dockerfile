# Optional heavy profile: run the REAL le-wm export end to end (PyTorch).
#
#   docker compose --profile real up --build real-export
#
# Builds the real le-wm V0 subgraph (random init, faithful to the pretrained:false
# config), loads it via torch.load, quantizes every linear, folds the BatchNorm,
# and emits the committed manifest. Multi-GB image; not part of the default demo.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch einops numpy
RUN git clone --depth 1 https://github.com/lucas-maes/le-wm /opt/le-wm
ENV LEWM_PATH=/opt/le-wm
WORKDIR /app
COPY . /app
CMD ["python", "scripts/export_lewm.py"]

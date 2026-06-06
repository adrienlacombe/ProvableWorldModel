# SPDX-License-Identifier: Apache-2.0
"""pwm-export Python side: the offline, trusted export pipeline.

- `canonical`  — byte-identical encoding + commitments (stdlib only; unit-tested
  for parity with the Rust verifier).
- `ingest`     — le-wm V0 subgraph extraction + dim validation (E-201; NumPy core,
  torch only to read a checkpoint).
- `quantize`   — int8 / power-of-two quantization + MAC bounds (E-202; NumPy).
- `fold`       — frozen BatchNorm folding into the preceding Linear (E-203; NumPy).
- `export`     — driver: ingest → fold → quantize → committed manifest (E-205).
- `data_adapter` — stable-worldmodel transitions / latents / candidate sets and
  the committed P2 input bundle (E-208; bundle assembly is NumPy, loaders need swm).

Only `canonical` is imported here so the parity test stays stdlib-only; the
NumPy-backed modules are imported on demand.
"""

from . import canonical  # noqa: F401

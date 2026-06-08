# SPDX-License-Identifier: Apache-2.0
"""Load a real PushT expert episode (frames + actions) from `lerobot/pusht`.

The le-wm dataset is 13 GB (and HDF5 needs the whole file), so it is impractical
for a demo. `lerobot/pusht` is the standard PushT expert dataset: a 0.7 MB parquet
(2D expert actions + agent state) and a 6.9 MB mp4 (96x96 observation frames),
both small and streamable. This gives a consistent real `(observation, action)`
pair from a real expert rollout to feed the demo.

The 10-dim action vector is `[action(2), agent_state(2), 0*6]` normalized to
`[-1, 1]`; the exact le-wm 10-dim layout (block pose, proprio) and its normalizer
live in the 13 GB lewm-pusht dataset. Needs pyarrow, imageio (+ imageio-ffmpeg),
Pillow.
"""
from __future__ import annotations

import io
import tempfile
import urllib.request

import numpy as np
from PIL import Image

BASE = "https://huggingface.co/datasets/lerobot/pusht/resolve/main"
PARQUET = f"{BASE}/data/chunk-000/file-000.parquet"
MP4 = f"{BASE}/videos/observation.image/chunk-000/file-000.mp4"
PUSHT_SCALE = 512.0  # PushT environment coordinate range


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pwm-export"})
    return urllib.request.urlopen(req).read()


def load_episode(history: int = 3, frameskip: int = 5, episode: int = 0):
    """Return `(pil_frames, action_10d, n)` for a real expert episode: `history`
    observation frames spaced by `frameskip`, and the matching 10-dim action
    vectors (real 2D action + agent state, normalized to `[-1, 1]`, the rest zero)."""
    import imageio
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(_fetch(PARQUET)))
    ep = np.array(table.column("episode_index").to_pylist())
    act = np.array(table.column("action").to_pylist(), dtype=np.float32)
    st = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
    rows = np.where(ep == episode)[0][: history * frameskip : frameskip][:history]

    mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    mp4.write(_fetch(MP4))
    mp4.close()
    reader = imageio.get_reader(mp4.name, "ffmpeg")
    frames = [Image.fromarray(reader.get_data(int(i))) for i in rows]
    reader.close()

    a10 = np.zeros((len(rows), 10), dtype=np.float32)
    a10[:, 0:2] = act[rows] / PUSHT_SCALE * 2 - 1
    a10[:, 2:4] = st[rows] / PUSHT_SCALE * 2 - 1
    return frames, a10, len(rows)

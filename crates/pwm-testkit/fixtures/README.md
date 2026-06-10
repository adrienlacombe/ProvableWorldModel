# pwm-testkit fixtures

## `lewm_predictor_real_compact.json`

Compact real-checkpoint predictor regression fixture for issue #178.

- Checkpoint: `quentinll/lewm-pusht` `weights.pt`, revision
  `22b330c28c27ead4bfd1888615af1340e3fe9052`, SHA-256
  `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`,
  MIT license.
- Input episode: `lerobot/pusht`, revision
  `7628202a2180972f291ba1bc6723834921e72c19`, episode `0`, frames
  `[0, 5, 10]`. The parquet SHA-256 is
  `9abc0431f12d10c33b5b37b1bf1e6e557c8ae1c8c29fb86658969e48fcbbbf01`;
  the mp4 SHA-256 is
  `f58d11857651dad5983c021b56a50a68a6ce19068834c1fb5cac099219fb3a78`.
- Input encoding: the three frames are encoded with the checkpoint ViT encoder
  and projector. The matching real 2D action and agent-state rows are normalized
  with the repository PushT adapter rule and encoded with the checkpoint action
  encoder. The fixture keeps the first 8 latent and action coordinates.
- Predictor slice: compact dimensions are `d=8`, `s=3`, `h=2`, `dh=4`,
  `mlp=16`, `depth=2`. Every compact weight is a row and column slice of the
  real checkpoint tensor, preserving Q/K/V head bands and AdaLN chunk structure.
- Quantization: generated with `pwm_export.predictor_quant.bundle_quant`, using
  calibrated SiLU, GELU, inverse-sqrt, and softmax-exp tables. The
  `weights_root` in the JSON is the Rust predictor root for tensor ids `1000+`
  in registration order `adaln,qkv,out,fc1,fc2` with scale ids `10+`.

The fixture is intentionally small enough for default `cargo test --workspace`.
It does not contain the full checkpoint, raw video, or raw parquet data.

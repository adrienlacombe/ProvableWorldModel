# ProvableWorldModel

A succinct-proof system for deterministic, quantized inference of a JEPA-style
world model (LeWorldModel), built as custom Circle-STARK arithmetization (AIR)
over the Mersenne-31 field using a vendored Stwo prover.

The system proves a precise arithmetic relation: given a committed quantized
predictor, a latent history, a goal latent, and a fixed set of candidate action
sequences, the claimed predicted latent trajectories, costs, and the selected
action are exactly the output of the specified fixed-point inference and
planning algorithm. It does not claim to prove floating-point PyTorch
equivalence, physical truth of predictions, or zero-knowledge privacy unless a
specific relation and audit establish those properties.

Status: specification phase. The canonical specification corpus and the
implementation issue set are under active construction. See `SPEC.md` for the
corpus index once the specification branch lands.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

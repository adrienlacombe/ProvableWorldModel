# Contributing

ProvableWorldModel is spec-driven. The specification corpus (`SPEC.md`,
`docs/spec/`, `docs/rfcs/`) is the source of truth; code implements it. Read the
relevant specification and RFC before writing code, and keep the spec and the
code in sync.

## Workflow

1. Every change traces to an issue, and every issue traces to a specification
   section or an RFC. If the work you want to do is not covered, propose the spec
   change first (see "Changing the specification").
2. One issue is one shippable pull request by one contributor in a bounded time.
   If a change does not fit that, decompose it.
3. Branch from `main`. Reference the issue in the pull request. Keep the diff
   focused on the issue's scope.

## Changing the specification

Load-bearing decisions are made in RFCs, not in code review. A change that alters
a public interface, an on-disk format, the proof semantics, the threat model, the
arithmetic, or a cross-cutting contract requires an RFC (see the template in any
file under `docs/rfcs/`). Editorial fixes to the spec do not.

Proof semantics are versioned. Any change to architecture, quantization, rounding,
an approximation, a planner rule, or serialization mints a new `relation_id`; see
[docs/spec/09-release-and-versioning.md](docs/spec/09-release-and-versioning.md).

## Code standards

- Rust: code must pass `cargo fmt --check` and `cargo clippy -- -D warnings`.
- The verifier crate (`pwm-verifier`) must build under `no_std`. Do not add
  `std`-only dependencies to it.
- The verifier must not depend on the export pipeline or on PyTorch.
- Public items carry doc comments stating preconditions, postconditions, and
  errors. Source files carry an `SPDX-License-Identifier: Apache-2.0` header.
- Python (export pipeline): typed, formatted, and linted per the project
  configuration; the fixed-point reference must match the Rust reference
  bit-for-bit.

## Testing

No component is complete without **both** accepting tests (valid witnesses
verify) and rejecting tests (each enumerated invalid witness is rejected with the
correct typed error). See
[docs/spec/07-testing-strategy.md](docs/spec/07-testing-strategy.md) and
[docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md).
New numerical primitives ship with committed golden vectors and stated domain
bounds.

## Continuous integration gates

A pull request must pass: formatting, `clippy -D warnings`, the unit and
integration test suites, golden-vector parity (Python fixed-point vs Rust
fixed-point), the `no_std` verifier build, the documentation build, the SPDX/license
check, and the performance-budget regression check. See
[docs/spec/07-testing-strategy.md](docs/spec/07-testing-strategy.md#ci-gates).

## Security

Do not file public issues for suspected vulnerabilities. See
[SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under the
Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

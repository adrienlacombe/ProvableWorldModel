# Security Policy

ProvableWorldModel is a cryptographic proving system. Its central security
property is **soundness**: a verifier must not accept a proof of a false
statement. A soundness break is the most severe class of defect in this project,
above availability or performance.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Report it privately
through GitHub's private vulnerability reporting ("Report a vulnerability" under
the repository Security tab). If that channel is unavailable, contact a
maintainer privately and request a secure channel before sharing details.

Include, where possible: the affected component or RFC, the relation_id and
versions involved, a minimal reproduction (manifest, public input, and witness or
artifact), and the security property you believe is violated.

## Scope and severity

| Class | Examples | Severity |
|---|---|---|
| Soundness | A constructed proof that a conformant verifier accepts for a false statement; field-wraparound or unrange-checked value admitting an invalid witness; an unbound manifest field that lets a prover change semantics | Critical |
| Binding | A commitment that fails to bind a value the soundness argument assumes is bound | Critical |
| Privacy (when ZK mode is claimed) | Witness or private-weight leakage from a proof, trace, lookup, or serialization that is advertised as hiding | High when ZK is claimed |
| Determinism | Inputs that produce non-reproducible proof inputs or claimed outputs across conformant runs | High |
| Availability | Prover/verifier crash, resource exhaustion | Medium |

V0 is a succinct validity proof and is **not** zero-knowledge. Reports asserting
a privacy break against V0 are out of scope unless a specific relation explicitly
advertised a hiding property. See
[docs/spec/06-security.md](docs/spec/06-security.md#privacy-and-zk).

## Process and disclosure

Maintainers acknowledge a report within a target of five business days and
propose a remediation and disclosure timeline. Coordinated disclosure is the
default; an embargo holds until a fix ships or by mutual agreement. Fixes that
change proof semantics mint a new `relation_id`; see
[docs/spec/09-release-and-versioning.md](docs/spec/09-release-and-versioning.md#relation-versioning).

No public soundness claim is made for any relation that has not passed the
security-review gate defined in
[docs/spec/07-testing-strategy.md](docs/spec/07-testing-strategy.md) and
[docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md).

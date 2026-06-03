# Observability: Logging, Metrics, Tracing, Redaction

Status: Normative specification. Role: defines the structured-logging schema, the
named metric set, the tracing span tree, and the binding redaction rules for the
ProvableWorldModel export, proving, and verification pipelines. This document is
authoritative for every observable signal a `pwm` process emits. It is binding on
`pwm-export`, `pwm-prover`, and `pwm-verifier`, and on the AIR trace builders in
`pwm-air`/`pwm-circuits`. Where it constrains witness handling it is subordinate
to the soundness rules in
[`docs/spec/06-security.md#privacy-and-zk`](06-security.md#privacy-and-zk); where
it names errors it defers to
[`docs/spec/04-error-model.md#error-taxonomy`](04-error-model.md#error-taxonomy).

## Scope and design stance

Observability in a proving system has a hard constraint that ordinary services do
not: **a log line can break soundness or privacy as surely as a missing
constraint.** A STARK proof attests to an arithmetic relation over a witness that
includes quantized weights, intermediate activations, attention scores, and
lookup multiplicities. None of those values is safe to emit when
`weights.visibility=private_committed` or when the system runs in a future ZK
mode. Conversely, the values that *are* safe to emit — commitments, hashes,
shapes, counts, timings, and the public inputs themselves — are exactly the
values an auditor needs to reproduce a run and bind a proof to its claimed model.

This document therefore treats observability as a typed, redaction-aware
contract, not as ad-hoc `println!`. Three principles govern every signal:

1. **Commitments, not contents.** Logs and metrics carry the 32-byte commitments,
   hashes, shapes, counts, and timings of an object — never the object's private
   field values. A reader learns *that* a tensor of shape `[5, 192]` was committed
   under root `0x9f3c…`, never *which* values it held.
2. **Determinism is observable.** Every proof run emits a single `determinism_digest`
   (defined in [Reproducibility and audit hook](#reproducibility-and-audit-hook))
   that an independent party can recompute from public inputs, manifest
   commitments, and tool versions. Two runs that claim the same thing must log the
   same digest; a digest mismatch is a first-class reproducibility failure.
3. **Failure is observable.** Metrics and the audit record are emitted on *every*
   terminal path, success or failure. A run that aborts at trace build still emits
   the metrics it had collected and an audit record with `outcome=error` and the
   triggering [error code](04-error-model.md#error-taxonomy). Silence is not an
   acceptable failure mode.

The signal backbone is the Rust `tracing` crate ecosystem (spans + structured
events) bridged to a JSON formatter for logs and to an in-process registry for
metrics. The Python side of `pwm-export` mirrors the same field names and the
same redaction rules over the standard `logging` module with a JSON formatter, so
that export logs and prover logs are queryable with one schema.

### Components that emit signals

| Component        | Crate / language        | Emits logs | Emits metrics | Root span(s)                              |
| ---------------- | ----------------------- | :--------: | :-----------: | ----------------------------------------- |
| Export pipeline  | `pwm-export` (Py + Rust)|     yes    |      yes      | `export`, `quantize`, `reference_inference` |
| Trace builder    | `pwm-prover`/`pwm-air`  |     yes    |      yes      | `trace_build.<component>`, `interaction_trace` |
| Prover           | `pwm-prover`            |     yes    |      yes      | `commit`, `fiat_shamir`, `fri`            |
| Verifier         | `pwm-verifier`          |     yes    |      yes      | `verify`                                  |

`pwm-core` does not own a process; it provides the redaction helpers, the
`determinism_digest` function, and the canonical serializer
([RFC-0014](../rfcs/RFC-0014-canonical-serialization-and-transcript.md)) that all
emitters call. The CLI shapes and artifact bundle that wrap these processes are
specified in
[`docs/spec/02-public-api.md#cli`](02-public-api.md#cli) and
[RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md).

---

## Logging

### Record schema

All log records are single-line JSON objects (newline-delimited JSON, one record
per line). The schema is versioned and stable; new optional fields may be added
under [the stability policy](02-public-api.md#stability-policy) but existing
fields never change meaning. The canonical record type is owned by `pwm-core`:

```rust
/// One structured log record. Serialized as a single line of JSON.
/// `pwm-core::obs::LogRecord`. All `*_commitment`/`*_hash` fields are lowercase
/// hex of 32 bytes. No field of this struct may ever hold a private witness value
/// (see INV-OBS-01).
pub struct LogRecord {
    pub log_schema_version: u32,     // = 1 for this spec
    pub ts: String,                  // RFC3339 UTC, e.g. "2026-06-03T12:00:00.123Z"
    pub level: Level,                // ERROR | WARN | INFO | DEBUG | TRACE
    pub stage: Stage,                // enum below; the pipeline stage emitting this
    pub span_id: u64,                // tracing span id; joins logs to spans
    pub parent_span_id: Option<u64>,
    pub run_id: [u8; 16],            // ULID for this process invocation (hex)
    pub relation_id: Option<String>, // "pwm.lewm.<statement>.v<N>" once known
    pub statement_type: Option<StatementType>,
    pub model_commitment: Option<[u8; 32]>,
    pub quantization_commitment: Option<[u8; 32]>,
    pub planner_config_commitment: Option<[u8; 32]>,
    pub component: Option<String>,   // AIR component name, e.g. "linear", "attention"
    pub shapes: Option<Vec<Shape>>,  // shapes of tensors touched; values NEVER included
    pub counts: Option<Counts>,      // rows/cols/lookups/ranges; see metrics section
    pub timing_ms: Option<f64>,      // wall time of the unit of work, if terminal
    pub outcome: Option<Outcome>,    // ok | error (set on terminal events)
    pub error_code: Option<String>,  // e.g. "PWM-VERIFY-0007"; see 04-error-model.md
    pub message: String,             // human-readable; MUST be value-free (INV-OBS-01)
    pub fields: BTreeMap<String, LoggableValue>, // structured extras, redaction-checked
}

pub enum Stage {
    Export, Quantize, ReferenceInference, ManifestWrite,
    TraceBuild, Commit, FiatShamir, InteractionTrace, Fri, Prove, Verify,
}

pub enum Outcome { Ok, Error }

pub struct Shape { pub tensor_id: u32, pub dims: Vec<u32>, pub scale_id: u32 }

/// The ONLY value variants permitted in a log record. There is deliberately no
/// `BoundedInt`, `Tensor`, `M31`, or raw-bytes-blob variant: a witness value
/// cannot be constructed as a LoggableValue. This is enforced by the type system
/// (INV-OBS-02), not by reviewer discipline.
pub enum LoggableValue {
    Count(u64),
    Bytes32Hex(String),   // a commitment or hash, hex-encoded
    Label(String),        // an enum label, op name, or relation_id — NOT a value
    Duration(f64),        // milliseconds
    Bool(bool),
}
```

`message` is a fixed, value-free string (e.g. `"linear component trace built"`),
never an interpolation of witness data. Variable detail lives in typed `fields`
constrained to `LoggableValue`, so the schema cannot smuggle a private integer
through a free-form string. `Counts` reuses the metric field names from
[Metrics](#metrics) so that a count appearing in a log line is the same quantity
the registry aggregates.

### Level policy

Levels are policy, not vibes. The default emit threshold is `INFO`. `DEBUG` and
`TRACE` are opt-in via configuration and are still bound by the redaction rules —
a lower threshold reveals *more events*, never *more value content*.

| Level   | What belongs here                                                                              | Example                                                              |
| ------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `ERROR` | A terminal failure of the run. Always paired with an `error_code` from the taxonomy.           | manifest hash mismatch; FRI verification failed; range check out of bound |
| `WARN`  | A recoverable anomaly or a fallback that changes cost but not correctness.                      | accumulator promoted to limb decomposition; lookup multiplicity near table capacity |
| `INFO`  | Stage transitions and per-stage summaries: start/end of each span, the metric set on completion.| "rollout trace built", with `trace_rows`, `trace_cols`, `timing_ms` |
| `DEBUG` | Per-component detail: per-op row counts, per-layer timings, challenge-derivation step labels.   | "attention head 7 scores committed", `counts.lookup_multiplicity_total` |
| `TRACE` | Per-row / per-cell control-flow labels for debugging the trace builder. Never values.          | "tensor_memory write at tensor_id=12 index=[3,0,0,0] time=88" (indices/ids only) |

Note the `TRACE` example: tensor *coordinates* (`tensor_id`, `index`, `scale_id`,
`time`) are loggable structural metadata; the cell's `value: M31` is not. See
[Redaction](#redaction) for the exhaustive loggable/forbidden split.

### Per-stage required fields

Each stage MUST emit at least one terminal `INFO` record on success and one
`ERROR` record on failure, carrying the fields below. "Required" means the field
is non-`None` on the terminal record of that stage.

| Stage (`stage`)        | Required fields on terminal record                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `Export`               | `model_commitment`, `relation_id`, `counts.tensor_count`, `shapes` (per exported tensor), `timing_ms`, `outcome`     |
| `Quantize`             | `model_commitment`, `quantization_commitment`, `counts.requant_op_count`, `timing_ms`, `outcome`                     |
| `ReferenceInference`   | `model_commitment`, `relation_id`, `statement_type`, `claimed_output_commitment`, `timing_ms`, `outcome`             |
| `ManifestWrite`        | `model_commitment`, `quantization_commitment`, `planner_config_commitment`, `relation_id`, `timing_ms`, `outcome`    |
| `TraceBuild`           | `component`, `counts.trace_rows`, `counts.trace_cols`, `counts.range_check_count`, `counts.lookup_multiplicity_total`, `timing_ms`, `outcome` |
| `Commit`               | `counts.trace_rows`, `counts.trace_cols`, the per-tree Merkle `root` as `Bytes32Hex`, `timing_ms`, `outcome`         |
| `FiatShamir`           | `relation_id`, the public-input digest as `Bytes32Hex`, `counts.challenge_count`, `timing_ms`, `outcome`             |
| `InteractionTrace`     | `counts.lookup_multiplicity_total`, `counts.trace_cols`, `timing_ms`, `outcome`                                      |
| `Fri`                  | `counts.fri_query_count`, `counts.fri_layers`, `counts.proof_size_bytes`, `timing_ms`, `outcome`                     |
| `Prove`                | `relation_id`, `model_commitment`, `claimed_output_commitment`, `counts.proof_size_bytes`, `determinism_digest`, `timing_ms`, `outcome` |
| `Verify`               | `relation_id`, `model_commitment`, `claimed_output_commitment`, `determinism_digest`, `outcome`, `error_code` (on reject), `timing_ms` |

The fields a log carries are exactly commitments, hashes, shapes, counts,
timings, enum labels, and error codes — never private values. This is restated as
[INV-OBS-01](#named-invariants) and made unrepresentable by the `LoggableValue`
type ([INV-OBS-02](#named-invariants)).

### Example records

A successful trace-build summary (INFO):

```json
{"log_schema_version":1,"ts":"2026-06-03T12:00:01.482Z","level":"INFO",
 "stage":"TraceBuild","span_id":418,"parent_span_id":12,
 "run_id":"01J9X4Z7P3K8Q2","relation_id":"pwm.lewm.fixed_candidate_planning.v1",
 "statement_type":"P2FixedCandidatePlanning","component":"linear",
 "model_commitment":"3a9f…(64 hex)…","counts":{"trace_rows":262144,
 "trace_cols":17,"range_check_count":786432,"lookup_multiplicity_total":1310720},
 "timing_ms":214.7,"outcome":"ok","message":"linear component trace built"}
```

A verifier rejection (ERROR):

```json
{"log_schema_version":1,"ts":"2026-06-03T12:00:09.001Z","level":"ERROR",
 "stage":"Verify","span_id":3,"run_id":"01J9X4ZB1M…",
 "relation_id":"pwm.lewm.fixed_candidate_planning.v1",
 "model_commitment":"3a9f…","claimed_output_commitment":"77c2…",
 "determinism_digest":"e10b…","outcome":"error","error_code":"PWM-VERIFY-0007",
 "message":"argmin tie-break constraint not satisfied"}
```

Neither record contains a weight, an activation, a cost value, or a latent — only
commitments, shapes, counts, an error code, and a value-free message.

---

## Metrics

### Named metric set

Metrics are a fixed, named set. Each metric has a stable name, a type
(`gauge`/`counter`/`histogram`), a unit, the labels it carries, and a defined
collection point. Names are snake_case and stable under
[the stability policy](02-public-api.md#stability-policy). Counters are
monotonic within a run; gauges are point-in-time; histograms record a
distribution over the run.

| Metric                        | Type      | Unit    | Labels                                  | Meaning / collection                                                                 |
| ----------------------------- | --------- | ------- | --------------------------------------- | ------------------------------------------------------------------------------------ |
| `trace_rows`                  | gauge     | rows    | `component`, `relation_id`              | Row count of a component's main trace. Set by the trace builder at the close of `trace_build.<component>`. Emitted per component and as a `component="_total"` aggregate. |
| `trace_cols`                  | gauge     | columns | `component`, `trace_kind`               | Column count, labeled `trace_kind` ∈ `{preprocessed, main, interaction}`. Set per component per trace kind. |
| `proof_size_bytes`            | gauge     | bytes   | `relation_id`                           | Serialized size of `Proof` bytes in the emitted `ProofArtifact`. Measured by the prover after serialization, before write. |
| `prove_time_seconds`          | histogram | seconds | `relation_id`, `phase`                  | Wall time, sampled per `phase` ∈ `{trace_build, commit, fiat_shamir, interaction_trace, fri, total}`. Recorded from span durations on the prove path. |
| `verify_time_seconds`         | histogram | seconds | `relation_id`, `phase`                  | Wall time on the verify path, `phase` ∈ `{deserialize, public_input_digest, fri, app_checks, total}`. Recorded from `verify` span durations. |
| `peak_memory_bytes`           | gauge     | bytes   | `process`                               | Peak resident set of the process, `process` ∈ `{export, prover, verifier}`. Sampled at each span close; the max is reported at process exit (including on failure). |
| `lookup_multiplicity_total`   | counter   | entries | `table`, `component`                    | Sum of LogUp multiplicities over all lookup uses, labeled by `table` ∈ `{u8, i8, u16, bounded_limb, gelu, silu, softmax, inv_sqrt, tensor_memory, weight_table}`. Incremented during `interaction_trace`. |
| `range_check_count`           | counter   | checks  | `range`, `component`                    | Number of bounded-integer values wired to a range relation, labeled `range` ∈ `{u8, i8, u16, bounded_limb}`. Incremented during `trace_build.<component>`. |
| `fri_query_count`             | gauge     | queries | `relation_id`                           | Number of FRI queries actually performed (the soundness-driven query count). Set at the close of the `fri` span. |

Two collection facts are normative:

- **`peak_memory_bytes` and `*_time_seconds` are emitted even on failure paths.**
  A process that aborts mid-`fri` still reports the peak memory it reached and the
  durations of the spans that closed. This is [INV-OBS-04](#named-invariants).
- **`trace_rows`/`trace_cols`/`range_check_count`/`lookup_multiplicity_total`**
  are derived from the witness *structure* (shapes, op counts, table sizes), never
  from witness *values*. Counting how many `i8` range checks a layer performs does
  not reveal any checked value. This keeps the metric set inside the redaction
  boundary ([INV-OBS-03](#named-invariants)).

### Metric record type and emission

Metrics are exposed two ways from the same in-process registry:

1. **As structured log events** — the terminal `INFO` record of each stage embeds
   the relevant counts in `LogRecord.counts`, so a log stream alone is sufficient
   to reconstruct the metric set for a run.
2. **As a machine-readable snapshot** — at process exit the registry serializes a
   `MetricsSnapshot` to the run's audit directory and (optionally) to a Prometheus
   text-format endpoint when `--metrics-endpoint` is configured on the CLI.

```rust
/// `pwm-core::obs::MetricSample`. The wire form of one metric observation.
pub struct MetricSample {
    pub name: String,                  // one of the names in the table above
    pub kind: MetricKind,              // Gauge | Counter | Histogram
    pub unit: String,                  // "rows" | "bytes" | "seconds" | ...
    pub labels: BTreeMap<String, String>,
    pub value: f64,                    // gauge/counter scalar
    pub buckets: Option<Vec<(f64, u64)>>, // histogram (upper_bound, cumulative_count)
}

pub struct MetricsSnapshot {
    pub run_id: [u8; 16],
    pub relation_id: Option<String>,
    pub outcome: Outcome,              // ok | error — set even when the run failed
    pub samples: Vec<MetricSample>,
}
```

`MetricSample.labels` may carry only the label keys enumerated above. A label
value is always an enum label, a component name, or a `relation_id` — never a
witness value. There is no `value`-typed label, by construction.

The performance targets and budgets these metrics are checked against (and the
regression gates that consume `prove_time_seconds`, `proof_size_bytes`, and
`peak_memory_bytes`) live in
[`docs/spec/08-performance-budget.md#perf-gates`](08-performance-budget.md#perf-gates).
This document defines *what* is measured and *how it is collected*; the budget
document defines the thresholds.

---

## Tracing

### Span tree

The pipeline is instrumented with `tracing` spans whose names and nesting are
normative, so that a flamegraph or trace viewer reads identically across runs and
contributors. Each span carries the same identifying fields as a `LogRecord`
(`relation_id`, `model_commitment`, etc.) as span attributes; child events
inherit them. Span attributes obey the same redaction rules as log fields
([INV-OBS-01](#named-invariants)) — a span may carry shapes and counts, never
values.

The canonical span set and their parent/child structure:

```text
export                                  (pwm-export, Python+Rust)
└── quantize                            per-tensor quantization of the exported graph
    └── reference_inference             deterministic fixed-point reference run
                                        (produces claimed_output_commitment + golden vectors)

prove_planning | prove_rollout          (pwm-prover root; one per CLI invocation)
├── trace_build.<component>             one span PER AIR component, in execution order:
│     trace_build.range_check
│     trace_build.tensor_memory
│     trace_build.linear
│     trace_build.matmul
│     trace_build.requant
│     trace_build.activation_lookup
│     trace_build.layernorm
│     trace_build.attention
│     trace_build.mlp
│     trace_build.predictor
│     trace_build.rollout
│     trace_build.cost
│     trace_build.argmin
│     (trace_build.cem — P3 only, Future)
├── commit                             Merkle commitment of preprocessed + main traces
├── fiat_shamir                        challenge derivation in canonical channel order
├── interaction_trace                  LogUp interaction-trace construction from challenges
└── fri                                FRI commit + query phase, proof serialization

verify                                  (pwm-verifier root; one per invocation)
├── public_input_digest                recompute the public-input digest (RFC-0014)
├── fri                                FRI verification
└── app_checks                         selected-index range, shape, commitment checks
```

The prose statement of the same tree, for readers who do not render the diagram:
export is the root of the offline path; under it, `quantize` produces the
quantized graph, and `reference_inference` runs the deterministic fixed-point
reference. On the proving path, the prover root (`prove_planning` or
`prove_rollout`) has one `trace_build.<component>` child per AIR component, built
in the execution order listed above, followed in sequence by `commit`,
`fiat_shamir`, `interaction_trace`, and `fri`. On the verification path, the
`verify` root has three children: `public_input_digest`, `fri`, and `app_checks`.

The component list under `trace_build.<component>` is exactly the AIR component
set defined for `pwm-air`
([`docs/spec/01-architecture.md#component-model`](01-architecture.md#component-model)).
`trace_build.cem` exists only when a P3/CEM statement is proven and is out of V0
scope.

### Span attributes and timing

| Span                       | Required attributes                                                             | Closes when                                  |
| -------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------- |
| `export`                   | `model_commitment`, `relation_id`                                               | manifest + golden vectors written            |
| `quantize`                 | `quantization_commitment`                                                       | all tensors quantized                        |
| `reference_inference`      | `statement_type`, `claimed_output_commitment`                                   | reference run done, outputs committed        |
| `trace_build.<component>`  | `component`, `trace_rows`, `trace_cols`, `range_check_count`, `lookup_multiplicity_total` | the component's main trace is materialized |
| `commit`                   | per-tree Merkle `root` (`Bytes32Hex`), `trace_rows`, `trace_cols`              | commitments derived                          |
| `fiat_shamir`              | public-input digest (`Bytes32Hex`), `challenge_count`                          | all challenges derived in canonical order    |
| `interaction_trace`        | `lookup_multiplicity_total`                                                     | interaction columns built                    |
| `fri`                      | `fri_query_count`, `fri_layers`, `proof_size_bytes`                            | proof serialized                             |
| `verify`                   | `relation_id`, `model_commitment`, `determinism_digest`, `outcome`             | accept or first rejection                    |

Each span's wall-clock duration feeds the `prove_time_seconds` /
`verify_time_seconds` histograms under the matching `phase` label. The
`fiat_shamir` span's canonical channel ordering is the same ordering specified in
[RFC-0014](../rfcs/RFC-0014-canonical-serialization-and-transcript.md); the span
exists so that a divergence between prover and verifier channel order is visible
in a trace, not only as a downstream FRI failure.

---

## Redaction

### The boundary

Redaction is governed by two manifest/runtime facts:

- `weights.visibility` in the manifest — `public` or `private_committed`
  ([`docs/spec/03-data-model.md#model-manifest`](03-data-model.md#model-manifest)).
- The runtime privacy mode — `validity` (V0 default) or `zk` (Future, gated on the
  hiding audit in
  [`docs/spec/06-security.md#privacy-and-zk`](06-security.md#privacy-and-zk)).

**When `weights.visibility=private_committed`, or whenever the process runs in ZK
mode, the prover, trace builder, and all log/metric/span emitters MUST NOT emit
weight or witness values. Only commitments, shapes, counts, and timings may be
emitted.** This is the central redaction rule and is named
[INV-OBS-01](#named-invariants). It holds at every log level, including `TRACE`:
lowering the threshold reveals more *events*, never more *value content*.

Even when `weights.visibility=public`, the redaction rules below still apply to
all *non-weight* private witnesses (activations, attention scores, costs,
argmin diffs, latents not present in `PublicInput`). Public weights are loggable
as commitments and shapes regardless; their raw values are still not logged,
because logging large tensors is a footgun and because a future flip to
`private_committed` must not require re-auditing the log surface.

### Exhaustive loggable / forbidden split

| Object                                              | Loggable (always)                                              | Forbidden to log (when private / ZK; see notes)                        |
| --------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Model weights (`QuantizedWeights`)                  | `commitment` (root), per-tensor `shape`, `scale_id`, tensor count | every `BoundedInt.value` in `data`; any limb of any weight             |
| Biases                                              | shape, count, commitment                                        | bias values                                                            |
| Latent history / goal latent                        | commitment; **public** components only if present in `PublicInput.*_public` | values when carried only as `*_commitment` (private)                   |
| Candidate actions                                   | commitment, count `S`, per-candidate length                     | action values when carried only as `candidate_actions_commitment`      |
| Intermediate activations (`predictor_activations`)  | per-tensor shape, count, `scale_id`                             | every activation value                                                 |
| Attention scores / probabilities                    | shape, head count, sequence length                              | every score, every softmax probability                                 |
| Accumulators / products / quotients / remainders    | column count, range label, `range_check_count`                 | every accumulator/product/quotient/remainder value                     |
| Costs (`costs`) and argmin diffs (`ArgminWitness.diffs`) | count `S`; `selected_index` and `selected_cost` **iff** they are public inputs | every `cost_s` value; every diff value; the cost ordering, when private |
| Tensor memory cells (`TensorCell`)                  | `tensor_id`, `index`, `scale_id`, `time` (structural coordinates) | `value: M31` of any cell                                               |
| Range witnesses (`RangeWitness`)                    | which `range` table, multiplicity totals                        | the checked value, the limb decomposition                             |
| Lookup witnesses (`LookupWitness`)                  | which `table`, total multiplicity, table *commitment*           | per-entry lookup keys/values; **private table contents** (see below)   |
| Public inputs (`PublicInput`)                       | every field is loggable — these are public by definition        | (none)                                                                 |
| Proof bytes (`Proof`)                               | `proof_size_bytes`, schema version                              | (proof bytes are public; not redacted, but not logged verbatim)        |

Notes on the table:

- "Forbidden to log (when private / ZK)" means: forbidden whenever the value is
  carried only as a commitment (the private encoding), and unconditionally
  forbidden in ZK mode. A latent that the statement publishes via
  `latent_history_public` is, by definition, public and loggable.
- **Lookup arguments must not leak private table contents.** A LogUp lookup binds
  a witness column to a table; logging the *table commitment* and the *total
  multiplicity* is safe, but logging the table's *entries* or the *per-row keys*
  reveals exactly the private values the lookup was meant to constrain. The
  activation/normalization tables (gelu, silu, softmax, inv_sqrt) are committed in
  the manifest and their commitments are loggable; their contents are loggable
  only when the table itself is public manifest data, never when reconstructed
  from private witness columns. This is [INV-OBS-05](#named-invariants).

### Enforcement, not etiquette

Redaction is enforced structurally so that a careless `info!()` cannot leak:

1. **Type-level barrier.** Log fields and span attributes accept only
   `LoggableValue` (counts, 32-byte hex, labels, durations, bools). There is no
   conversion from `BoundedInt`, `Tensor`, `M31`, `TensorCell.value`, or a raw
   byte blob into `LoggableValue`. A developer who tries to log a witness value
   gets a compile error, not a runtime leak. This is
   [INV-OBS-02](#named-invariants).
2. **Redaction helper.** `pwm-core::obs::commit_for_log(&T) -> Bytes32Hex` is the
   only sanctioned way to put a private object into a log; it returns the object's
   commitment, never its contents. Witness structs implement a `LogShape` trait
   that yields shape/count/scale metadata and *cannot* yield values.
3. **Test gate.** A redaction test (owned by
   [`docs/spec/07-testing-strategy.md#negative-tests`](07-testing-strategy.md#negative-tests))
   runs the full prove path with `weights.visibility=private_committed`, captures
   every log record, metric sample, and span attribute, and asserts that no
   emitted byte sequence equals any witness value's canonical encoding. A leak
   fails CI ([INV-OBS-06](#named-invariants)).

### Failure modes (redaction)

| Failure mode                                                          | System response                                                                                                   |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Code attempts to log a `BoundedInt`/`Tensor`/`M31` value directly     | Compile error (no `LoggableValue` conversion exists). [INV-OBS-02](#named-invariants).                            |
| A `message` string is built by interpolating a witness value          | Caught by the redaction test ([INV-OBS-06](#named-invariants)); `message` must be a constant value-free string.   |
| Manifest sets `private_committed` but a span attribute carries a value | Redaction test fails the run in CI; at runtime, the emit path refuses non-`LoggableValue` attributes (type-level). |
| Lookup builder logs reconstructed table entries                        | [INV-OBS-05](#named-invariants) violated; redaction test detects per-entry values in the log stream and fails CI. |
| ZK mode active but emitter not redaction-aware                         | The emitter consults the runtime privacy mode; in `zk` mode the loggable set is identical to `private_committed`. Any value emission is a type error or a test failure. |

---

## Reproducibility and audit hook

### Determinism digest

Every proof run — prove and verify — computes one `determinism_digest`, a 32-byte
hash over the canonical encoding (RFC-0014) of: the public inputs, the manifest
commitments, and the tool versions. It is the single observable value that lets
an independent party confirm two runs claim the same arithmetic relation under
the same toolchain. It is emitted on the terminal `Prove`/`Verify` log record and
in the audit record.

```rust
/// `pwm-core::obs::determinism_digest`. Domain-separated Blake2s over the
/// RFC-0014 canonical serialization of its inputs, in this fixed order.
/// Hashing public inputs + commitments + versions ONLY — never any witness value
/// (INV-OBS-07).
pub struct DeterminismInputs<'a> {
    pub relation_id: &'a str,                  // "pwm.lewm.<statement>.v<N>"
    pub public_input: &'a PublicInput,         // canonical bytes; all fields public
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],
    pub manifest_canonical_json_hash: [u8; 32],
    pub artifact_version: u32,
    pub log_schema_version: u32,
    pub tool_versions: ToolVersions,
}

pub struct ToolVersions {
    pub pwm_version: String,        // workspace semver, e.g. "0.1.0"
    pub pwm_git_rev: [u8; 20],      // git commit of the pwm build
    pub stwo_revision: [u8; 20],    // pinned rev from third_party/stwo/REVISION (RFC-0015)
    pub stwo_circuits_revision: [u8; 20], // pinned rev from third_party/stwo-circuits/REVISION
    pub rustc_version: String,      // MSRV-pinned toolchain, see 09-release-and-versioning.md#msrv
}

pub fn determinism_digest(inputs: &DeterminismInputs) -> [u8; 32];
```

The digest deliberately includes the pinned `stwo`/`stwo-circuits` revisions
(from `third_party/<crate>/REVISION` per
[RFC-0015](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md)) and the
MSRV-pinned `rustc_version`
([`docs/spec/09-release-and-versioning.md#msrv`](09-release-and-versioning.md#msrv)),
because a proof's reproducibility is contingent on the exact prover substrate, not
only on the inputs. A digest mismatch between a claimed run and a re-run is a
reproducibility failure surfaced as error class
`PWM-REPRO-*` in
[`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes).

### Audit record

Every terminal proof run writes exactly one audit record to the run's audit
directory (path defined by the CLI in
[RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md)). The
audit record binds `relation_id` + all commitments + `artifact_version` and is
the immutable receipt an auditor reconciles against a `ProofArtifact`.

```rust
/// `pwm-core::obs::AuditRecord`. Written once per run on EVERY terminal path,
/// success or failure (INV-OBS-08). Contains only public/commitment data.
pub struct AuditRecord {
    pub audit_schema_version: u32,        // = 1
    pub run_id: [u8; 16],
    pub ts: String,                       // RFC3339 UTC
    pub role: Role,                       // Prover | Verifier | Exporter
    pub relation_id: String,
    pub statement_type: StatementType,
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],
    pub claimed_output_commitment: [u8; 32],
    pub artifact_version: u32,            // ProofArtifact.artifact_version
    pub determinism_digest: [u8; 32],
    pub tool_versions: ToolVersions,
    pub outcome: Outcome,                 // ok | error
    pub error_code: Option<String>,       // set when outcome == Error
    pub metrics_summary: Vec<MetricSample>, // proof_size, times, peak_memory, fri_query_count
}
```

The audit record carries `metrics_summary` so that one file answers both "what was
proven, under what toolchain, with what result" and "what did it cost". It binds
`relation_id` + the four commitments + `artifact_version` so that an auditor can
confirm a `ProofArtifact` on disk is the one this run produced and that its
`relation_id` matches the manifest's. The full bit-for-bit reproducibility
contract — same inputs and toolchain produce the same proof inputs/outputs and the
same `determinism_digest` — is locked in
[RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md); this
document specifies the observable evidence (the digest and the audit record) that
makes that contract checkable.

### Failure modes (audit and reproducibility)

| Failure mode                                                       | System response                                                                                              |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Run aborts before emitting an audit record                          | A panic hook / `Drop` guard flushes a partial `AuditRecord` with `outcome=Error` and the last known fields. [INV-OBS-08](#named-invariants). |
| `determinism_digest` differs from a prior run with same inputs      | `PWM-REPRO-*` error; the differing `ToolVersions`/commitment field is logged at `ERROR` to localize the drift. |
| Audit directory not writable                                        | `ERROR` log with the I/O error code; the run still returns its proof/verify result, but is flagged non-auditable (audit write failure does not silently change the proof outcome). |
| `tool_versions` cannot be resolved (missing `REVISION` pin file)    | Run aborts before proving with a config error; an unpinned substrate cannot produce a reproducible digest (RFC-0015). |

---

## Named invariants

These invariants are binding. Each is testable; the test that enforces it lives in
[`docs/spec/07-testing-strategy.md`](07-testing-strategy.md).

| Invariant     | Statement                                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INV-OBS-01    | No log record, metric sample, or span attribute at any level (including `TRACE`) contains a private witness value. Only commitments, hashes, shapes, counts, timings, enum labels, and error codes are emitted. |
| INV-OBS-02    | A private witness value is not *representable* as a loggable field: there is no conversion from `BoundedInt`/`Tensor`/`M31`/`TensorCell.value`/raw-bytes into `LoggableValue`. Redaction is a type-level guarantee, not reviewer discipline. |
| INV-OBS-03    | Every metric in the named set is derived from witness *structure* (shapes, op counts, table sizes) or from public timings/sizes — never from witness *values*. |
| INV-OBS-04    | Metrics, including `peak_memory_bytes` and the `*_time_seconds` histograms, are emitted on every terminal path, success or failure. A failed run still reports the costs it incurred. |
| INV-OBS-05    | Lookup arguments do not leak private table contents: table *commitments* and total *multiplicities* are loggable; per-entry keys/values reconstructed from private witness columns are not. |
| INV-OBS-06    | The redaction CI gate runs the full prove path under `weights.visibility=private_committed`, captures all logs/metrics/span attributes, and asserts no emitted bytes equal any witness value's canonical encoding. A leak fails the build. |
| INV-OBS-07    | `determinism_digest` is computed over public inputs + manifest commitments + tool versions only; no witness value is an input to the digest. |
| INV-OBS-08    | Exactly one `AuditRecord` is written per terminal run, on both success and failure paths, via a flush guard that survives panics. The record binds `relation_id` + all four commitments + `artifact_version`. |
| INV-OBS-09    | The set of log levels at which an event is emitted may change with configuration; the *value content* of emitted records is invariant to the level threshold (it is governed by INV-OBS-01, not by the level). |

## Failure-mode summary (observability subsystem)

Beyond the per-section tables above, the observability subsystem as a whole has
these terminal failure modes and responses:

| Failure mode                                          | System response                                                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Log sink unavailable (stdout/file closed)             | Emit to stderr fallback; never block proving on log I/O; record the sink error once at `WARN`.               |
| Metrics endpoint unreachable                          | Continue; the `MetricsSnapshot` is still written to the audit directory. Endpoint failure never aborts a run. |
| Span instrumentation overhead exceeds budget          | `DEBUG`/`TRACE` spans are compiled out in release builds via the `tracing` level filter; `INFO` spans remain. |
| Clock skew makes `ts` non-monotonic                   | `ts` is informational; ordering for audit uses `time` on `TensorCell` and span ids, not wall-clock. Logged at `WARN` if detected. |
| `log_schema_version` of a consumed log differs        | Consumers reject unknown major schema versions; this document defines version `1`. Bumps follow [the stability policy](02-public-api.md#stability-policy). |

## Cross-references

- Error codes and the verifier rejection taxonomy referenced by `error_code`:
  [`docs/spec/04-error-model.md#error-taxonomy`](04-error-model.md#error-taxonomy),
  [`docs/spec/04-error-model.md#verifier-rejections`](04-error-model.md#verifier-rejections),
  [`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes).
- Canonical types referenced by the schemas here (`PublicInput`, `Witness`,
  `Tensor`, `BoundedInt`, `QuantizedWeights`, `TensorCell`, `ArgminWitness`,
  `StatementType`, manifest):
  [`docs/spec/03-data-model.md`](03-data-model.md).
- AIR component set used as `component` labels and `trace_build.<component>` spans:
  [`docs/spec/01-architecture.md#component-model`](01-architecture.md#component-model),
  [`docs/spec/01-architecture.md#trace-model`](01-architecture.md#trace-model).
- Privacy/ZK boundary that activates the strict redaction set:
  [`docs/spec/06-security.md#privacy-and-zk`](06-security.md#privacy-and-zk).
- Performance thresholds consuming these metrics:
  [`docs/spec/08-performance-budget.md#perf-gates`](08-performance-budget.md#perf-gates).
- Reproducibility contract and CLI/artifact layout:
  [RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md),
  [`docs/spec/02-public-api.md#cli`](02-public-api.md#cli),
  [`docs/spec/02-public-api.md#artifact-formats`](02-public-api.md#artifact-formats).
- Canonical serialization and Fiat-Shamir channel order observed by `fiat_shamir`:
  [RFC-0014](../rfcs/RFC-0014-canonical-serialization-and-transcript.md).
- Pinned substrate revisions feeding `ToolVersions`:
  [RFC-0015](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md).
- Founding analysis (redaction and prover flow this document expands):
  [`docs/feasibility-study.md`](../feasibility-study.md) §8 and §10.3.

## Open questions

- OPEN QUESTION: Whether the Prometheus text-format endpoint is the supported
  long-term metrics export or whether an OpenTelemetry OTLP exporter is added.
  Owner: `area:prover` maintainer. Resolution: deferred to
  [RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md) follow-up
  at milestone `v1.0 — Fixed-Candidate Planning Proof`; V0 ships the in-process
  registry + `MetricsSnapshot` file, which is sufficient for the perf gates.
- OPEN QUESTION: The exact hash function family for `determinism_digest` (Blake2s
  is assumed here to match the Stwo channel backend). Owner: `area:core`
  maintainer. Resolution: bound by
  [RFC-0014](../rfcs/RFC-0014-canonical-serialization-and-transcript.md); this
  document follows whatever RFC-0014 locks for the public-input digest so the two
  hashes share a backend.

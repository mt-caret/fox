# Task: Bug hunting & test coverage for fox

You are in a dedicated git worktree on branch **`test-coverage`**. Read this whole file,
then start. Iterate until the "Definition of done" is met, committing to `test-coverage` as
you go. Work ONLY in this directory — do **not** touch `../fox` (the main checkout) or
`../fox-pytorch-backend` (a parallel job running in another worktree).

## Goal

**Find latent bugs in fox and raise test coverage.** Your two sharpest tools:
(a) **differential testing** — run the same program two ways and assert they agree; and
(b) **finite-difference gradient checks** — compare analytic gradients to a numerical
approximation. For each bug, minimize it to a small failing test, then either fix it (and
keep the test green) or, if a fix is risky / out of scope, leave a clear note and a test
that *documents* the failure rather than hiding it.

## What fox is (orientation)

fox is a JAX-inspired OCaml autodiff + XLA-JIT library. Read `README.md` and skim
`lib/core/`. Two backends exist: the eager `Tensor` backend (run via the `eval` handler in
`lib/core/handler.ml`) and the XLA JIT backend (`lib/jit/fox_jit.ml`). Autodiff transforms
(`jvp`, `vjp`, `grad`, `linearize`) live in `lib/core/handler.ml`; shape inference is
`Op.infer_shape` in `lib/core/op.ml`.

Key places:
- `test/test_eval_backends.ml` — the existing property-based differential test: generates
  random `Op` trees over random-shaped tensors, evaluates via the eager backend **and** via
  XLA JIT, and asserts `allclose`. **This is your main lever — read it, then broaden it**
  (more ops, more shapes, and gradients).
- `test/test_jvp.ml`, `test/test_staging.ml`, `test/test_partial_eval.ml`,
  `test/test_handler.ml`, `test/test_linear_regression.ml` — the other existing tests; match
  their style.

## Environment & build/test

This worktree shares the main checkout's `_opam` / `xla_extension`. Always use:
```bash
export XLA_EXTENSION_DIR=/home/ubuntu/dev/fox/xla_extension
opam exec --switch /home/ubuntu/dev/fox -- dune build @default @runtest
opam exec --switch /home/ubuntu/dev/fox -- dune runtest test/   # just the test suite
opam exec --switch /home/ubuntu/dev/fox -- dune fmt             # format; inspect first
```
`dune promote` to accept expect-test diffs you've eyeballed. The switch is
`ocaml-variants.5.2.0+ox`. `dune build` uses this worktree's own `_build`.

## Concrete starting points (hypotheses to verify — NOT conclusions)

Each is a lead to investigate, not a confirmed bug:
1. **Reverse mode may cover fewer ops than forward mode.** `jvp` (forward) handles
   sqrt/log/sigmoid; the reverse transpose `eval_expr_transposed` in `handler.ml` may not —
   if so, `grad' ~f:Value.sqrt` raises "Invalid var/val op combination". Test `grad` for
   *every* op `jvp` supports.
2. **Binary transpose, constant on either side / both-vars:** `grad (c / x)`, `grad (x / c)`,
   `grad (c - x)`, matmul with the variable on the left vs the right.
3. **Shape inference & broadcasting:** rank-0; a broadcast that both pads rank and stretches
   dims; `sum` over multiple or duplicate dims; `reshape` with `-1`; matmul rank /
   contraction-dim mismatches (do they error cleanly?).
4. **Bool tensors end-to-end:** `eq`/`gt`/`lt` and feeding the result downstream.
5. **Numeric edge cases:** div-by-zero, `log` of ≤ 0, `sqrt` of < 0 — does eager agree with
   XLA, and what *should* happen?

## High-value coverage to add

- **Finite-difference gradient check.** Write a helper that, for a scalar `f`, compares
  `grad' ~f ~x` against a central-difference numerical gradient
  `(f(x+h) - f(x-h)) / 2h` and asserts closeness. Run it over a battery of functions (each
  op, and compositions). This directly validates the vjp rules and tends to surface missing
  or wrong transpose cases. (Run `f` under the eager `eval` handler.)
- **Gradient differential test.** Extend the `test_eval_backends.ml` harness to gradients:
  compute `grad`/`vjp` of a random program both eager and via XLA and compare.

## Approach & definition of done

- For each finding, reduce to a minimal `let%expect_test` (or property test). Eager vs XLA
  disagreement, or a transform raising, is a bug.
- Fix bugs where the fix is clear and low-risk, keeping the test green. For risky / large
  fixes, leave a `(* CR: <describe> *)` note at the code and a test that documents the
  failure (e.g. `[@@expect.uncaught_exn]`, or an expect block showing the wrong output) so
  it's tracked, not buried.
- `dune build @default @runtest` and `dune fmt` green at the end.
- Keep a running **Findings** log at the bottom of this file: each bug (minimal repro, eager
  vs XLA, fixed or documented) and the coverage you added.

## Working agreements

- Follow the OCaml style in `~/.claude/CLAUDE.md` (open `Core`; Jane Street idioms; pretty-
  print the observed value in expect tests, not a derived bool; **build + runtest + fmt
  before declaring anything done**).
- Commit to the `test-coverage` branch in small steps as you make progress.

## Findings (append as you go)

### Bugs found by the multi-agent sweep

A parallel sweep (7 isolated agents, one per op/transform class) surfaced these, each
independently reproduced on the post-fix tree before acting:

- **BUG-4 (fixed): a failed XLA compilation poisoned all later ones.** `fox_jit.ml` used one
  process-global `lazy` `Xla.Builder`. An XLA builder records its first error and replays it
  from every subsequent `build`, so a single failing compile (e.g. a bool-const program)
  made every later — even unrelated, valid — `jit'` raise a stale error. Fixed by creating a
  fresh builder per compilation in `xla_callable`. Regression: `test_robustness.ml` "a failed
  compilation does not poison later compilations". (Side effect: HLO op-ids are now numbered
  per-compile, so the `fox_jit.ml` HLO expect tests were re-promoted — semantically identical.)
- **BUG-5 (fixed): `reshape` with a `-1` placeholder aborted the process under JIT.**
  `fox_jit.ml` forwarded the raw dims (incl. `-1`) to `Xla.Op.reshape`, hitting a C++ CHECK
  → SIGABRT (uncatchable), while eager resolves `-1` fine. Fixed by resolving the placeholder
  via `Op.infer_shape_exn` before calling XLA. Regression in `test_robustness.ml`.
- **BUG-6 (fixed): `Op.infer_shape` accepted sign-aliased duplicate sum dims.** The duplicate
  check ran on the raw dims, before negative-index normalization, so `sum ~dims:(`Just [0;-3])`
  on a rank-3 tensor (0 and -3 are the same axis) was accepted with a wrong shape (the
  positive duplicate `[0;0]` was already rejected). Eager then double-reduced and tripped the
  internal shape-consistency check; XLA raised "Duplicate reduction dimension". Fixed by
  normalizing before the dedup check in `op.ml`. Regression in `test_op.ml`.

### Limitations documented (CR note at the code + tracking test in `test_robustness.ml`)

- **Bool constants are unsupported under JIT.** `fox_jit.ml` hard-codes constant parameters
  to `F64` and the literal conversion is float-only, so a jitted program closing over a bool
  constant raises (eager handles it). Risky to fix (touches the float-only literal/buffer
  path); CR note at `fox_jit.ml`.
- **`grad`/`vjp` raises when an output does not depend on the input.** The traced tangent
  program keeps only input-dependent outputs, so an all-constant output yields an empty
  `return_vals` (and `Expr` requires a `Nonempty_list`); a tuple with a constant component
  hits a cotangent/return-val length mismatch. Forward mode already yields a zero tangent, so
  the correct answer is a zero gradient. The `Nonempty_list` return type makes a proper fix
  invasive; CR note at `handler.ml`.
- The sweep's matmul agent flagged the matmul-with-vector gradient as still broken, but that
  was a stale-worktree false positive — it is fixed (see BUG-2) and verified across eager and
  finite differences.

### Bugs found & fixed (initial pass)

- **BUG-1 (fixed): `Op.infer_shape` Sum reduction-dim bounds check was a no-op.**
  In `op.ml`, the in-bounds test was `dim < dims_length || dims_length + dim >= 0`. With
  `||`, *every* integer passes (a positive out-of-range dim satisfies the second clause; a
  negative out-of-range dim satisfies the first), so out-of-range reduction dims were
  accepted and `infer_shape` returned a wrong (often unchanged) output shape instead of an
  error. Minimal repro: `Op.infer_shape (Sum {value={dims=[3];Float}; dims=`Just [5];
  keep_dims=false})` returned `Ok {dims=[3]}` (should be `Error`); same for `dim=-5`.
  Fix: `||` → `&&`. Regression + edge cases in `test/test_op.ml`.

- **BUG-2 (fixed): reverse-mode gradient of `matmul matrix vector` raised.**
  Forward `x[n,m] @ v[m] = y[n]` works on both backends, but `grad`/`vjp` *with the matrix
  as the differentiated variable* raised `"infer_dims: Transpose: unsupported transpose
  dimensions"`: the transpose rule for `Matmul (Var, Value v)` did `matmul cotangent
  (transpose v)`, and `transpose` only supports rank 2, so transposing the rank-1 `v`
  failed. Minimal repro: `grad' ~f:(fun x -> Value.sum (Value.matmul x vec)) ~x:matrix`.
  Fix (`handler.ml`): when `v` is a vector, form the outer product `y_ct[n,1] @ v[1,m]`
  instead of transposing. Verified by finite differences in `test/test_gradcheck.ml`
  ("matmul x[2,3]@vec[3] (matrix var)").

### Hypotheses investigated and refuted (eager and XLA agree / gradients correct)

- Negative `sum` dims agree between eager and XLA (the XLA `reduce_sum` binding normalizes
  negative axes just as the eager backend does).
- `grad` of every op `jvp` supports — incl. `sqrt`/`log`/`sigmoid` and `c/x`, `x/c`,
  `c-x` — is correct (the jvp rules emit only linear primitives, all of which the transpose
  handles). All finite-difference checks pass.
- Higher-order reverse-mode (`grad (grad ...)`) is correct for cube, `1/x`, and `sum(x^3)`.
- Numeric edge cases (`1/0`, `log 0`, `log (-1)`, `sqrt (-1)`, `0/0`) produce identical
  IEEE inf/nan on both backends.
- Bool tensors end-to-end (`eq`/`gt`/`lt`, chained bool `eq`, `reshape`/`broadcast` of a
  bool) agree between eager and XLA.

### Coverage added

- `test/test_op.ml` — `Op.infer_shape` unit tests: sum bounds/dups/keep_dims/bool,
  broadcast rank-padding + dim-stretching + error cases, reshape `-1` inference, matmul
  rank/contraction/type checks, transpose rank, elementwise rank-0 / dim-agreement / bool
  typing.
- `test/test_gradcheck.ml` — central-difference gradient-check harness + battery over every
  op and several compositions (validates the vjp rules independent of jvp).
- `test/test_eval_backends.ml` — broadened differential testing:
  - Multi-op random programs (`op_nums` 1..4) instead of only single ops, for both the
    forward (eval vs XLA) and the gradient (grad eval vs XLA) differential tests.
  - `Sum` now generates `keep_dims=false` as well as `true` (was hardcoded to `true`).
  - Periodic `Gc.full_major ()` so the XLA thread leak doesn't hit `pthread_create` EAGAIN
    over the larger trial count.

### Test-harness issues the broadening surfaced (fixed in the test, not the library)

- **Generator built invalid `sum`/`matmul` ops once bool intermediates existed.** With
  multi-op programs, `eq`/`gt`/`lt` produce bool shapes; the `Sum`/`Matmul` generator arms
  picked operand shapes without checking `Op.infer_shape`, so they could build a float-only
  op over a bool operand and `infer_shape_exn` raised during generation. Fixed by filtering
  candidate shapes through `Op.infer_shape` (the `Unary`/`Binary`/`Broadcast`/`Reshape` arms
  already did this).
- **Differential equality used an absolute tolerance.** `Tensor.allclose` compares via
  `Float.robustly_compare` (absolute 1e-7). The eager and XLA backends accumulate reductions
  in different orders, so large-magnitude results like `sum (exp x)` (≈1e42) legitimately
  differ by ~1 ULP (~1e26 absolute), which the absolute tolerance flags as a mismatch.
  Confirmed it is purely summation order: `transpose` and `exp` agree bit-for-bit, only the
  final `sum` differs in the last ULP. Fixed by comparing with a relative tolerance in the
  differential tests (`Tensor.allclose` left unchanged, as bool/nan/inf handling relies on
  it). This is not a fox bug — it is expected floating-point behavior.

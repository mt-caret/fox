# Task: PyTorch (libtorch) backend for fox

You are in a dedicated git worktree on branch **`pytorch-backend`**. Read this whole file,
then start. Iterate until the "Definition of done" is met, committing to `pytorch-backend`
as you go. Work ONLY in this directory — do **not** touch `../fox` (the main checkout) or
`../fox-test-coverage` (a parallel job running in another worktree).

## Goal

Add a **PyTorch / libtorch backend**: an interpreter that evaluates fox tensor programs on
torch tensors, primarily so results can be **differential-tested** against the existing
eager `Tensor` backend (and the XLA JIT backend). Mirror the eager backend op-for-op; an
`Expr.t` → torch *compiler* is a stretch goal, not the target.

## What fox is (orientation)

fox is a JAX-inspired OCaml autodiff + XLA-JIT library. Read `README.md` and skim
`lib/core/`. The pieces you need:

- **Backend interface:** `lib/core/operators_intf.ml` — `Operators_intf.S` is the set of
  tensor primitives a backend provides (`neg`/`sin`/…/`matmul`/`sum`/`broadcast`/`reshape`).
- **Eager backend:** `lib/core/tensor.ml` (`Tensor`) implements `Operators_intf.S` over
  bigarray-backed tensors. Element types are `Float` and `Bool` (`lib/core/type.ml`);
  comparisons (`eq`/`gt`/`lt`) produce `Bool`.
- **The effect + handler:** programs perform `Fox_effect.Op` (`lib/core/fox_effect.ml`); a
  handler interprets each op via a backend. The eager handler is `eval` in
  `lib/core/handler.ml`:
  ```ocaml
  let eval ~f =
    Fox_effect.handle ~f ~handle:(fun op ->
      Op.map op ~f:Value.to_tensor_exn
      |> Op.eval (module Tensor : Operators_intf.S with type t = Tensor.t)
      |> Value.of_tensor)
  ```
  Your backend gets a `Pytorch.handle ~f` that does the same thing through torch.
- **`Op.eval`** (`lib/core/op.ml`): `(module Operators_intf.S with type t = 'a) -> 'a Op.t
  -> 'a` evaluates one op-node with a backend's primitives. The primitives it needs are
  exactly the `Op.t` constructors: neg, sin, cos, sqrt, exp, log, sigmoid, add, sub, mul,
  div, eq, gt, lt, matmul, transpose, sum, broadcast, reshape.
- **fox ↔ external boundary:** `lib/jit/fox_jit.ml` converts fox `Tensor.t` ↔ XLA via
  bigarrays (`Tensor.Private.to_bigarray` / `Tensor.Private.of_float_bigarray`). Your torch
  backend converts the same way (`Torch.Tensor.of_bigarray` / `to_bigarray`).

## Environment & build

This worktree has no `_opam` or `xla_extension` of its own — it shares the main checkout's.
Always build/test with:
```bash
export XLA_EXTENSION_DIR=/home/ubuntu/dev/fox/xla_extension
opam exec --switch /home/ubuntu/dev/fox -- dune build @default @runtest
opam exec --switch /home/ubuntu/dev/fox -- dune fmt   # format; inspect the diff first
```
The switch is `ocaml-variants.5.2.0+ox` (OxCaml). `dune build` uses this worktree's own
`_build`, so it won't collide with the other worktrees.

## The task

1. **Get ocaml-torch building under the ox switch FIRST — this is the main risk.** Add
   `torch` (LaurentMazare/ocaml-torch) and install it into the shared switch
   (`opam install --switch /home/ubuntu/dev/fox torch` — pulls a CPU libtorch). **If the
   `5.2.0+ox` variant blocks ocaml-torch (quite possible), STOP and write up the
   incompatibility and the options** — pin an older torch, a dedicated non-ox switch just
   for the backend library, raw libtorch FFI, etc. — at the bottom of this file before
   sinking hours into a workaround. Note: installing into the shared switch also affects the
   main checkout; that's acceptable (it's a real new project dep), just be aware.
2. **Implement the backend** — a module providing `Operators_intf.S with type t =
   Torch.Tensor.t` (at minimum the `Op.eval` primitive subset). See how `Tensor` builds the
   full interface (it likely uses `Op.Make_operators` to derive derived ops from
   primitives) and follow the same shape.
3. **Boundary conversion** fox `Tensor.t` ↔ `Torch.Tensor.t` via bigarrays. Watch dtype:
   fox is **float64**, torch defaults to **float32** — set torch to f64 (or compare with a
   tolerance). `Bool` tensors: torch comparisons yield bool/uint8 — handle both element
   types.
4. **Handler** `Pytorch.handle ~f` analogous to `eval`.
5. **Differential test** — run the same random program through `eval` and `Pytorch.handle`
   and assert `allclose`. `test/test_eval_backends.ml` already does eager-vs-XLA
   differential testing with quickcheck generators; read it and add an eager-vs-torch
   comparison the same way.

## Definition of done

- `dune build @default @runtest` green (the shared-switch command above).
- A differential test exercises the torch backend against the eager backend and passes.
- The new dependency is in the opam metadata / lockfile (or, if it can't be locked under
  ox, the workaround is documented here).
- `dune fmt` clean.

## Working agreements

- Follow the OCaml style in `~/.claude/CLAUDE.md` (open `Core`; `.mli` for every non-test
  `.ml` except the library-name module; Jane Street idioms; **build + runtest + fmt before
  declaring anything done**).
- Commit to the `pytorch-backend` branch in small steps as you make progress.

## Notes / decisions / blockers (append as you go)

### Switch & dependency setup
- Per the invocation override (use a dedicated switch so the concurrent `fox-test-coverage`
  job's switch isn't disturbed), all torch work is in a **dedicated opam switch
  `fox-pytorch`** (`ocaml-variants.5.2.0+ox`, repos `ox,default`) — the shared
  `/home/ubuntu/dev/fox` switch is left untouched. Build/test with:
  ```bash
  export XLA_EXTENSION_DIR=/home/ubuntu/dev/fox/xla_extension
  opam exec --switch fox-pytorch -- dune build @default @runtest
  opam exec --switch fox-pytorch -- dune fmt
  ```
- `torch` added to `dune-project`/`fox.opam` depends.

### ocaml-torch under OxCaml — NOT blocked by ox
- The `ox` opam repo ships `torch v0.18~preview.130.91+190` (Jane Street's ocaml-torch fork)
  with explicit oxcaml support (`conflicts: oxcaml-compiler {< "5.2.0minus31"}`; we have
  `5.2.0minus31`). All JS deps already match `v0.18~preview.130.91+190`. So OxCaml itself is
  fine — the real obstacles were CUDA/libtorch, below.

### libtorch: version + CPU-only patch (the actual blockers)
- `torch` declares `libtorch` a depopt; the opam `libtorch` package maxes at 2.2.1, so
  libtorch is supplied manually and placed inside the switch at
  `$prefix/lib/libtorch` (sandbox-readable; found by torch's `discover.ml` via
  `OPAM_SWITCH_PREFIX`, rpath baked at build time).
- **Version:** the fork commit (dated 2026-04-06) calls the *2.4+* unified autocast API
  `at::autocast::is_autocast_enabled(DeviceType)`, absent from 2.3.1. The generated ATen
  bindings must match the libtorch version, so libtorch **2.7.1+cpu** (newest available on
  the CDN; exports exactly that autocast symbol) is used — NOT the 2.3.x implied by torch's
  (stale) opam conflict bound.
- **CPU patch:** the fork's `torch_api.cpp`/`.h` (and the `wrapper_refcounted` twins)
  unconditionally include CUDA-only headers (`model_container_runner_cuda.h`,
  `cuda/memory_snapshot.h`, `<cuda_runtime.h>`), which fail without a CUDA toolkit, and
  reference symbols only in `libtorch_cuda` (absent from CPU libtorch). Fix: pin torch to a
  patched source at `/home/ubuntu/dev/torch-patched` that drops those includes and stubs the
  AOTI-CUDA runner + CUDA memory-snapshot entry points (`caml_failwith`), typedef'ing the
  CUDA runner handle to `void *`. `torch::cuda::{device_count,is_available,cudnn_is_available}`
  and the autocast calls remain (present in libtorch_cpu 2.7.1). Pinning torch also pulls
  `ctypes` 0.24.0+ox -> 0.23.0+ox (upstream torch.opam bound); harmless in the isolated switch.

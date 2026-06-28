# fox_torch — PyTorch (libtorch) backend

`Pytorch` implements `Operators_intf.S` over `Torch.Tensor.t` and provides
`Pytorch.handle ~f`, a handler analogous to `Handler.eval` that interprets each fox `Op`
through libtorch. Its purpose is differential testing against the eager `Tensor` backend
(see `test/test_eval_backends.ml`, "eval expr vs torch").

fox is float64, so floats map to torch `Double` tensors; fox `Bool` tensors (stored as 0/1
bytes) map to torch `Uint8`, and torch comparison results (`Bool` kind) are normalised back
to fox `Bool`. Conversion goes through bigarrays (`Tensor.Private` ⟷ `Torch.Tensor`).

## Building (CPU-only)

This depends on `ocaml-torch` (Jane Street's `torch`, the only ocaml-torch that builds under
the OxCaml compiler) plus a libtorch the project supplies manually — the opam `libtorch`
package is stale (≤ 2.2.1) and does not satisfy `torch`'s constraints. Two adjustments are
required for a CPU-only host without a CUDA toolkit:

1. **libtorch 2.7.1+cpu** (cxx11-abi). The `torch` source calls the 2.4+ unified autocast
   API (`at::autocast::is_autocast_enabled(DeviceType)`) and its generated ATen bindings
   must match the libtorch version, so 2.3.x (implied by `torch`'s stale opam conflict) does
   not work. Unpack it inside the switch at `$OPAM_SWITCH_PREFIX/lib/libtorch` (where
   `discover.ml` finds it) or point `LIBTORCH` at it.

2. **A CPU patch to `torch`.** Upstream `torch_api.cpp`/`.h` (and the `wrapper_refcounted`
   twins) unconditionally include CUDA-only headers and reference symbols that live only in
   `libtorch_cuda`, and `discover.ml` both hardcodes `-ltorch_cuda` and only forces
   `--no-as-needed` when CUDA is present. The patch drops the CUDA includes, stubs the
   AOTI-CUDA runner + CUDA memory-snapshot entry points, drops `torch_cuda` from the required
   libs, always passes `--no-as-needed`, and additionally emits the libtorch link flags as
   `c_library_flags` so they propagate to executables that link this backend (dune does not
   replay `(flags)` `-ccopt` link flags downstream).

### Reproducing the build (overlay + lockfile)

The patch is captured reproducibly, not as an ad-hoc local pin:

- [`../../opam-overlay/`](../../opam-overlay) is a local opam-repository whose only package is
  `torch.v0.18~preview.130.91+190` — identical to the `ox` repo's, plus
  `patches: ["torch-cpu.patch"]`. It fetches the upstream tarball (by checksum) and applies
  `files/torch-cpu.patch`, so no patched sources are vendored.
- [`../../fox.opam.locked`](../../fox.opam.locked) pins every dependency (including `torch`
  and the `ctypes` 0.23.x it pulls) to exact versions.

To reproduce in a fresh switch (the overlay must out-rank `ox` so its patched `torch` wins
for the shared version), then supply libtorch as in (1):

```bash
opam switch create . ocaml-variants.5.2.0+ox \
  --repos fox-overlay=./opam-overlay,ox=git+https://github.com/oxcaml/opam-repository.git,default=https://opam.ocaml.org
# unpack libtorch 2.7.1+cpu into $(opam var prefix)/lib/libtorch, then:
XLA_EXTENSION_DIR=/path/to/xla_extension opam install . --deps-only --locked
```

Or, in an existing OxCaml switch: `opam repo add fox-overlay ./opam-overlay --rank 1` then
`opam install . --deps-only --locked`.

See the repo-root `PLAN.md` "Notes / decisions / blockers" for the full rationale.

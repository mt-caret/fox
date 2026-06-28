# Fox - Functional OCaml + XLA

Fox is an OCaml library for automatic differentiation and numerical computing,
inspired heavily by Jax. It provides a flexible framework for automatically
differentiating and XLA-jit-compiling tensor-based computations.

Please note that this library is still under active development and lacks
many features.

## Available features

- **Automatic Differentiation**: Support for forward-mode and reverse-mode automatic differentiation
- **Higher-Order Derivatives**: Compute nth-order derivatives of functions
- **JIT Compilation**: Just-in-time compilation to XLA
- **Tree-based Value Representation**: Efficient handling of complex data structures through tree-based representations

<!-- TODO: add usage example -->

## Building

Fox targets [OxCaml](https://oxcaml.org/): it uses OxCaml's effect-handler
syntax and the modal Jane Street libraries, and builds against a local opam
switch on `ocaml-variants.5.2.0+ox` with a committed lockfile
(`fox.opam.locked`). It depends on a pinned
[`xla`](https://github.com/mt-caret/ocaml-xla), which binds a prebuilt XLA
extension blob.

### One-time setup

Download the XLA extension (elixir-nx/xla v0.4.4) and point `XLA_EXTENSION_DIR`
at it. The C++ stubs need its headers at build time; the runtime `rpath` is
baked in, so no `LD_LIBRARY_PATH` is needed once built:

```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cpu.tar.gz
tar -xzf xla_extension-x86_64-linux-gnu-cpu.tar.gz   # -> ./xla_extension
export XLA_EXTENSION_DIR=$PWD/xla_extension
```

(On platforms other than linux x86_64, download the matching archive instead.)

Create the switch from the lockfile:

```bash
opam switch create . 5.2.0+ox \
  --repos ox=git+https://github.com/oxcaml/opam-repository.git,default \
  --locked
```

### Build and test

```bash
XLA_EXTENSION_DIR=$PWD/xla_extension dune build @default @runtest
```

### Regenerating the lockfile

After changing dependencies, refresh `fox.opam.locked`:

```bash
XLA_EXTENSION_DIR=$PWD/xla_extension opam install . --deps-only --with-test
opam lock .
```

## TODO

- [x] Support non-singleton tensors in vjp
- [x] Better shape and type story
- [x] Basic tensor operations for simple neural network example
  - [x] 2d matmuls
  - [x] sum
  - [x] random other things
- [x] Print XLA HLO module
- [ ] Add JIT caching story
- [ ] Pytorch backend
- [x] Testing framework for diffing op backends (XLA, Pytorch, OCaml)
  - [x] Quickcheck generators
  - [x] Diff XLA backend and OCaml
  - [ ] Diff OCaml/XLA vjp and Pytorch
- [ ] Support various types (ints, bools, etc.)
- [ ] Shape inference?
- [ ] Mixed-precision support
- [ ] Custom operator support
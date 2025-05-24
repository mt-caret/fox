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

```bash
DYLD_LIBRARY_PATH=/absolute-path-to-ocaml-xla-extension-lib dune build @default @runtest -w
```

## TODO

- [x] Support non-singleton tensors in vjp
- [x] Better shape and type story
- [ ] Basic tensor operations for simple neural network example
  - [x] 2d matmuls
  - [x] sum
  - [ ] TODO: add more here
- [ ] Print XLA HLO module
- [ ] Add JIT caching story
- [ ] Pytorch backend
- [ ] Testing framework for diffing op backends (XLA, Pytorch, OCaml)
  - [ ] Quickcheck generators
  - [ ] Diff XLA backend and OCaml
  - [ ] Diff OCaml/XLA vjp and Pytorch
- [ ] Shape inference?
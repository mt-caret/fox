# Ox

Ox is an OCaml library for automatic differentiation and numerical computing,
based on Jax. It provides a flexible and efficient framework for computing
derivatives and performing numerical operations on tensors.

Please note that this library is still under active development and lacks
many features.

## Available features

- **Automatic Differentiation**: Support for forward-mode and reverse-mode automatic differentiation
- **Higher-Order Derivatives**: Compute nth-order derivatives of functions
- **JIT Compilation**: Just-in-time compilation to XLA
- **Tree-based Value Representation**: Efficient handling of complex data structures through tree-based representations

## Usage Example

```ocaml
open Ox

(* Define a function *)
let f x = Value.O.(x * (x + Value.of_float 3.))

(* Compute the function value *)
let result = Eval.handle ~f:(fun () -> f (Value.of_float 2.))

(* Compute the derivative *)
let derivative = derivative ~f ~x:(Value.of_float 2.)
```

## Building

```bash
DYLD_LIBRARY_PATH=/absolute-path-to-ocaml-xla-extension-lib dune build @default @runtest -w
```

## TODO

- [x] Support non-singleton tensors in vjp
- [x] Better shape and type story
- [ ] Basic tensor operations for simple neural network example
  - [x] 2d matmuls
  - [ ] TODO: add more here
- [ ] Add JIT caching story
- [ ] Pytorch backend
- [ ] Testing framework for diffing op backends (XLA, Pytorch, OCaml)
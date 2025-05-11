open! Core

type t = Value0.t = T : 'a * 'a Type_equal.Id.t -> t [@@deriving sexp_of]

val tree_def : Value_tree.Def.t
val of_tensor : Tensor.t -> t
val to_tensor_exn : t -> Tensor.t
val of_float : float -> t
val to_float_exn : t -> float

include Treeable.S with type t := t
include Operators_intf.S with type t := t

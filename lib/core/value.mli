open! Core

type t = Value0.t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; dims : int array
      }
      -> t
[@@deriving sexp_of]

val dims : t -> int array
val tree_def : dims:int array -> Value_tree.Def.t
val of_tensor : Tensor.t -> t
val to_tensor_exn : t -> Tensor.t
val of_float : float -> t
val to_float_exn : t -> float

include Treeable.S with type t := t
include Operators_intf.S with type t := t

module Tuple2 : sig
  include Treeable.S with type t = t * t

  val tree_def : dims1:int array -> dims2:int array -> Value_tree.Def.t
end

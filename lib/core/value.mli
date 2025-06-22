open! Core
include module type of Value0

val tree_def : dims:int array -> Value_tree.Def.t
val of_tensor : 'a Tensor.t -> 'a t
val to_tensor_exn : 'a t -> 'a Tensor.t
val of_float : float -> float t
val to_float_exn : float t -> float

include Operators_intf.S with type 'a t := 'a t

module Packed : sig
  include module type of Value0.Packed
  include Treeable.S with type t := t
end

module Tuple2 : sig
  include Treeable.S with type t = Packed.t * Packed.t

  val tree_def : dims1:int array -> dims2:int array -> Value_tree.Def.t
end

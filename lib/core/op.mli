open! Core

type 'value t =
  | Neg of 'value
  | Sin of 'value
  | Cos of 'value
  | Sqrt of 'value
  | Add of 'value * 'value
  | Sub of 'value * 'value
  | Mul of 'value * 'value
  | Div of 'value * 'value
  | Matmul of 'value * 'value
  | Transpose of 'value
  | Sum of
      { value : 'value
      ; dims : [ `Just of int array | `All ]
      ; keep_dims : bool
      }
  | Broadcast of
      { value : 'value
      ; dims : int array
      }
[@@deriving sexp_of]

val map : 'a t -> f:('a -> 'b) -> 'b t
val eval : (module Operators_intf.S with type t = 'a) -> 'a t -> 'a
val to_string : 'a t -> f:('a -> string) -> string
val infer_dims : int array t -> int array

module Make_operators (M : sig
    type value [@@deriving sexp_of]

    val of_float : float -> value
    val eval : value t -> value
    val dims : value -> int array
  end) : Operators_intf.S with type t := M.value

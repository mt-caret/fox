open! Core

type 'value t =
  | Add of 'value * 'value
  | Sub of 'value * 'value
  | Mul of 'value * 'value
  | Neg of 'value
  | Sin of 'value
  | Cos of 'value
  | Matmul of 'value * 'value
  | Transpose of 'value
  | Sum of
      { value : 'value
      ; dims : int array
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

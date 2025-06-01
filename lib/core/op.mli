open! Core

module Unary : sig
  type t =
    | Neg
    | Sin
    | Cos
    | Sqrt
  [@@deriving sexp, enumerate]
end

module Binary : sig
  type t =
    | Add
    | Sub
    | Mul
    | Div
  [@@deriving sexp, enumerate]
end

type 'value t =
  | Unary of Unary.t * 'value
  | Binary of Binary.t * 'value * 'value
  | Matmul of 'value * 'value
  | Transpose of 'value
  | Sum of
      { value : 'value
      ; dims : [ `Just of int Nonempty_list.t | `All ]
      ; keep_dims : bool
      }
  | Broadcast of
      { value : 'value
      ; dims : int array
      }
[@@deriving sexp_of]

val map : 'a t -> f:('a -> 'b) -> 'b t
val to_list : 'a t -> 'a list
val eval : (module Operators_intf.S with type t = 'a) -> 'a t -> 'a
val to_string : 'a t -> f:('a -> string) -> string
val infer_dims : int array t -> int array Or_error.t
val infer_dims_exn : int array t -> int array

module Make_operators (M : sig
    type value [@@deriving sexp_of]

    val of_float : float -> value
    val eval : value t -> value
    val dims : value -> int array
  end) : Operators_intf.S with type t := M.value

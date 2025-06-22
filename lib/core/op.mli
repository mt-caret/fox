open! Core

module Unary : sig
  type t =
    | Neg
    | Sin
    | Cos
    | Sqrt
    | Exp
    | Log
    | Sigmoid
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

type ('type_, 'f) t =
  | Unary : Unary.t * (float -> 'f) Higher_kinded.t -> (float, 'f) t
  | Binary :
      Binary.t * (float -> 'f) Higher_kinded.t * (float -> 'f) Higher_kinded.t
      -> (float, 'f) t
  | Matmul :
      (float -> 'f) Higher_kinded.t * (float -> 'f) Higher_kinded.t
      -> (float, 'f) t
  | Transpose : ('type_ -> 'f) Higher_kinded.t -> ('type_, 'f) t
  | Sum :
      { value : (float -> 'f) Higher_kinded.t
      ; dims : [ `Just of int Nonempty_list.t | `All ]
      ; keep_dims : bool
      }
      -> (float, 'f) t
  | Broadcast :
      { value : ('type_ -> 'f) Higher_kinded.t
      ; dims : int array
      }
      -> ('type_, 'f) t
  | Reshape :
      { value : ('type_ -> 'f) Higher_kinded.t
      ; dims : int array
      }
      -> ('type_, 'f) t

module Non_higher_kinded : sig
  type ('a, 'f) higher_kinded := ('a, 'f) t

  type 'a t =
    | Unary of Unary.t * 'a
    | Binary of Binary.t * 'a * 'a
    | Matmul of 'a * 'a
    | Transpose of 'a
    | Sum of
        { value : 'a
        ; dims : [ `Just of int Nonempty_list.t | `All ]
        ; keep_dims : bool
        }
    | Broadcast of
        { value : 'a
        ; dims : int array
        }
    | Reshape of
        { value : 'a
        ; dims : int array
        }
  [@@deriving variants, sexp]

  type ('f, 'b) f = { f : 'a. ('a -> 'f) Higher_kinded.t -> 'b }

  val of_higher_kinded : ('a, 'f) higher_kinded -> f:('f, 'b) f -> 'b t
  val to_list : 'a t -> 'a list
end

module Packed : sig
  type ('a, 'f) typed := ('a, 'f) t
  type 'f t = T : ('a, 'f) typed -> 'f t

  val sexp_of_t : f:('f, Sexp.t) Non_higher_kinded.f -> 'f t -> Sexp.t
end

type ('f1, 'f2) f = { f : 'a. ('a -> 'f1) Higher_kinded.t -> ('a -> 'f2) Higher_kinded.t }

val map : ('a, 'f1) t -> f:('f1, 'f2) f -> ('a, 'f2) t
val to_string : ('a, 'f) t -> f:(('a -> 'f) Higher_kinded.t -> string) -> string

module Dims : sig
  type _ t = int array [@@deriving sexp_of]

  include Higher_kinded.S with type 'a t := 'a t
end

val infer_dims : (_, Dims.higher_kinded) t -> int array Or_error.t
val infer_dims_exn : (_, Dims.higher_kinded) t -> int array

module Make_operators (M : sig
    type ('a, 'f) op := ('a, 'f) t
    type 'a t [@@deriving sexp_of]

    include Higher_kinded.S with type 'a t := 'a t

    val of_float : float -> float t
    val eval : ('a, higher_kinded) op -> 'a t
    val dims : 'a t -> int array
  end) : Operators_intf.S with type 'a t := 'a M.t

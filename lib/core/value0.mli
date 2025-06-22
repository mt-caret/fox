open! Core

type 'type_ t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; dims : int array
      ; type_ : 'type_ Type.t
      }
      -> 'type_ t
[@@deriving sexp_of]

include Higher_kinded.S with type 'a t := 'a t

module Packed : sig
  type 'a typed := 'a t
  type t = T : 'a typed -> t [@@deriving sexp_of]

  val dims : t -> int array
end

val dims : _ t -> int array
val type_ : 'a t -> 'a Type.t

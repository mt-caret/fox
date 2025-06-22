open! Core

type _ t =
  | Bool : bool t
  | Float : float t
[@@deriving sexp_of]

val type_equal_id : 'a t -> 'a Type_equal.Id.t

module Packed : sig
  type 'a typed := 'a t
  type t = T : 'a typed -> t

  val all : t list
end

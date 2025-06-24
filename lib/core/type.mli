open! Core

type _ t =
  | Float : float t
  | Bool : bool t
[@@deriving sexp_of]

val type_equal_id : 'a t -> 'a Type_equal.Id.t

module Packed : sig
  type 'a typed := 'a t
  type t = T : 'a typed -> t [@@deriving enumerate, equal, compare, sexp]
end

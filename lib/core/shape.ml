open! Core

type t =
  { dims : int iarray
  ; type_ : Type.Packed.t
  }
[@@deriving equal, compare, sexp, fields ~getters]

include functor Comparable.Make_plain

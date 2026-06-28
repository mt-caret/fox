open! Core

type t =
  { dims : int iarray
  ; type_ : Type.Packed.t
  }
[@@deriving equal, compare, hash, sexp, fields ~getters]

include functor Comparable.Make_plain

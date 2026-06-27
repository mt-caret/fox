open! Core

type t =
  { dims : int iarray
  ; type_ : Type.Packed.t
  }
[@@deriving equal, compare, sexp, fields ~getters]

include Comparable.S_plain with type t := t

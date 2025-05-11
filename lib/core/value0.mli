open! Core

type t = T : 'a * 'a Type_equal.Id.t -> t [@@deriving sexp_of]

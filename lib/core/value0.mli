open! Core

type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; dims : int array
      }
      -> t
[@@deriving sexp_of]

val dims : t -> int array

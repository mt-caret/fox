open! Core

type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; dims : int array option
      }
      -> t
[@@deriving sexp_of]

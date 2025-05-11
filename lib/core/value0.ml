open! Core

type t = T : 'a * 'a Type_equal.Id.t -> t

let sexp_of_t (T (x, id)) =
  let x = Type_equal.Id.to_sexp id x in
  [%message (Type_equal.Id.name id) ~_:(x : Sexp.t)]
;;

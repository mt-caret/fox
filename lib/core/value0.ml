open! Core

(* TODO: switch to use iarrays once they land: 
  https://github.com/ocaml/ocaml/pull/13097 *)
type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; dims : int array
      }
      -> t

let dims (T { value = _; type_id = _; dims }) = dims

let sexp_of_t (T { value; type_id; dims }) =
  let x = Type_equal.Id.to_sexp type_id value in
  match dims with
  | [||] -> [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t)]
  | dims -> [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t) (dims : int array)]
;;

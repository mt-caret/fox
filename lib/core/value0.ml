open! Core

(* TODO: switch to use iarrays once they land: 
  https://github.com/ocaml/ocaml/pull/13097 *)
type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; shape : Shape.t
      }
      -> t

let dims (T { value = _; type_id = _; shape = { dims; type_ = _ } }) = dims
let type_ (T { value = _; type_id = _; shape = { dims = _; type_ } }) = type_
let shape (T { value = _; type_id = _; shape }) = shape

let coerce
      (type a)
      (T { value; type_id = type_id'; shape = _ })
      ~(type_id : a Type_equal.Id.t)
  : a option
  =
  match Type_equal.Id.same_witness type_id type_id' with
  | Some T -> Some value
  | None -> None
;;

let coerce_exn
      (type a)
      (T { value; type_id = type_id'; shape = _ })
      ~(type_id : a Type_equal.Id.t)
  : a
  =
  let T = Type_equal.Id.same_witness_exn type_id type_id' in
  value
;;

let sexp_of_t (T { value; type_id; shape = { dims; type_ } }) =
  let x = Type_equal.Id.to_sexp type_id value in
  match dims with
  | [||] ->
    [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t) ~_:(type_ : Type.Packed.t)]
  | dims ->
    [%message
      (Type_equal.Id.name type_id)
        ~_:(x : Sexp.t)
        (dims : int array)
        ~(type_ : Type.Packed.t)]
;;

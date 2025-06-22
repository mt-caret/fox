open! Core

module T = struct
  (* TODO: switch to use iarrays once they land: 
  https://github.com/ocaml/ocaml/pull/13097 *)
  type 'type_ t =
    | T :
        { value : 'a
        ; type_id : 'a Type_equal.Id.t
        ; dims : int array
        ; type_ : 'type_ Type.t
        }
        -> 'type_ t
end

include T
include Higher_kinded.Make (T)

module Packed = struct
  type 'a typed = 'a t
  type t = T : 'a typed -> t

  let sexp_of_t (T (T { value; type_id; dims; type_ = _ })) =
    let x = Type_equal.Id.to_sexp type_id value in
    match dims with
    | [||] -> [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t)]
    | dims -> [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t) (dims : int array)]
  ;;

  let dims (T (T { value = _; type_id = _; dims; type_ = _ })) = dims
end

let dims (T { value = _; type_id = _; dims; type_ = _ }) = dims
let type_ (T { value = _; type_id = _; dims = _; type_ }) = type_
let sexp_of_t _sexp_of_a t = [%sexp_of: Packed.t] (T t)

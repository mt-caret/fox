open! Core

(* TODO: switch to use iarrays once they land: https://github.com/ocaml/ocaml/pull/13097 *)
type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; shape : Shape.t
      ; id : Id.t
      }
      -> t

let create ~value ~type_id ~shape = T { value; type_id; shape; id = Id.create () }
let dims (T { value = _; type_id = _; shape = { dims; type_ = _ }; id = _ }) = dims
let type_ (T { value = _; type_id = _; shape = { dims = _; type_ }; id = _ }) = type_
let shape (T { value = _; type_id = _; shape; id = _ }) = shape

let coerce
  (type a)
  (T { value; type_id = type_id'; shape = _; id = _ })
  ~(type_id : a Type_equal.Id.t)
  : a option
  =
  match Type_equal.Id.same_witness type_id type_id' with
  | Some T -> Some value
  | None -> None
;;

let coerce_exn
  (type a)
  (T { value; type_id = type_id'; shape = _; id = _ })
  ~(type_id : a Type_equal.Id.t)
  : a
  =
  let T = Type_equal.Id.same_witness_exn type_id type_id' in
  value
;;

let sexp_of_t (T { value; type_id; shape = { dims; type_ }; id = _ }) =
  let x = Type_equal.Id.to_sexp type_id value in
  match dims with
  | [||] ->
    [%message (Type_equal.Id.name type_id) ~_:(x : Sexp.t) ~_:(type_ : Type.Packed.t)]
  | dims ->
    [%message
      (Type_equal.Id.name type_id)
        ~_:(x : Sexp.t)
        (dims : int array)
        ~type_:(type_ : Type.Packed.t)]
;;

(* Values carry a unique [id] so that an identical constant reused across a traced
   computation can be deduplicated by identity. *)
module On_id = struct
  module T = struct
    type nonrec t = t [@@deriving sexp_of]

    let compare t1 t2 =
      Comparable.lift [%compare: Id.t] t1 t2 ~f:(fun (T { id; _ }) -> id)
    ;;
  end

  include T
  include Comparable.Make_plain (T)
end

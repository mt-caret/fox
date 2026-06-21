open! Core

type t = private
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; shape : Shape.t
      ; id : Id.t
      }
      -> t
[@@deriving sexp_of]

val create : value:'a -> type_id:'a Type_equal.Id.t -> shape:Shape.t -> t
val dims : t -> int array
val type_ : t -> Type.Packed.t
val shape : t -> Shape.t
val coerce : t -> type_id:'a Type_equal.Id.t -> 'a option
val coerce_exn : t -> type_id:'a Type_equal.Id.t -> 'a

(** Compares values by their unique [id]. *)
module On_id : Comparable.S_plain with type t := t

open! Core

type t =
  | T :
      { value : 'a
      ; type_id : 'a Type_equal.Id.t
      ; shape : Shape.t
      }
      -> t
[@@deriving sexp_of]

val dims : t -> int array
val type_ : t -> Type.Packed.t
val shape : t -> Shape.t
val coerce : t -> type_id:'a Type_equal.Id.t -> 'a option
val coerce_exn : t -> type_id:'a Type_equal.Id.t -> 'a

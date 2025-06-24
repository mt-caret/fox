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

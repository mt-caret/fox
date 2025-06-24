open! Core

type _ t =
  | Float : float t
  | Bool : bool t

let sexp_of_t (type a) _sexp_of_a (t : a t) =
  [%sexp_of: [ `Float | `Bool ]]
    (match t with
     | Float -> `Float
     | Bool -> `Bool)
;;

let bool = Type_equal.Id.create ~name:"Bool" [%sexp_of: bool]
let float = Type_equal.Id.create ~name:"Float" [%sexp_of: float]

let type_equal_id (type a) (t : a t) : a Type_equal.Id.t =
  match t with
  | Bool -> bool
  | Float -> float
;;

module Packed = struct
  type 'a typed = 'a t
  type t = T : 'a typed -> t

  let all = [ T Bool; T Float ]
end

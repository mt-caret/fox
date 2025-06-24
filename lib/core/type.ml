open! Core

type _ t =
  | Float : float t
  | Bool : bool t

let rank (type a) (t : a t) =
  match t with
  | Float -> 0
  | Bool -> 1
;;

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
  type 'a typed = 'a t [@@deriving sexp_of]
  type t = T : 'a typed -> t

  let compare (T t1) (T t2) = compare (rank t1) (rank t2)
  let equal = [%compare.equal: t]
  let sexp_of_t (T t) = [%sexp_of: _ typed] t

  let t_of_sexp sexp =
    match [%of_sexp: [ `Float | `Bool ]] sexp with
    | `Float -> T Float
    | `Bool -> T Bool
  ;;

  let all = [ T Bool; T Float ]
end

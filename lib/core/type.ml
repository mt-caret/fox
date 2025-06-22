open! Core

type _ t =
  | Bool : bool t
  | Float : float t

let bool = Type_equal.Id.create ~name:"Bool" [%sexp_of: bool]
let float = Type_equal.Id.create ~name:"Float" [%sexp_of: float]

let type_equal_id (type a) (t : a t) : a Type_equal.Id.t =
  match t with
  | Bool -> bool
  | Float -> float
;;

module Packed = struct
  module T = struct
    type 'a typed = 'a t
    type t = T : 'a typed -> t
  end

  include T

  include
    Sexpable.Of_sexpable
      (struct
        type t =
          [ `Bool
          | `Float
          ]
        [@@deriving sexp]
      end)
      (struct
        include T

        let to_sexpable = function
          | T Bool -> `Bool
          | T Float -> `Float
        ;;

        let of_sexpable = function
          | `Bool -> T Bool
          | `Float -> T Float
        ;;
      end)

  let all = [ T Bool; T Float ]

  let _remember_to_add_to_all =
    fun (type a) (t : a typed) ->
    match t with
    | Bool -> ()
    | Float -> ()
  ;;
end

let sexp_of_t _sexp_of_a t = Packed.sexp_of_t (T t)

open! Core

module Dual_number = struct
  type t =
    { primal : Value.t
    ; tangent : Value.t option
    ; id : Id.t
    }
  [@@deriving sexp_of, fields ~getters]

  let type_id = Type_equal.Id.create ~name:"Dual_number" [%sexp_of: t]
  let to_value t : Value.t = Value.create ~value:t ~type_id ~shape:(Value.shape t.primal)
end

type t = { id : Id.t }

let create () = { id = Id.create () }

let dual_number t ~primal ~tangent : Dual_number.t =
  Option.iter tangent ~f:(fun tangent ->
    [%test_eq: int iarray] (Value.dims primal) (Value.dims tangent);
    assert (Type.Packed.equal (Value.type_ primal) (Value.type_ tangent)));
  { primal; tangent; id = t.id }
;;

let lift
  t
  (T { value = _; type_id = _; shape = { dims; type_ }; id = _ } as value : Value.t)
  : Dual_number.t
  =
  let zeros () = Value.of_typed_tensor (Tensor.Typed.create Float ~dims 0.) in
  match Value.coerce value ~type_id:Dual_number.type_id with
  | Some x ->
    if Id.equal t.id x.id
    then x
    else dual_number t ~primal:value ~tangent:(Some (zeros ()))
  | None ->
    (match type_ with
     | T Float -> dual_number t ~primal:value ~tangent:(Some (zeros ()))
     | T Bool -> dual_number t ~primal:value ~tangent:None)
;;

let handle t ~f =
  Fox_effect.handle ~f ~handle:(fun op ->
    let result =
      match Op.map op ~f:(lift t) with
      | Unary (Neg, a) ->
        dual_number
          t
          ~primal:Value.O.(-a.primal)
          ~tangent:(Option.map a.tangent ~f:Value.neg)
      | Unary (Sin, a) ->
        dual_number
          t
          ~primal:(Value.sin a.primal)
          ~tangent:
            (Option.map a.tangent ~f:(fun tangent ->
               Value.O.(Value.cos a.primal * tangent)))
      | Unary (Cos, a) ->
        dual_number
          t
          ~primal:(Value.cos a.primal)
          ~tangent:
            (Option.map a.tangent ~f:(fun tangent ->
               Value.O.(-Value.sin a.primal * tangent)))
      | Unary (Sqrt, a) ->
        dual_number
          t
          ~primal:(Value.sqrt a.primal)
          ~tangent:
            (Option.map a.tangent ~f:(fun tangent ->
               Value.div tangent (Value.scale (Value.sqrt a.primal) 2.)))
      | Unary (Exp, a) ->
        dual_number
          t
          ~primal:(Value.exp a.primal)
          ~tangent:
            (Option.map a.tangent ~f:(fun tangent ->
               Value.O.(Value.exp a.primal * tangent)))
      | Unary (Log, a) ->
        dual_number
          t
          ~primal:(Value.log a.primal)
          ~tangent:(Option.map a.tangent ~f:(fun tangent -> Value.div tangent a.primal))
      | Unary (Sigmoid, a) ->
        dual_number
          t
          ~primal:(Value.sigmoid a.primal)
          ~tangent:
            (Option.map a.tangent ~f:(fun tangent ->
               Value.O.(
                 Value.sigmoid a.primal
                 * ((Value.of_float 1. |> Value.broadcast ~dims:(Value.dims a.primal))
                    - Value.sigmoid a.primal)
                 * tangent)))
      | Binary (Add, a, b) ->
        dual_number
          t
          ~primal:Value.O.(a.primal + b.primal)
          ~tangent:
            (Option.map2 a.tangent b.tangent ~f:(fun a_tangent b_tangent ->
               Value.O.(a_tangent + b_tangent)))
      | Binary (Sub, a, b) ->
        dual_number
          t
          ~primal:Value.O.(a.primal - b.primal)
          ~tangent:
            (Option.map2 a.tangent b.tangent ~f:(fun a_tangent b_tangent ->
               Value.O.(a_tangent - b_tangent)))
      | Binary (Mul, a, b) ->
        dual_number
          t
          ~primal:Value.O.(a.primal * b.primal)
          ~tangent:
            (Option.map2 a.tangent b.tangent ~f:(fun a_tangent b_tangent ->
               Value.O.((a_tangent * b.primal) + (a.primal * b_tangent))))
      | Binary (Div, a, b) ->
        dual_number
          t
          ~primal:Value.O.(a.primal / b.primal)
          ~tangent:
            (Option.map2 a.tangent b.tangent ~f:(fun a_tangent b_tangent ->
               Value.O.(
                 ((a_tangent * b.primal) - (a.primal * b_tangent)) / (b.primal * b.primal))))
      | Binary (Eq, a, b) ->
        dual_number t ~primal:Value.O.(a.primal = b.primal) ~tangent:None
      | Binary (Gt, a, b) ->
        dual_number t ~primal:Value.O.(a.primal > b.primal) ~tangent:None
      | Binary (Lt, a, b) ->
        dual_number t ~primal:Value.O.(a.primal < b.primal) ~tangent:None
      | Matmul (a, b) ->
        dual_number
          t
          ~primal:(Value.matmul a.primal b.primal)
          ~tangent:
            (Option.map2 a.tangent b.tangent ~f:(fun a_tangent b_tangent ->
               Value.O.(Value.matmul a_tangent b.primal + Value.matmul a.primal b_tangent)))
      | Transpose a ->
        dual_number
          t
          ~primal:(Value.transpose a.primal)
          ~tangent:(Option.map a.tangent ~f:Value.transpose)
      | Sum { value; dims; keep_dims } ->
        dual_number
          t
          ~primal:(Value.sum ~dims ~keep_dims value.primal)
          ~tangent:(Option.map value.tangent ~f:(Value.sum ~dims ~keep_dims))
      | Broadcast { value; dims } ->
        dual_number
          t
          ~primal:(Value.broadcast ~dims value.primal)
          ~tangent:(Option.map value.tangent ~f:(Value.broadcast ~dims))
      | Reshape { value; dims } ->
        dual_number
          t
          ~primal:(Value.reshape value.primal ~dims)
          ~tangent:(Option.map value.tangent ~f:(Value.reshape ~dims))
    in
    Dual_number.to_value result)
;;

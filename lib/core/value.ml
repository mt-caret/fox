open! Core

module T = struct
  include Value0

  let tree_of_t t = Value_tree.of_value t
  let t_of_tree tree = Value_tree.to_value_exn tree
end

include T

let tree_def ~dims = Value_tree.Def.leaf ~dims

let of_typed_tensor (type a) (tensor : a Tensor.Typed.t) =
  let type_id =
    Tensor.Typed.type_ tensor |> Type.type_equal_id |> Tensor.Typed.type_equal_id
  in
  T { value = tensor; type_id; shape = Tensor.Typed.shape tensor }
;;

let of_tensor (Tensor.T tensor) = of_typed_tensor tensor

let to_typed_tensor_exn type_ t =
  coerce_exn t ~type_id:(Type.type_equal_id type_ |> Tensor.Typed.type_equal_id)
;;

let to_tensor_exn t =
  let (T type_) = shape t |> Shape.type_ in
  Tensor.T
    (coerce_exn t ~type_id:(Type.type_equal_id type_ |> Tensor.Typed.type_equal_id))
;;

let of_float x = of_typed_tensor (Tensor.Typed.of_lit Float x)
let to_float_exn t : float = to_typed_tensor_exn Float t |> Tensor.Typed.item

include Op.Make_operators (struct
    type nonrec t = t [@@deriving sexp_of]

    let of_float float = of_typed_tensor (Tensor.Typed.of_lit Float float)
    let eval op = Effect.perform (Fox_effect.Op op)
    let shape = shape
  end)

module Tuple2 = struct
  include Treeable.Tuple2 (T) (T)

  let tree_def ~dims1 ~dims2 =
    tree_of_t
      ( of_typed_tensor (Tensor.Typed.zeros ~dims:dims1)
      , of_typed_tensor (Tensor.Typed.zeros ~dims:dims2) )
    |> Value_tree.to_def
  ;;
end

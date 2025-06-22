open! Core
include Value0

let tree_def ~dims = Value_tree.Def.leaf ~dims

let of_tensor value =
  let type_id = Tensor.type_equal_id (Type.type_equal_id (Tensor.type_ value)) in
  T { value; type_id; dims = Tensor.dims value; type_ = Tensor.type_ value }
;;

let to_tensor_exn (type a) (T { value; type_id; dims = _; type_ } as t : a t) : a Tensor.t
  =
  match
    Type_equal.Id.same_witness type_id (Tensor.type_equal_id (Type.type_equal_id type_))
  with
  | Some T -> value
  | None -> raise_s [%message "Invalid value" (t : _ t)]
;;

let of_float x = of_tensor (Tensor.of_lit Float x)
let to_float_exn t : float = to_tensor_exn t |> Tensor.item

include Op.Make_operators (struct
    include Value0

    let of_float float = of_float float
    let eval op = Effect.perform (Fox_effect.Op op)
    let dims = dims
  end)

module Packed = struct
  include Value0.Packed

  let tree_of_t t = Value_tree.of_value t
  let t_of_tree tree = Value_tree.to_value_exn tree
end

module Tuple2 = struct
  include Treeable.Tuple2 (Packed) (Packed)

  let tree_def ~dims1 ~dims2 =
    tree_of_t
      (T (of_tensor (Tensor.zeros ~dims:dims1)), T (of_tensor (Tensor.zeros ~dims:dims2)))
    |> Value_tree.to_def
  ;;
end

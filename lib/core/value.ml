open! Core

module T = struct
  include Value0

  let tree_of_t t = Value_tree.of_value t
  let t_of_tree tree = Value_tree.to_value_exn tree
end

include T

let tree_def ~dims = Value_tree.Def.leaf ~dims
let of_tensor value = T { value; type_id = Tensor.type_id; dims = Tensor.dims value }

let to_tensor_exn (T { value; type_id; dims = _ } as t) : Tensor.t =
  match Type_equal.Id.same_witness type_id Tensor.type_id with
  | Some T -> value
  | None -> raise_s [%message "Invalid value" (t : t)]
;;

let of_float x = of_tensor (Tensor.of_float x)
let to_float_exn t : float = to_tensor_exn t |> Tensor.item

include Op.Make_operators (struct
    type value = t [@@deriving sexp_of]

    let of_float float = of_tensor (Tensor.of_float float)
    let eval op = Effect.perform (Fox_effect.Op op)
    let dims = dims
  end)

module Tuple2 = struct
  include Treeable.Tuple2 (T) (T)

  let tree_def ~dims1 ~dims2 =
    tree_of_t (of_tensor (Tensor.zeros ~dims:dims1), of_tensor (Tensor.zeros ~dims:dims2))
    |> Value_tree.to_def
  ;;
end

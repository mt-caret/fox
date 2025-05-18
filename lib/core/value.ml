open! Core

module T = struct
  include Value0

  let tree_def = Value_tree.Def.leaf
  let tree_of_t t = Value_tree.of_value t
  let t_of_tree tree = Value_tree.to_value_exn tree
end

include T

let dims (T { value = _; type_id = _; dims }) = dims

let of_tensor value =
  T { value; type_id = Tensor.type_id; dims = Some (Tensor.dims value |> Array.to_list) }
;;

let to_tensor_exn (T { value; type_id; dims = _ } as t) : Tensor.t =
  match Type_equal.Id.same_witness type_id Tensor.type_id with
  | Some T -> value
  | None -> raise_s [%message "Invalid value" (t : t)]
;;

let of_float x = of_tensor (Tensor.of_float x)
let to_float_exn t : float = to_tensor_exn t |> Tensor.item
let add a b = Effect.perform (Fox_effect.Op (Add (a, b)))
let sub a b = Effect.perform (Fox_effect.Op (Sub (a, b)))
let mul a b = Effect.perform (Fox_effect.Op (Mul (a, b)))
let neg a = Effect.perform (Fox_effect.Op (Neg a))
let sin a = Effect.perform (Fox_effect.Op (Sin a))
let cos a = Effect.perform (Fox_effect.Op (Cos a))
let matmul a b = Effect.perform (Fox_effect.Op (Matmul (a, b)))
let transpose a = Effect.perform (Fox_effect.Op (Transpose a))

module O = struct
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul
  let ( ~- ) = neg
end

module Tuple2 = struct
  include Treeable.Tuple2 (T) (T)

  (* TODO: is this correct? *)
  let tree_def = Value_tree.to_def (tree_of_t (of_float 0., of_float 0.))
end

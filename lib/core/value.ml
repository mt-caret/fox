open! Core

module T = struct
  include Value0

  let tree_def = Value_tree.Def.leaf
  let tree_of_t t = Value_tree.of_value t
  let t_of_tree tree = Value_tree.to_value_exn tree
end

include T

let of_tensor t = T (t, Tensor.type_id)

let to_tensor_exn (T (x, id) as t) : Tensor.t =
  match Type_equal.Id.same_witness id Tensor.type_id with
  | Some T -> x
  | None -> raise_s [%message "Invalid value" (t : t)]
;;

let of_float x = of_tensor (Tensor.of_float x)
let to_float_exn t : float = to_tensor_exn t |> Tensor.item
let add a b = Effect.perform (Ox_effect.Op (Add (a, b)))
let sub a b = Effect.perform (Ox_effect.Op (Sub (a, b)))
let mul a b = Effect.perform (Ox_effect.Op (Mul (a, b)))
let neg a = Effect.perform (Ox_effect.Op (Neg a))
let sin a = Effect.perform (Ox_effect.Op (Sin a))
let cos a = Effect.perform (Ox_effect.Op (Cos a))

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

open! Core
open! Fox_core
module Kind = Torch_core.Kind

let double = Kind.(T Double)
let uint8 = Kind.(T Uint8)

let of_tensor (tensor : Tensor.t) : Torch.Tensor.t =
  match Tensor.type_ tensor with
  | T Float ->
    Tensor.to_typed_exn Float tensor
    |> Tensor.Private.to_bigarray
    |> Torch.Tensor.of_bigarray
  | T Bool ->
    Tensor.to_typed_exn Bool tensor
    |> Tensor.Private.to_char_bigarray
    |> Torch.Tensor.of_bigarray
;;

let to_tensor (t : Torch.Tensor.t) : Tensor.t =
  let of_double_tensor t =
    Torch.Tensor.to_bigarray t ~kind:Bigarray.float64
    |> Tensor.Private.of_float_bigarray
    |> Tensor.of_typed
  in
  let of_uint8_tensor t =
    Torch.Tensor.to_bigarray t ~kind:Bigarray.char
    |> Tensor.Private.of_char_bigarray
    |> Tensor.of_typed
  in
  let open Kind in
  match Torch.Tensor.kind t with
  | T Double -> of_double_tensor t
  | T Float -> of_double_tensor (Torch.Tensor.to_type t ~type_:double)
  | T Uint8 -> of_uint8_tensor t
  | T Bool -> of_uint8_tensor (Torch.Tensor.to_type t ~type_:uint8)
  | T _ ->
    raise_s
      [%message
        "Pytorch.to_tensor: unsupported torch kind"
          ~shape:(Torch.Tensor.shape t : int list)]
;;

module Backend = struct
  type t = Torch.Tensor.t

  include Op.Make_operators (struct
      type nonrec t = t

      let sexp_of_t t = [%sexp_of: int list] (Torch.Tensor.shape t)
      let of_float f = Torch.Tensor.of_double0 f

      let shape (t : t) : Shape.t =
        let dims = Torch.Tensor.shape t |> Array.of_list |> Iarray.of_array in
        let open Kind in
        match Torch.Tensor.kind t with
        | T Double | T Float -> { dims; type_ = T Float }
        | T Bool | T Uint8 -> { dims; type_ = T Bool }
        | T _ ->
          raise_s
            [%message
              "Pytorch: unsupported torch kind" ~shape:(Iarray.to_list dims : int list)]
      ;;

      let eval (op : t Op.t) : t =
        let module Tt = Torch.Tensor in
        match op with
        | Unary (kind, a) ->
          (match kind with
           | Neg -> Tt.neg a
           | Sin -> Tt.sin a
           | Cos -> Tt.cos a
           | Sqrt -> Tt.sqrt a
           | Exp -> Tt.exp a
           | Log -> Tt.log a
           | Sigmoid -> Tt.sigmoid a)
        | Binary (kind, a, b) ->
          (match kind with
           | Add -> Tt.add a b
           | Sub -> Tt.sub a b
           | Mul -> Tt.mul a b
           | Div -> Tt.div a b
           | Eq -> Tt.eq_tensor a b
           | Gt -> Tt.gt_tensor a b
           | Lt -> Tt.lt_tensor a b)
        | Matmul (a, b) -> Tt.matmul a b
        | Transpose a -> Tt.transpose a ~dim0:0 ~dim1:1
        | Sum { value; dims; keep_dims } ->
          let dim =
            match dims with
            | `All -> List.init (List.length (Tt.shape value)) ~f:Fn.id
            | `Just dims -> Nonempty_list.to_list dims
          in
          Tt.sum_dim_intlist value ~dim:(Some dim) ~keepdim:keep_dims ~dtype:double
        | Broadcast { value; dims } -> Tt.broadcast_to value ~size:(Iarray.to_list dims)
        | Reshape { value; dims } -> Tt.reshape value ~shape:(Iarray.to_list dims)
      ;;
    end)
end

let handle ~f =
  Fox_effect.handle ~f ~handle:(fun op ->
    Op.map op ~f:(fun value -> Value.to_tensor_exn value |> of_tensor)
    |> Op.eval (module Backend)
    |> to_tensor
    |> Value.of_tensor)
;;

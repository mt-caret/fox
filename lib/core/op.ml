open! Core

module Unary = struct
  type t =
    | Neg
    | Sin
    | Cos
    | Sqrt
  [@@deriving sexp, enumerate]

  let to_string t = [%sexp_of: t] t |> Sexp.to_string |> String.lowercase
end

module Binary = struct
  type t =
    | Add
    | Sub
    | Mul
    | Div
  [@@deriving sexp, enumerate]

  let to_string t = [%sexp_of: t] t |> Sexp.to_string |> String.lowercase
end

type 'value t =
  | Unary of Unary.t * 'value
  | Binary of Binary.t * 'value * 'value
  | Matmul of 'value * 'value
  | Transpose of 'value
  | Sum of
      { value : 'value
      ; dims : [ `Just of int array | `All ]
      ; keep_dims : bool
      }
  | Broadcast of
      { value : 'value
      ; dims : int array
      }
[@@deriving sexp_of, variants]

let map t ~f =
  match t with
  | Unary (kind, a) -> Unary (kind, f a)
  | Binary (kind, a, b) -> Binary (kind, f a, f b)
  | Matmul (a, b) -> Matmul (f a, f b)
  | Transpose a -> Transpose (f a)
  | Sum { value; dims; keep_dims } -> Sum { value = f value; dims; keep_dims }
  | Broadcast { value; dims } -> Broadcast { value = f value; dims }
;;

let eval (type a) (module M : Operators_intf.S with type t = a) (t : a t) =
  match t with
  | Unary (kind, a) ->
    let f =
      match kind with
      | Neg -> M.neg
      | Sin -> M.sin
      | Cos -> M.cos
      | Sqrt -> M.sqrt
    in
    f a
  | Binary (kind, a, b) ->
    let f =
      match kind with
      | Add -> M.add
      | Sub -> M.sub
      | Mul -> M.mul
      | Div -> M.div
    in
    f a b
  | Matmul (a, b) -> M.matmul a b
  | Transpose a -> M.transpose a
  | Sum { value; dims; keep_dims } -> M.sum value ~dims ~keep_dims
  | Broadcast { value; dims } -> M.broadcast value ~dims
;;

let to_string t ~f =
  match t with
  | Unary (kind, a) -> [%string "%{kind#Unary} %{f a}"]
  | Binary (kind, a, b) -> [%string "%{kind#Binary} %{f a} %{f b}"]
  | Matmul (a, b) -> [%string "matmul %{f a} %{f b}"]
  | Transpose a -> [%string "transpose %{f a}"]
  | Sum { value; dims; keep_dims } ->
    let dims =
      match dims with
      | `All -> "all"
      | `Just dims ->
        let dims =
          Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:", "
        in
        [%string "[%{dims}]"]
    in
    [%string "sum %{f value} dims=%{dims} keep_dims=%{keep_dims#Bool}"]
  | Broadcast { value; dims } ->
    let dims =
      Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:", "
    in
    [%string "broadcast %{f value} dims=[%{dims}]"]
;;

(* TODO: test against operations in tensor.ml *)
let infer_dims = function
  | Unary ((Neg | Sin | Cos | Sqrt), dims) -> dims
  | Binary ((Add | Sub | Mul | Div), dims1, dims2) ->
    [%test_eq: int array] dims1 dims2;
    dims1
  | Matmul (dims1, dims2) ->
    (match dims1, dims2 with
     | [| n; m |], [| m' |] ->
       [%test_eq: int] m m';
       [| n |]
     | [| n; m |], [| m'; k |] ->
       [%test_eq: int] m m';
       [| n; k |]
     | _ ->
       raise_s
         [%message
           "infer_dims: Invalid matmul dimensions" (dims1 : int array) (dims2 : int array)])
  | Transpose dims ->
    (match dims with
     | [| n; k |] -> [| k; n |]
     | dims ->
       raise_s [%message "infer_dims: Invalid transpose dimensions" (dims : int array)])
  | Sum { value = dims; dims = dims_to_sum; keep_dims } ->
    (match dims_to_sum with
     | `All -> if keep_dims then Array.map dims ~f:(fun _ -> 1) else [||]
     | `Just dims_to_sum ->
       let dims_length = Array.length dims in
       [%test_pred: int array]
         ~message:"infer_dims: sum: dims out of bounds"
         (Array.for_all ~f:(fun dim -> dim < dims_length || dims_length + dim >= 0))
         dims_to_sum;
       let dims_to_sum =
         Array.map dims_to_sum ~f:(fun dim -> if dim < 0 then dims_length + dim else dim)
         |> Int.Set.of_array
       in
       if keep_dims
       then
         Array.mapi dims ~f:(fun index dim ->
           if Set.mem dims_to_sum index then 1 else dim)
       else Array.filteri dims ~f:(fun index _dim -> not (Set.mem dims_to_sum index)))
  | Broadcast { value = from_dims; dims = to_dims } ->
    if Array.length to_dims < Array.length from_dims
    then
      raise_s
        [%message
          "broadcast: can't broadcast to a larger rank"
            (from_dims : int array)
            (to_dims : int array)];
    let dims_padding_length = Array.length to_dims - Array.length from_dims in
    let padded_from_dims =
      Array.append (Array.create ~len:dims_padding_length 1) from_dims
    in
    Array.zip_exn padded_from_dims to_dims
    |> [%test_pred: (int * int) array]
         ~message:"broadcast: can't broadcast"
         (Array.for_all ~f:(fun (from, to_) -> to_ >= from && to_ % from = 0));
    to_dims
;;

module Make_operators (M : sig
    type value [@@deriving sexp_of]

    val of_float : float -> value
    val eval : value t -> value
    val dims : value -> int array
  end) : Operators_intf.S with type t := M.value = struct
  let eval =
    fun t ->
    let inferred_out_dims = map t ~f:M.dims |> infer_dims in
    let out = M.eval t in
    [%test_result: int array]
      (M.dims out)
      ~expect:inferred_out_dims
      ~message:"[Op.infer_dims] dims mismatch with actual dims";
    out
  ;;

  let neg a = eval (Unary (Neg, a))
  let sin a = eval (Unary (Sin, a))
  let cos a = eval (Unary (Cos, a))
  let sqrt a = eval (Unary (Sqrt, a))
  let add a b = eval (Binary (Add, a, b))
  let sub a b = eval (Binary (Sub, a, b))
  let mul a b = eval (Binary (Mul, a, b))
  let div a b = eval (Binary (Div, a, b))
  let matmul a b = eval (Matmul (a, b))
  let transpose a = eval (Transpose a)

  let sum ?(dims = `All) ?(keep_dims = false) value =
    eval (Sum { value; dims; keep_dims })
  ;;

  let broadcast value ~dims = eval (Broadcast { value; dims })

  module O = struct
    let ( ~- ) = neg
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
  end

  let scale value float = O.(value * broadcast (M.of_float float) ~dims:(M.dims value))
  let length value = Array.fold (M.dims value) ~init:1 ~f:( * )

  let mean ?dims ?keep_dims value =
    let sum = sum ?dims ?keep_dims value in
    let reduction_factor = length value / length sum |> Int.to_float |> M.of_float in
    O.(sum / broadcast reduction_factor ~dims:(M.dims sum))
  ;;

  let var ?dims ?keep_dims ?(correction = true) value =
    let mean = mean ?dims ~keep_dims:true value |> broadcast ~dims:(M.dims value) in
    let diff = O.(value - mean) in
    let diff_squared = O.(diff * diff) in
    let sum = sum ?dims ?keep_dims diff_squared in
    let reduction_factor = length value / length sum in
    let reduction_factor =
      (if correction then reduction_factor - 1 else reduction_factor)
      |> Int.to_float
      |> M.of_float
    in
    O.(sum / broadcast reduction_factor ~dims:(M.dims sum))
  ;;

  let std ?dims ?keep_dims ?correction value =
    var ?dims ?keep_dims ?correction value |> sqrt
  ;;
end

open! Core

type 'value t =
  | Add of 'value * 'value
  | Sub of 'value * 'value
  | Mul of 'value * 'value
  | Div of 'value * 'value
  | Neg of 'value
  | Sin of 'value
  | Cos of 'value
  | Sqrt of 'value
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
  | Add (a, b) -> Add (f a, f b)
  | Sub (a, b) -> Sub (f a, f b)
  | Mul (a, b) -> Mul (f a, f b)
  | Div (a, b) -> Div (f a, f b)
  | Neg a -> Neg (f a)
  | Sin a -> Sin (f a)
  | Cos a -> Cos (f a)
  | Sqrt a -> Sqrt (f a)
  | Matmul (a, b) -> Matmul (f a, f b)
  | Transpose a -> Transpose (f a)
  | Sum { value; dims; keep_dims } -> Sum { value = f value; dims; keep_dims }
  | Broadcast { value; dims } -> Broadcast { value = f value; dims }
;;

let eval (type a) (module M : Operators_intf.S with type t = a) (t : a t) =
  match t with
  | Add (a, b) -> M.add a b
  | Sub (a, b) -> M.sub a b
  | Mul (a, b) -> M.mul a b
  | Div (a, b) -> M.div a b
  | Neg a -> M.neg a
  | Sin a -> M.sin a
  | Cos a -> M.cos a
  | Sqrt a -> M.sqrt a
  | Matmul (a, b) -> M.matmul a b
  | Transpose a -> M.transpose a
  | Sum { value; dims; keep_dims } -> M.sum value ~dims ~keep_dims
  | Broadcast { value; dims } -> M.broadcast value ~dims
;;

let to_string t ~f =
  match t with
  | Add (a, b) -> [%string "add %{f a} %{f b}"]
  | Sub (a, b) -> [%string "sub %{f a} %{f b}"]
  | Mul (a, b) -> [%string "mul %{f a} %{f b}"]
  | Div (a, b) -> [%string "div %{f a} %{f b}"]
  | Neg a -> [%string "neg %{f a}"]
  | Sin a -> [%string "sin %{f a}"]
  | Cos a -> [%string "cos %{f a}"]
  | Sqrt a -> [%string "sqrt %{f a}"]
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
  | Add (dims1, dims2) | Sub (dims1, dims2) | Mul (dims1, dims2) | Div (dims1, dims2) ->
    [%test_eq: int array] dims1 dims2;
    dims1
  | Neg dims | Sin dims | Cos dims | Sqrt dims -> dims
  | Matmul (dims1, dims2) ->
    (match dims1, dims2 with
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
    type value

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

  let add a b = eval (Add (a, b))
  let sub a b = eval (Sub (a, b))
  let mul a b = eval (Mul (a, b))
  let div a b = eval (Div (a, b))
  let neg a = eval (Neg a)
  let sin a = eval (Sin a)
  let cos a = eval (Cos a)
  let sqrt a = eval (Sqrt a)
  let matmul a b = eval (Matmul (a, b))
  let transpose a = eval (Transpose a)

  let sum ?(dims = `All) ?(keep_dims = false) value =
    eval (Sum { value; dims; keep_dims })
  ;;

  let broadcast value ~dims = eval (Broadcast { value; dims })

  module O = struct
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) = div
    let ( ~- ) = neg
  end

  let scale value float = O.(value * broadcast (M.of_float float) ~dims:(M.dims value))

  let mean ?dims ?keep_dims value =
    let sum = sum ?dims ?keep_dims value in
    let sum_dims = M.dims sum in
    let reduction_factor =
      Array.fold (M.dims value) ~init:1 ~f:( * ) / Array.fold sum_dims ~init:1 ~f:( * )
    in
    let reduction_factor = M.of_float (Int.to_float reduction_factor) in
    O.(sum / broadcast reduction_factor ~dims:sum_dims)
  ;;
end

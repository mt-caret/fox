open! Core

module Unary = struct
  type t =
    | Neg
    | Sin
    | Cos
    | Sqrt
    | Exp
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
      ; dims : [ `Just of int Nonempty_list.t | `All ]
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

let to_list = function
  | Unary (_, a) -> [ a ]
  | Binary (_, a, b) -> [ a; b ]
  | Matmul (a, b) -> [ a; b ]
  | Transpose a -> [ a ]
  | Sum { value; dims = _; keep_dims = _ } -> [ value ]
  | Broadcast { value; dims = _ } -> [ value ]
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
      | Exp -> M.exp
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
          Nonempty_list.to_list dims
          |> List.map ~f:Int.to_string
          |> String.concat ~sep:", "
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

let infer_dims t =
  match t with
  | Unary ((Neg | Sin | Cos | Sqrt | Exp), dims) -> Ok dims
  | Binary ((Add | Sub | Mul | Div), dims1, dims2) ->
    let%map.Or_error () =
      if [%equal: int array] dims1 dims2
      then Ok ()
      else Or_error.error_s [%message "infer_dims: dims mismatch" ~op:(t : int array t)]
    in
    dims1
  | Matmul (dims1, dims2) ->
    let dim_mismatch () =
      Or_error.error_s
        [%message "infer_dims: Matmul: dims mismatch" ~op:(t : int array t)]
    in
    (match dims1, dims2 with
     | [| n; m |], [| m' |] -> if m <> m' then dim_mismatch () else Ok [| n |]
     | [| n; m |], [| m'; k |] -> if m <> m' then dim_mismatch () else Ok [| n; k |]
     | _ ->
       Or_error.error_s
         [%message
           "infer_dims: Matmul: unsupported matmul dimensions" ~op:(t : int array t)])
  | Transpose dims ->
    (match dims with
     | [| n; k |] -> Ok [| k; n |]
     | _ ->
       Or_error.error_s
         [%message
           "infer_dims: Transpose: unsupported transpose dimensions" ~op:(t : int array t)])
  | Sum { value = dims; dims = dims_to_sum; keep_dims } ->
    (match dims_to_sum with
     | `All -> Ok (if keep_dims then Array.map dims ~f:(fun _ -> 1) else [||])
     | `Just dims_to_sum ->
       let%bind.Or_error () =
         if Nonempty_list.to_list dims_to_sum |> List.contains_dup ~compare:Int.compare
         then
           Or_error.error_s
             [%message
               "infer_dims: Sum: duplicate reduction dimension" ~op:(t : int array t)]
         else Ok ()
       in
       let dims_length = Array.length dims in
       (match
          Nonempty_list.for_all dims_to_sum ~f:(fun dim ->
            dim < dims_length || dims_length + dim >= 0)
        with
        | false ->
          Or_error.error_s
            [%message "infer_dims: dims out of bounds" ~op:(t : int array t)]
        | true ->
          let dims =
            let dims_to_sum =
              Nonempty_list.map dims_to_sum ~f:(fun dim ->
                if dim < 0 then dims_length + dim else dim)
              |> Nonempty_list.to_list
              |> Int.Set.of_list
            in
            if keep_dims
            then
              Array.mapi dims ~f:(fun index dim ->
                if Set.mem dims_to_sum index then 1 else dim)
            else Array.filteri dims ~f:(fun index _dim -> not (Set.mem dims_to_sum index))
          in
          Ok dims))
  | Broadcast { value = from_dims; dims = to_dims } ->
    let%bind.Or_error () =
      if Array.for_all to_dims ~f:Int.is_positive
      then Ok ()
      else
        Or_error.error_s
          [%message "infer_dims: dims must be positive" ~op:(t : int array t)]
    in
    let%bind.Or_error () =
      if Array.length to_dims >= Array.length from_dims
      then Ok ()
      else
        Or_error.error_s
          [%message "infer_dims: can't broadcast to a larger rank" ~op:(t : int array t)]
    in
    let%bind.Or_error () =
      let dims_padding_length = Array.length to_dims - Array.length from_dims in
      let padded_from_dims =
        Array.append (Array.create ~len:dims_padding_length 1) from_dims
      in
      if
        Array.zip_exn padded_from_dims to_dims
        |> Array.for_all ~f:(fun (from, to_) -> to_ = from || from = 1)
      then Ok ()
      else Or_error.error_s [%message "infer_dims: can't broadcast" ~op:(t : int array t)]
    in
    Ok to_dims
;;

let infer_dims_exn t = infer_dims t |> ok_exn

module Make_operators (M : sig
    type value [@@deriving sexp_of]

    val of_float : float -> value
    val eval : value t -> value
    val dims : value -> int array
  end) : Operators_intf.S with type t := M.value = struct
  let eval =
    fun t ->
    let inferred_out_dims = map t ~f:M.dims |> infer_dims_exn in
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
  let exp a = eval (Unary (Exp, a))
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

  let sigmoid a =
    let one = M.of_float 1. |> broadcast ~dims:(M.dims a) in
    O.(one / (one + exp (-a)))
  ;;

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

  let softmax ~dim value =
    let exp_value = exp value in
    let sum_exp_value =
      sum exp_value ~dims:(`Just [ dim ]) ~keep_dims:true
      |> broadcast ~dims:(M.dims value)
    in
    O.(exp_value / sum_exp_value)
  ;;
end

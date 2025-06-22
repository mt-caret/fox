open! Core

module Unary = struct
  type t =
    | Neg
    | Sin
    | Cos
    | Sqrt
    | Exp
    | Log
    | Sigmoid
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

type ('type_, 'f) t =
  | Unary : Unary.t * (float -> 'f) Higher_kinded.t -> (float, 'f) t
  | Binary :
      Binary.t * (float -> 'f) Higher_kinded.t * (float -> 'f) Higher_kinded.t
      -> (float, 'f) t
  | Matmul :
      (float -> 'f) Higher_kinded.t * (float -> 'f) Higher_kinded.t
      -> (float, 'f) t
  | Transpose : ('type_ -> 'f) Higher_kinded.t -> ('type_, 'f) t
  | Sum :
      { value : (float -> 'f) Higher_kinded.t
      ; dims : [ `Just of int Nonempty_list.t | `All ]
      ; keep_dims : bool
      }
      -> (float, 'f) t
  | Broadcast :
      { value : ('type_ -> 'f) Higher_kinded.t
      ; dims : int array
      }
      -> ('type_, 'f) t
  | Reshape :
      { value : ('type_ -> 'f) Higher_kinded.t
      ; dims : int array
      }
      -> ('type_, 'f) t
[@@deriving variants]

module Non_higher_kinded = struct
  type ('type_, 'f) higher_kinded = ('type_, 'f) t

  type 'a t =
    | Unary of Unary.t * 'a
    | Binary of Binary.t * 'a * 'a
    | Matmul of 'a * 'a
    | Transpose of 'a
    | Sum of
        { value : 'a
        ; dims : [ `Just of int Nonempty_list.t | `All ]
        ; keep_dims : bool
        }
    | Broadcast of
        { value : 'a
        ; dims : int array
        }
    | Reshape of
        { value : 'a
        ; dims : int array
        }
  [@@deriving variants, sexp]

  type ('f, 'b) f = { f : 'a. ('a -> 'f) Higher_kinded.t -> 'b }

  let of_higher_kinded (type a f' b) (t : (a, f') higher_kinded) ~f:({ f } : (f', b) f)
    : b t
    =
    match t with
    | Unary (kind, a) -> Unary (kind, f a)
    | Binary (kind, a, b) -> Binary (kind, f a, f b)
    | Matmul (a, b) -> Matmul (f a, f b)
    | Transpose a -> Transpose (f a)
    | Sum { value; dims; keep_dims } -> Sum { value = f value; dims; keep_dims }
    | Broadcast { value; dims } -> Broadcast { value = f value; dims }
    | Reshape { value; dims } -> Reshape { value = f value; dims }
  ;;

  let to_list = function
    | Unary (_, a) -> [ a ]
    | Binary (_, a, b) -> [ a; b ]
    | Matmul (a, b) -> [ a; b ]
    | Transpose a -> [ a ]
    | Sum { value; dims = _; keep_dims = _ } -> [ value ]
    | Broadcast { value; dims = _ } -> [ value ]
    | Reshape { value; dims = _ } -> [ value ]
  ;;

  let infer_dims t =
    match t with
    | Unary ((Neg | Sin | Cos | Sqrt | Exp | Log | Sigmoid), dims) -> Ok dims
    | Binary (((Add | Sub | Mul | Div) as binary), dims1, dims2) ->
      let%map.Or_error () =
        if [%equal: int array] dims1 dims2
        then Ok ()
        else
          Or_error.error_s
            [%message
              "infer_dims: dims mismatch"
                (binary : Binary.t)
                (dims1 : int array)
                (dims2 : int array)]
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
             "infer_dims: Transpose: unsupported transpose dimensions"
               ~op:(t : int array t)])
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
              else
                Array.filteri dims ~f:(fun index _dim -> not (Set.mem dims_to_sum index))
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
            [%message
              "infer_dims: can't broadcast to a larger rank" ~op:(t : int array t)]
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
        else
          Or_error.error_s [%message "infer_dims: can't broadcast" ~op:(t : int array t)]
      in
      Ok to_dims
    | Reshape { value = from_dims; dims = to_dims } ->
      let from_dims_elements = Array.fold from_dims ~init:1 ~f:( * ) in
      (match Array.count to_dims ~f:(fun dim -> dim = -1) with
       | 0 ->
         let to_dims_elements = Array.fold to_dims ~init:1 ~f:( * ) in
         if from_dims_elements <> to_dims_elements
         then
           Or_error.error_s [%message "infer_dims: can't reshape" ~op:(t : int array t)]
         else Ok to_dims
       | 1 ->
         let length_without_unknown_dim =
           Array.filter to_dims ~f:(fun dim -> dim <> -1) |> Array.fold ~init:1 ~f:( * )
         in
         (match from_dims_elements % length_without_unknown_dim with
          | 0 ->
            let dims =
              Array.map to_dims ~f:(fun dim ->
                if dim = -1 then from_dims_elements / length_without_unknown_dim else dim)
            in
            Ok dims
          | _ ->
            Or_error.error_s
              [%message "infer_dims: no valid implicit dimension" ~op:(t : int array t)])
       | _ ->
         Or_error.error_s
           [%message "infer_dims: more than one -1 in dims" ~op:(t : int array t)])
  ;;
end

module Packed = struct
  type ('a, 'f) typed = ('a, 'f) t
  type 'f t = T : ('a, 'f) typed -> 'f t

  let sexp_of_t ~f (T t) =
    Non_higher_kinded.of_higher_kinded t ~f |> [%sexp_of: Sexp.t Non_higher_kinded.t]
  ;;
end

type ('f1, 'f2) f = { f : 'a. ('a -> 'f1) Higher_kinded.t -> ('a -> 'f2) Higher_kinded.t }

let map (type a f1 f2) (t : (a, f1) t) ~f:({ f } : (f1, f2) f) : (a, f2) t =
  match t with
  | Unary (kind, a) -> Unary (kind, f a)
  | Binary (kind, a, b) -> Binary (kind, f a, f b)
  | Matmul (a, b) -> Matmul (f a, f b)
  | Transpose a -> Transpose (f a)
  | Sum { value; dims; keep_dims } -> Sum { value = f value; dims; keep_dims }
  | Broadcast { value; dims } -> Broadcast { value = f value; dims }
  | Reshape { value; dims } -> Reshape { value = f value; dims }
;;

let to_string (type a f) (t : (a, f) t) ~(f : (a -> f) Higher_kinded.t -> string) =
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
  | Reshape { value; dims } ->
    let dims =
      Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:", "
    in
    [%string "reshape %{f value} dims=[%{dims}]"]
;;

module Dims = struct
  module T = struct
    type _ t = int array [@@deriving sexp_of]
  end

  include T
  include Higher_kinded.Make (T)
end

let infer_dims t =
  Non_higher_kinded.of_higher_kinded t ~f:{ f = Dims.project }
  |> Non_higher_kinded.infer_dims
;;

let infer_dims_exn t = infer_dims t |> ok_exn

module Make_operators (M : sig
    type ('a, 'f) op := ('a, 'f) t
    type 'a t [@@deriving sexp_of]

    include Higher_kinded.S with type 'a t := 'a t

    val of_float : float -> float t
    val eval : ('a, higher_kinded) op -> 'a t
    val dims : 'a t -> int array
  end) : Operators_intf.S with type 'a t := 'a M.t = struct
  let eval =
    fun t ->
    let inferred_out_dims =
      map t ~f:{ f = (fun t -> M.project t |> M.dims |> Dims.inject) } |> infer_dims_exn
    in
    let out = M.eval t in
    [%test_result: int array]
      (M.dims out)
      ~expect:inferred_out_dims
      ~message:"[Op.infer_dims] dims mismatch with actual dims";
    out
  ;;

  let neg a = eval (Unary (Neg, M.inject a))
  let sin a = eval (Unary (Sin, M.inject a))
  let cos a = eval (Unary (Cos, M.inject a))
  let sqrt a = eval (Unary (Sqrt, M.inject a))
  let exp a = eval (Unary (Exp, M.inject a))
  let log a = eval (Unary (Log, M.inject a))
  let sigmoid a = eval (Unary (Sigmoid, M.inject a))
  let add a b = eval (Binary (Add, M.inject a, M.inject b))
  let sub a b = eval (Binary (Sub, M.inject a, M.inject b))
  let mul a b = eval (Binary (Mul, M.inject a, M.inject b))
  let div a b = eval (Binary (Div, M.inject a, M.inject b))
  let matmul a b = eval (Matmul (M.inject a, M.inject b))
  let transpose a = eval (Transpose (M.inject a))

  let sum ?(dims = `All) ?(keep_dims = false) value =
    eval (Sum { value = M.inject value; dims; keep_dims })
  ;;

  let broadcast value ~dims = eval (Broadcast { value = M.inject value; dims })
  let reshape value ~dims = eval (Reshape { value = M.inject value; dims })

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

  (* TODO: subtract max to avoid overflow *)
  let softmax ~dim value =
    let exp_value = exp value in
    let sum_exp_value =
      sum exp_value ~dims:(`Just [ dim ]) ~keep_dims:true
      |> broadcast ~dims:(M.dims value)
    in
    O.(exp_value / sum_exp_value)
  ;;
end

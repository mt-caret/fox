open! Core

type 'value t =
  | Add of 'value * 'value
  | Sub of 'value * 'value
  | Mul of 'value * 'value
  | Neg of 'value
  | Sin of 'value
  | Cos of 'value
  | Matmul of 'value * 'value
  | Transpose of 'value
  | Sum of
      { value : 'value
      ; dims : int array
      ; keep_dims : bool
      }
  | Broadcast of
      { value : 'value
      ; dims : int array
      }
[@@deriving sexp_of]

let map t ~f =
  match t with
  | Add (a, b) -> Add (f a, f b)
  | Sub (a, b) -> Sub (f a, f b)
  | Mul (a, b) -> Mul (f a, f b)
  | Neg a -> Neg (f a)
  | Sin a -> Sin (f a)
  | Cos a -> Cos (f a)
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
  | Neg a -> M.neg a
  | Sin a -> M.sin a
  | Cos a -> M.cos a
  | Matmul (a, b) -> M.matmul a b
  | Transpose a -> M.transpose a
  | Sum { value; dims; keep_dims } -> M.sum value ~dims ~keep_dims
  | Broadcast { value; dims } -> M.broadcast value ~dims
;;

(* TODO: this can be written in terms of [eval]. *)
let to_string t ~f =
  match t with
  | Add (a, b) -> [%string "add %{f a} %{f b}"]
  | Sub (a, b) -> [%string "sub %{f a} %{f b}"]
  | Mul (a, b) -> [%string "mul %{f a} %{f b}"]
  | Neg a -> [%string "neg %{f a}"]
  | Sin a -> [%string "sin %{f a}"]
  | Cos a -> [%string "cos %{f a}"]
  | Matmul (a, b) -> [%string "matmul %{f a} %{f b}"]
  | Transpose a -> [%string "transpose %{f a}"]
  | Sum { value; dims; keep_dims } ->
    let dims =
      Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:", "
    in
    [%string "sum %{f value} dims=[%{dims}] keep_dims=%{keep_dims#Bool}"]
  | Broadcast { value; dims } ->
    let dims =
      Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:", "
    in
    [%string "broadcast %{f value} dims=[%{dims}]"]
;;

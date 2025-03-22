open! Core

type 'value t =
  | Add of 'value * 'value
  | Sub of 'value * 'value
  | Mul of 'value * 'value
  | Neg of 'value
  | Sin of 'value
  | Cos of 'value
[@@deriving sexp_of]

let map t ~f =
  match t with
  | Add (a, b) -> Add (f a, f b)
  | Sub (a, b) -> Sub (f a, f b)
  | Mul (a, b) -> Mul (f a, f b)
  | Neg a -> Neg (f a)
  | Sin a -> Sin (f a)
  | Cos a -> Cos (f a)
;;

let eval (type a) (module M : Operators_intf.S with type t = a) (t : a t) =
  match t with
  | Add (a, b) -> M.add a b
  | Sub (a, b) -> M.sub a b
  | Mul (a, b) -> M.mul a b
  | Neg a -> M.neg a
  | Sin a -> M.sin a
  | Cos a -> M.cos a
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
;;

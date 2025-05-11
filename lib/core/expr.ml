open! Core

module Var = struct
  module T = struct
    type t = Var of string [@@deriving compare, sexp]
  end

  include T
  include Comparable.Make_plain (T)

  let type_id = Type_equal.Id.create ~name:"Var" [%sexp_of: t]
  let to_string (Var name) = name
end

module Atom = struct
  type t =
    | Var of Var.t
    | Value of Value.t
  [@@deriving sexp_of]

  let to_string = function
    | Var name -> Var.to_string name
    | Value value -> Sexp.to_string ([%sexp_of: Value.t] value)
  ;;

  let of_value (T (x, id) as value : Value.t) : t =
    match Type_equal.Id.same_witness id Var.type_id with
    | Some T -> Var x
    | None -> Value value
  ;;
end

module Eq = struct
  type t =
    { var : Var.t
    ; op : Atom.t Op.t
    }
  [@@deriving sexp_of]

  let to_string { var; op } =
    let op_string = Op.to_string op ~f:Atom.to_string in
    [%string "%{var#Var} = %{op_string}"]
  ;;
end

type t =
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of]

let to_string_hum { parameters; equations; return_vals; out_tree_def = _ } =
  let parameters =
    String.concat ~sep:" " (List.map parameters ~f:(fun (Var name) -> name))
  in
  let equations = String.concat ~sep:"\n" (List.map equations ~f:Eq.to_string) in
  let return_vals =
    Nonempty_list.to_list return_vals
    |> List.map ~f:Atom.to_string
    |> String.concat ~sep:", "
  in
  [%string "%{parameters#String} ->\n%{equations#String}\nin ( %{return_vals} )"]
;;

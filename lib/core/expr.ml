open! Core

module Var = struct
  module T = struct
    type t =
      { name : string
      ; dims : int array
      }
    [@@deriving compare, sexp, fields ~getters]
  end

  include T
  include Comparable.Make_plain (T)

  let type_id = Type_equal.Id.create ~name:"Var" [%sexp_of: t]

  let to_string { name; dims } =
    let dims =
      Array.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:","
    in
    [%string "%{name}[%{dims}]"]
  ;;
end

module Atom = struct
  type t =
    | Var of Var.t
    | Value of Value.t
  [@@deriving sexp_of]

  let to_string = function
    | Var { name; dims = _ } -> name
    | Value value -> Sexp.to_string ([%sexp_of: Value.t] value)
  ;;

  let of_value (T { value = x; type_id; dims } as value : Value.t) : t =
    match Type_equal.Id.same_witness type_id Var.type_id with
    | Some T ->
      [%test_eq: int array] dims (Var.dims x);
      Var x
    | None -> Value value
  ;;

  let dims = function
    | Var var -> Var.dims var
    | Value value -> Value.dims value
  ;;
end

module Eq = struct
  type t =
    { var : Var.t
    ; op : Atom.t Op.t
    }
  [@@deriving sexp_of, fields ~getters]

  let to_string { var; op } =
    let op_string = Op.to_string op ~f:Atom.to_string in
    [%string "%{var#Var} = %{op_string};"]
  ;;
end

type t =
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, fields ~getters]

let to_string_hum { parameters; equations; return_vals; out_tree_def = _ } =
  let parameters = String.concat ~sep:" " (List.map parameters ~f:Var.to_string) in
  let equations = String.concat ~sep:"\n" (List.map equations ~f:Eq.to_string) in
  let return_vals =
    Nonempty_list.to_list return_vals
    |> List.map ~f:Atom.to_string
    |> String.concat ~sep:", "
  in
  [%string "%{parameters#String} ->\n%{equations#String}\n( %{return_vals} )"]
;;

let validate ({ parameters; equations; return_vals; out_tree_def = _ } as t) =
  let env = Var.Set.of_list parameters in
  let validate_atoms ~env (atoms : Atom.t list) =
    match
      List.filter_map atoms ~f:(function
        | Var var -> Some var
        | Value _ -> None)
      |> List.filter ~f:(Fn.non (Set.mem env))
    with
    | [] -> ()
    | missing_vars ->
      raise_s
        [%message
          "Undefined variable" (missing_vars : Var.t list) ~expr:(to_string_hum t)]
  in
  let env =
    List.fold equations ~init:env ~f:(fun env { var; op } ->
      Op.to_list op |> validate_atoms ~env;
      Set.add env var)
  in
  Nonempty_list.to_list return_vals |> validate_atoms ~env
;;

let create ~parameters ~equations ~return_vals ~out_tree_def =
  let t = { parameters; equations; return_vals; out_tree_def } in
  validate t;
  t
;;

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
  module T = struct
    type 'a t =
      | Var of Var.t
      | Value of 'a Value.t
    [@@deriving sexp_of]
  end

  include T
  include Higher_kinded.Make (T)

  module Packed = struct
    type 'a typed = 'a t [@@deriving sexp_of]
    type t = T : 'a typed -> t

    let sexp_of_t (T t) = [%sexp_of: _ typed] t
  end

  let to_string (type a) (t : a t) =
    match t with
    | Var { name; dims = _ } -> name
    | Value value -> Sexp.to_string ([%sexp_of: Value.Packed.t] (T value))
  ;;

  let of_value
        (type a)
        (T { value = x; type_id; dims; type_ = _ } as value : a Value.t)
        ~(vars : Var.Set.t)
    : a t
    =
    match Type_equal.Id.same_witness type_id Var.type_id with
    | Some T ->
      [%test_eq: int array] dims (Var.dims x);
      (match Set.mem vars x with
       | true -> Var x
       | false -> Value value)
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
    ; op : Atom.higher_kinded Op.Packed.t
    }
  [@@deriving fields ~getters]

  let to_string { var; op = T op } =
    let op_string = Op.to_string op ~f:(fun t -> Atom.to_string (Atom.project t)) in
    [%string "%{var#Var} = %{op_string};"]
  ;;

  let sexp_of_t { var; op } =
    [%sexp
      { var : Var.t
      ; op : Sexp.t =
          Op.Packed.sexp_of_t
            ~f:{ f = (fun t -> [%sexp_of: _ Atom.t] (Atom.project t)) }
            op
      }]
  ;;
end

type t =
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.Packed.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, fields ~getters]

let to_string_hum { parameters; equations; return_vals; out_tree_def = _ } =
  let parameters = String.concat ~sep:" " (List.map parameters ~f:Var.to_string) in
  let equations = String.concat ~sep:"\n" (List.map equations ~f:Eq.to_string) in
  let return_vals =
    Nonempty_list.to_list return_vals
    |> List.map ~f:(fun (T t) -> Atom.to_string t)
    |> String.concat ~sep:", "
  in
  [%string "%{parameters#String} ->\n%{equations#String}\n( %{return_vals} )"]
;;

let validate ({ parameters; equations; return_vals; out_tree_def = _ } as t) =
  let env = Var.Set.of_list parameters in
  let validate_atoms ~env (atoms : Atom.Packed.t list) =
    match
      List.filter_map atoms ~f:(function
        | T (Var var) -> Some var
        | T (Value _) -> None)
      |> List.filter ~f:(Fn.non (Set.mem env))
    with
    | [] -> ()
    | missing_vars ->
      raise_s
        [%message
          "Undefined variable" (missing_vars : Var.t list) ~expr:(to_string_hum t)]
  in
  let env =
    List.fold equations ~init:env ~f:(fun env { var; op = T op } ->
      Op.Non_higher_kinded.of_higher_kinded
        op
        ~f:{ f = (fun atom -> Atom.Packed.T (Atom.project atom)) }
      |> Op.Non_higher_kinded.to_list
      |> validate_atoms ~env;
      Set.add env var)
  in
  Nonempty_list.to_list return_vals |> validate_atoms ~env
;;

let create ~parameters ~equations ~return_vals ~out_tree_def =
  let t = { parameters; equations; return_vals; out_tree_def } in
  validate t;
  t
;;

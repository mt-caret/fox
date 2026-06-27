open! Core

module Var = struct
  type t =
    { name : string
    ; shape : Shape.t
    }
  [@@deriving compare, hash, sexp, fields ~getters]

  include functor Comparable.Make_plain

  let type_id = Type_equal.Id.create ~name:"Var" [%sexp_of: t]
  let dims t = shape t |> Shape.dims

  let to_string { name; shape = { dims; type_ } } =
    let dims =
      Iarray.to_list dims |> List.map ~f:Int.to_string |> String.concat ~sep:","
    in
    let type_ = [%sexp_of: Type.Packed.t] type_ |> Sexp.to_string |> String.lowercase in
    [%string "%{name}[%{dims}]: %{type_}"]
  ;;
end

module Atom = struct
  type 'v t =
    | Var of Var.t
    | Value of 'v
  [@@deriving sexp_of, compare, hash]

  let map t ~f =
    match t with
    | Var var -> Var var
    | Value value -> Value (f value)
  ;;

  let to_string t ~value_to_string =
    match t with
    | Var var -> Var.name var
    | Value value -> value_to_string value
  ;;

  let of_value
    (T { value = x; type_id; shape; id = _ } as value : Value.t)
    ~(vars : Var.Set.t)
    : Value.t t
    =
    match Type_equal.Id.same_witness type_id Var.type_id with
    | Some T ->
      [%test_eq: Shape.t] shape (Var.shape x);
      (match Set.mem vars x with
       | true -> Var x
       | false -> Value value)
    | None -> Value value
  ;;

  let shape : Value.t t -> Shape.t = function
    | Var var -> Var.shape var
    | Value value -> Value.shape value
  ;;

  let dims t = shape t |> Shape.dims
end

module Eq = struct
  type 'v t =
    { var : Var.t
    ; op : 'v Atom.t Op.t
    }
  [@@deriving sexp_of, compare, hash, fields ~getters]

  let map { var; op } ~f = { var; op = Op.map op ~f:(Atom.map ~f) }

  let to_string { var; op } ~value_to_string =
    let op_string = Op.to_string op ~f:(Atom.to_string ~value_to_string) in
    [%string "%{var#Var} = %{op_string};"]
  ;;
end

type 'a t =
  { parameters : Var.t list
  ; consts : 'a Map.M(Var).t
  ; equations : 'a Eq.t list
  ; return_vals : 'a Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, compare, hash, fields ~getters]

let map t ~f =
  { t with
    consts = Map.map t.consts ~f
  ; equations = List.map t.equations ~f:(Eq.map ~f)
  ; return_vals = Nonempty_list.map t.return_vals ~f:(Atom.map ~f)
  }
;;

let to_string_hum
  { parameters; consts; equations; return_vals; out_tree_def = _ }
  ~value_to_string
  =
  let parameters = String.concat ~sep:" " (List.map parameters ~f:Var.to_string) in
  let consts =
    match Map.is_empty consts with
    | true -> ""
    | false ->
      let consts =
        Map.to_alist consts
        |> List.map ~f:(fun (var, value) ->
          [%string "  %{var#Var} = %{value_to_string value}"])
        |> String.concat ~sep:"\n"
      in
      [%string "\nconsts:\n%{consts}"]
  in
  let equations =
    String.concat ~sep:"\n" (List.map equations ~f:(Eq.to_string ~value_to_string))
  in
  let return_vals =
    Nonempty_list.to_list return_vals
    |> List.map ~f:(Atom.to_string ~value_to_string)
    |> String.concat ~sep:", "
  in
  [%string "%{parameters#String} ->%{consts}\n%{equations#String}\n( %{return_vals} )"]
;;

let validate ({ parameters; consts; equations; return_vals; out_tree_def = _ } as t) =
  let env = Var.Set.of_list (parameters @ Map.keys consts) in
  let validate_atoms ~env (atoms : Value.t Atom.t list) =
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
          "Undefined variable"
            (missing_vars : Var.t list)
            ~expr:(to_string_hum t ~value_to_string:Value.to_string)]
  in
  let env =
    List.fold equations ~init:env ~f:(fun env { var; op } ->
      Op.to_list op |> validate_atoms ~env;
      Set.add env var)
  in
  Nonempty_list.to_list return_vals |> validate_atoms ~env
;;

let create ~parameters ~consts ~equations ~return_vals ~out_tree_def =
  let t = { parameters; consts; equations; return_vals; out_tree_def } in
  validate t;
  t
;;

open! Core

type t =
  { mutable equations : Value.t Expr.Eq.t list
  ; mutable var_name_counter : int
  ; mutable const_name_counter : int
  ; mutable vars : Expr.Var.Set.t
  ; mutable consts : Expr.Var.t Value.On_id.Map.t
  }

let create () =
  { equations = []
  ; var_name_counter = 0
  ; const_name_counter = 0
  ; vars = Expr.Var.Set.empty
  ; consts = Value.On_id.Map.empty
  }
;;

let fresh_var t ~shape =
  let name = [%string "v_%{t.var_name_counter#Int}"] in
  t.var_name_counter <- t.var_name_counter + 1;
  let var : Expr.Var.t = { name; shape } in
  t.vars <- Set.add t.vars var;
  var
;;

let intern_value t value : Value.t Expr.Atom.t =
  match Expr.Atom.of_value ~vars:t.vars value with
  | Var var -> Var var
  | Value value ->
    (match Map.find t.consts value with
     | Some const -> Var const
     | None ->
       let name = [%string "c_%{t.const_name_counter#Int}"] in
       t.const_name_counter <- t.const_name_counter + 1;
       let const : Expr.Var.t = { name; shape = Value.shape value } in
       t.consts <- Map.add_exn t.consts ~key:value ~data:const;
       t.vars <- Set.add t.vars const;
       Var const)
;;

let handle t ~f =
  Fox_effect.handle ~f ~handle:(fun op ->
    let shape = Op.map op ~f:Value.shape |> Op.infer_shape_exn in
    let binder = fresh_var t ~shape in
    t.equations <- { var = binder; op = Op.map op ~f:(intern_value t) } :: t.equations;
    Value.create ~value:binder ~type_id:Expr.Var.type_id ~shape)
;;

let equations t = List.rev t.equations

let consts_map t =
  Map.to_alist t.consts |> List.map ~f:Tuple2.swap |> Expr.Var.Map.of_alist_exn
;;

open! Core

module Partial_value = struct
  type t =
    | Known of Value.t
    | Unknown of Expr.Var.t
  [@@deriving sexp_of]

  let shape = function
    | Known value -> Value.shape value
    | Unknown var -> Expr.Var.shape var
  ;;

  let to_atom t ~vars =
    match t with
    | Known value -> Expr.Atom.of_value value ~vars
    | Unknown var -> Expr.Atom.Var var
  ;;

  let type_id = Type_equal.Id.create ~name:"Partial_value" [%sexp_of: t]
end

(* The partial-evaluation tracer. [Known] inputs are evaluated through; an op with any
   [Unknown] operand is recorded as an equation and produces a fresh [Unknown]. *)
module Tracer = struct
  type t =
    { mutable equations : Value.t Expr.Eq.t list
    ; mutable name_counter : int
    ; mutable vars : Expr.Var.Set.t
    }

  let create () = { equations = []; name_counter = 0; vars = Expr.Var.Set.empty }

  let fresh_var t ~shape =
    let name = [%string "p_%{t.name_counter#Int}"] in
    t.name_counter <- t.name_counter + 1;
    let var : Expr.Var.t = { name; shape } in
    t.vars <- Set.add t.vars var;
    var
  ;;

  let lift value =
    Value.coerce value ~type_id:Partial_value.type_id
    |> Option.value ~default:(Partial_value.Known value)
  ;;

  let value_to_atom t value = Partial_value.to_atom value ~vars:t.vars

  let handle t ~f =
    Fox_effect.handle ~f ~handle:(fun op ->
      let result : Partial_value.t =
        match Op.map op ~f:lift with
        | Unary (kind, Known a) -> Known (Op.eval (module Value) (Unary (kind, a)))
        | Binary (kind, Known a, Known b) ->
          Known (Op.eval (module Value) (Binary (kind, a, b)))
        | Matmul (Known a, Known b) -> Known (Value.matmul a b)
        | Transpose (Known a) -> Known (Value.transpose a)
        | Sum { value = Known a; dims; keep_dims } -> Known (Value.sum a ~dims ~keep_dims)
        | Broadcast { value = Known a; dims } -> Known (Value.broadcast a ~dims)
        | Reshape { value = Known a; dims } -> Known (Value.reshape a ~dims)
        | ( Unary ((Neg | Sin | Cos | Sqrt | Exp | Log | Sigmoid), _)
          | Binary ((Add | Sub | Mul | Div | Eq | Gt | Lt), _, _)
          | Matmul _ | Transpose _ | Sum _ | Broadcast _ | Reshape _ ) as op ->
          let shape = Op.map op ~f:Partial_value.shape |> Op.infer_shape_exn in
          let binder = fresh_var t ~shape in
          t.equations
          <- { var = binder; op = Op.map op ~f:(value_to_atom t) } :: t.equations;
          Unknown binder
      in
      Value.create
        ~value:result
        ~type_id:Partial_value.type_id
        ~shape:(Partial_value.shape result))
  ;;

  let equations t = List.rev t.equations
end

let partially_apply_expr_flat
  (inputs : Partial_value.t list)
  ~(f : Value.t list -> Value.t list * Value_tree.Def.t)
  : Partial_value.t list * Value.t Expr.t
  =
  let partial = Tracer.create () in
  let outputs, out_tree_def =
    Tracer.handle partial ~f:(fun () ->
      List.map inputs ~f:(fun input ->
        Value.create
          ~value:input
          ~type_id:Partial_value.type_id
          ~shape:(Partial_value.shape input))
      |> f)
  in
  let outputs = List.map outputs ~f:Tracer.lift in
  let only_unknowns =
    List.filter_map ~f:(function
      | Partial_value.Known _ -> None
      | Unknown var -> Some var)
  in
  ( outputs
  , match
      Expr.create
        ~parameters:(only_unknowns inputs)
        ~consts:Expr.Var.Map.empty
        ~equations:(Tracer.equations partial)
        ~return_vals:
          (only_unknowns outputs
           |> List.map ~f:(fun var -> Expr.Atom.Var var)
           |> Nonempty_list.of_list_exn)
        ~out_tree_def
    with
    | exception exn ->
      raise_s
        [%message "Failed to create expr" (exn : exn) (inputs : Partial_value.t list)]
    | expr -> expr )
;;

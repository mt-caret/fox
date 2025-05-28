open! Core
open! Effect
open! Effect.Deep
module Expr = Expr
module Op = Op
module Tensor = Tensor
module Treeable = Treeable
module Treeable_intf = Treeable_intf
module Value = Value
module Value_tree = Value_tree

let flatten_function
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~(f : in_ -> out)
      ~in_tree_def
      ~here
  =
  let out_tree_def = Set_once.create () in
  ( Staged.stage (fun values ->
      let in_tree = Value_tree.unflatten values ~def:in_tree_def in
      let out = f (In.t_of_tree in_tree) in
      let out_tree = Out.tree_of_t out in
      Set_once.set_exn out_tree_def here (Value_tree.to_def out_tree);
      Value_tree.flatten out_tree)
  , out_tree_def )
;;

module Eval = struct
  let handle ~f =
    try f () with
    | effect Fox_effect.Op op, k ->
      let result =
        Op.map op ~f:Value.to_tensor_exn |> Op.eval (module Tensor) |> Value.of_tensor
      in
      continue k result
  ;;
end

(* foo(x) = x(x + 3) = x^2 + 3x
   foo'(x) = 2x + 3
   foo''(x) = 2
   foo'''(x) = 0
*)
let foo x = Value.O.(x * (x + Value.of_float 3.))

let%expect_test "foo" =
  Eval.handle ~f:(fun () -> foo (Value.of_float 2.)) |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 10) |}]
;;

module Dual_number = struct
  type t =
    { primal : Value.t
    ; tangent : Value.t
    ; id : Id.t
    }
  [@@deriving sexp_of, fields ~getters]

  let type_id = Type_equal.Id.create ~name:"Dual_number" [%sexp_of: t]
  let to_value t : Value.t = T { value = t; type_id; dims = Value.dims t.primal }
end

module Jvp = struct
  type t = { id : Id.t }

  let create () = { id = Id.create () }

  let dual_number t ~primal ~tangent : Dual_number.t =
    [%test_eq: int array] (Value.dims primal) (Value.dims tangent);
    { primal; tangent; id = t.id }
  ;;

  let lift t (T { value = x; type_id; dims } as value : Value.t) : Dual_number.t =
    let zeros () = Value.of_tensor (Tensor.create ~dims 0.) in
    match Type_equal.Id.same_witness type_id Dual_number.type_id with
    | Some T ->
      if Id.equal t.id x.id then x else dual_number t ~primal:value ~tangent:(zeros ())
    | None -> dual_number t ~primal:value ~tangent:(zeros ())
  ;;

  let handle t ~f =
    try f () with
    | effect Fox_effect.Op op, k ->
      let result =
        match Op.map op ~f:(lift t) with
        | Add (a, b) ->
          dual_number
            t
            ~primal:Value.O.(a.primal + b.primal)
            ~tangent:Value.O.(a.tangent + b.tangent)
        | Sub (a, b) ->
          dual_number
            t
            ~primal:Value.O.(a.primal - b.primal)
            ~tangent:Value.O.(a.tangent - b.tangent)
        | Mul (a, b) ->
          dual_number
            t
            ~primal:Value.O.(a.primal * b.primal)
            ~tangent:Value.O.((a.tangent * b.primal) + (a.primal * b.tangent))
        | Div (a, b) ->
          dual_number
            t
            ~primal:Value.O.(a.primal / b.primal)
            ~tangent:
              Value.O.(
                ((a.tangent * b.primal) - (a.primal * b.tangent)) / (a.primal * a.primal))
        | Neg a -> dual_number t ~primal:Value.O.(-a.primal) ~tangent:Value.O.(-a.tangent)
        | Sin a ->
          dual_number
            t
            ~primal:(Value.sin a.primal)
            ~tangent:Value.O.(Value.cos a.primal * a.tangent)
        | Cos a ->
          dual_number
            t
            ~primal:(Value.cos a.primal)
            ~tangent:Value.O.(-Value.sin a.primal * a.tangent)
        | Sqrt a ->
          dual_number
            t
            ~primal:(Value.sqrt a.primal)
            ~tangent:(Value.div a.tangent (Value.scale (Value.sqrt a.primal) 2.))
        | Matmul (a, b) ->
          dual_number
            t
            ~primal:(Value.matmul a.primal b.primal)
            ~tangent:
              Value.O.(Value.matmul a.tangent b.primal + Value.matmul a.primal b.tangent)
        | Transpose a ->
          dual_number
            t
            ~primal:(Value.transpose a.primal)
            ~tangent:(Value.transpose a.tangent)
        | Sum { value; dims; keep_dims } ->
          dual_number
            t
            ~primal:(Value.sum ~dims ~keep_dims value.primal)
            ~tangent:(Value.sum ~dims ~keep_dims value.tangent)
        | Broadcast { value; dims } ->
          dual_number
            t
            ~primal:(Value.broadcast ~dims value.primal)
            ~tangent:(Value.broadcast ~dims value.tangent)
      in
      continue k (Dual_number.to_value result)
  ;;
end

let jvp
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~f
      ~(primals : in_)
      ~(tangents : in_)
  =
  let jvp = Jvp.create () in
  let primals_tree, tangents_tree = In.tree_of_t primals, In.tree_of_t tangents in
  let primals_tree_def, tangents_tree_def =
    Value_tree.to_def primals_tree, Value_tree.to_def tangents_tree
  in
  [%test_eq: Value_tree.Def.t] primals_tree_def tangents_tree_def;
  let inputs =
    List.zip_exn (Value_tree.flatten primals_tree) (Value_tree.flatten tangents_tree)
    |> List.map ~f:(fun (primal, tangent) ->
      Dual_number.to_value (Jvp.dual_number jvp ~primal ~tangent))
  in
  let f, out_tree_def =
    flatten_function
      (module In)
      (module Out)
      ~f
      ~in_tree_def:primals_tree_def
      ~here:[%here]
  in
  let f = Staged.unstage f in
  let primals, tangents =
    Jvp.handle jvp ~f:(fun () -> f inputs)
    |> List.map ~f:(Jvp.lift jvp)
    |> List.map ~f:(fun { primal; tangent; id = _ } -> primal, tangent)
    |> List.unzip
  in
  let out_tree_def = Set_once.get_exn out_tree_def [%here] in
  ( Out.t_of_tree (Value_tree.unflatten primals ~def:out_tree_def)
  , Out.t_of_tree (Value_tree.unflatten tangents ~def:out_tree_def) )
;;

let%expect_test "jvp'" =
  Eval.handle ~f:(fun () ->
    jvp
      (module Value)
      (module Value)
      ~f:foo
      ~primals:(Value.of_float 2.)
      ~tangents:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10) (Tensor 7)) |}]
;;

let jvp' ~f ~primal ~tangent =
  jvp (module Value) (module Value) ~f ~primals:primal ~tangents:tangent
;;

let%expect_test "jvp" =
  Eval.handle ~f:(fun () ->
    jvp' ~f:foo ~primal:(Value.of_float 2.) ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10) (Tensor 7)) |}];
  Eval.handle ~f:(fun () ->
    jvp'
      ~f:(fun x ->
        let _, tangent = jvp' ~f:foo ~primal:x ~tangent:(Value.of_float 1.) in
        tangent)
      ~primal:(Value.of_float 2.)
      ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 7) (Tensor 2)) |}]
;;

let derivative ~f ~x =
  let (_primal : Value.t), tangent = jvp' ~f ~primal:x ~tangent:(Value.of_float 1.) in
  tangent
;;

let rec nth_order_derivative ~n ~f ~x =
  match n with
  | 0 -> f x
  | _ -> derivative ~f:(fun x -> nth_order_derivative ~n:(n - 1) ~f ~x) ~x
;;

let%expect_test "nth_order_derivative" =
  let print ~n =
    Eval.handle ~f:(fun () -> nth_order_derivative ~n ~f:foo ~x:(Value.of_float 2.))
    |> [%sexp_of: Value.t]
    |> print_s
  in
  print ~n:0;
  [%expect {| (Tensor 10) |}];
  print ~n:1;
  [%expect {| (Tensor 7) |}];
  print ~n:2;
  [%expect {| (Tensor 2) |}];
  print ~n:3;
  [%expect {| (Tensor 0) |}];
  print ~n:4;
  [%expect {| (Tensor 0) |}]
;;

let%expect_test "pertubation confusion avoidance" =
  let f x =
    let g (_y : Value.t) = x in
    let should_be_zero = derivative ~f:g ~x:(Value.of_float 0.) in
    Value.O.(x * should_be_zero)
  in
  Eval.handle ~f:(fun () -> derivative ~f ~x:(Value.of_float 0.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 0) |}]
;;

module Staging = struct
  type t =
    { mutable equations : Expr.Eq.t list
    ; mutable name_counter : int
    }

  let create () = { equations = []; name_counter = 0 }

  let fresh_var t ~dims : Expr.Var.t =
    let name = [%string "v_%{t.name_counter#Int}"] in
    t.name_counter <- t.name_counter + 1;
    { name; dims }
  ;;

  let handle t ~f =
    try f () with
    | effect Fox_effect.Op op, k ->
      let dims = Op.map op ~f:Value.dims |> Op.infer_dims in
      let binder = fresh_var t ~dims in
      t.equations <- { var = binder; op = Op.map op ~f:Expr.Atom.of_value } :: t.equations;
      let dims = Op.map op ~f:Value.dims |> Op.infer_dims in
      continue k (T { value = binder; type_id = Expr.Var.type_id; dims })
  ;;
end

(* TODO: could [Expr.t] instead be something like [(in_, out) Expr.t], storing
   the modules internally? *)
let build_expr
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~f
      ~in_tree_def
  : Expr.t
  =
  let staging = Staging.create () in
  let parameters =
    Value_tree.Def.flatten in_tree_def
    |> List.map ~f:(fun dims -> Staging.fresh_var staging ~dims)
  in
  let f, out_tree_def =
    flatten_function (module In) (module Out) ~f ~in_tree_def ~here:[%here]
  in
  let f = Staged.unstage f in
  let result =
    Staging.handle staging ~f:(fun () ->
      List.map parameters ~f:(fun parameter ->
        Value.T
          { value = parameter
          ; type_id = Expr.Var.type_id
          ; dims = Expr.Var.dims parameter
          })
      |> f)
  in
  { parameters
  ; equations = List.rev staging.equations
  ; return_vals =
      Nonempty_list.of_list_exn result |> Nonempty_list.map ~f:Expr.Atom.of_value
  ; out_tree_def = Set_once.get_exn out_tree_def [%here]
  }
;;

let build_expr' ~f ~in_dims : Expr.t =
  build_expr (module Value) (module Value) ~f ~in_tree_def:(Value.tree_def ~dims:in_dims)
;;

let%expect_test "build_expr" =
  build_expr' ~f:foo ~in_dims:[||] |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = add v_0 (Tensor 3)
    v_2[] = mul v_0 v_1
    in ( v_2 )
    |}]
;;

let%expect_test "build_expr2" =
  build_expr' ~f:(fun _x -> Value.O.(Value.of_float 2. * Value.of_float 2.)) ~in_dims:[||]
  |> Expr.to_string_hum
  |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = mul (Tensor 2) (Tensor 2)
    in ( v_1 )
    |}]
;;

let eval_expr_flat (expr : Expr.t) (input : Value.t list) =
  let eval_atom (atom : Expr.Atom.t) ~env =
    match atom with
    | Var var -> Map.find_exn env var
    | Value value -> value
  in
  let env =
    List.fold
      expr.equations
      ~init:(List.zip_exn expr.parameters input |> Expr.Var.Map.of_alist_exn)
      ~f:(fun env eq ->
        let result = Op.map eq.op ~f:(eval_atom ~env) |> Op.eval (module Value) in
        Map.add_exn env ~key:eq.var ~data:result)
  in
  Nonempty_list.map expr.return_vals ~f:(eval_atom ~env) |> Nonempty_list.to_list
;;

let eval_expr
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      (expr : Expr.t)
      (input : in_)
  : out
  =
  In.tree_of_t input
  |> Value_tree.flatten
  |> eval_expr_flat expr
  |> Value_tree.unflatten ~def:expr.out_tree_def
  |> Out.t_of_tree
;;

let eval_expr' = eval_expr (module Value) (module Value)

let%expect_test "eval_expr" =
  Eval.handle ~f:(fun () ->
    eval_expr' (build_expr' ~f:foo ~in_dims:[||]) (Value.of_float 2.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 10) |}]
;;

let%expect_test "jvp and eval_expr" =
  Eval.handle ~f:(fun () ->
    jvp'
      ~f:(fun x -> eval_expr' (build_expr' ~f:foo ~in_dims:[||]) x)
      ~primal:(Value.of_float 2.)
      ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10) (Tensor 7)) |}]
;;

let%expect_test "nth_order_derivative build_expr" =
  let print ~n =
    build_expr' ~f:(fun x -> nth_order_derivative ~n ~f:foo ~x) ~in_dims:[||]
    |> Expr.to_string_hum
    |> print_endline
  in
  print ~n:0;
  [%expect
    {|
    v_0[] ->
    v_1[] = add v_0 (Tensor 3)
    v_2[] = mul v_0 v_1
    in ( v_2 )
    |}];
  print ~n:1;
  [%expect
    {|
    v_0[] ->
    v_1[] = add (Tensor 1) (Tensor 0)
    v_2[] = add v_0 (Tensor 3)
    v_3[] = mul v_0 v_1
    v_4[] = mul (Tensor 1) v_2
    v_5[] = add v_4 v_3
    v_6[] = mul v_0 v_2
    in ( v_5 )
    |}];
  print ~n:2;
  [%expect
    {|
    v_0[] ->
    v_1[] = add (Tensor 0) (Tensor 0)
    v_2[] = add (Tensor 1) (Tensor 0)
    v_3[] = add (Tensor 1) (Tensor 0)
    v_4[] = add v_0 (Tensor 3)
    v_5[] = mul v_0 v_1
    v_6[] = mul (Tensor 1) v_2
    v_7[] = add v_6 v_5
    v_8[] = mul v_0 v_2
    v_9[] = mul (Tensor 1) v_3
    v_10[] = mul (Tensor 0) v_4
    v_11[] = add v_10 v_9
    v_12[] = mul (Tensor 1) v_4
    v_13[] = add v_11 v_7
    v_14[] = add v_12 v_8
    v_15[] = mul v_0 v_3
    v_16[] = mul (Tensor 1) v_4
    v_17[] = add v_16 v_15
    v_18[] = mul v_0 v_4
    in ( v_13 )
    |}]
;;

module Partial_value = struct
  type t =
    | Known of Value.t
    | Unknown of Expr.Var.t
  [@@deriving sexp_of]

  let dims = function
    | Known value -> Value.dims value
    | Unknown var -> Expr.Var.dims var
  ;;

  let to_atom : t -> Expr.Atom.t = function
    | Known value -> Expr.Atom.of_value value
    | Unknown var -> Expr.Atom.Var var
  ;;

  let type_id = Type_equal.Id.create ~name:"Partial_value" [%sexp_of: t]
end

module Partial = struct
  type t =
    { mutable equations : Expr.Eq.t list
    ; mutable name_counter : int
    }

  let create () = { equations = []; name_counter = 0 }

  let fresh_var t ~dims : Expr.Var.t =
    let name = [%string "v_%{t.name_counter#Int}"] in
    t.name_counter <- t.name_counter + 1;
    { name; dims }
  ;;

  let lift (T { value = x; type_id; dims = _ } as value : Value.t) : Partial_value.t =
    match Type_equal.Id.same_witness type_id Partial_value.type_id with
    | Some T -> x
    | None -> Known value
  ;;

  let handle t ~f =
    try f () with
    | effect Fox_effect.Op op, k ->
      let result : Partial_value.t =
        match Op.map op ~f:lift with
        | Add (Known a, Known b) -> Known Value.O.(a + b)
        | Sub (Known a, Known b) -> Known Value.O.(a - b)
        | Mul (Known a, Known b) -> Known Value.O.(a * b)
        | Div (Known a, Known b) -> Known Value.O.(a / b)
        | Neg (Known a) -> Known Value.O.(-a)
        | Sin (Known a) -> Known (Value.sin a)
        | Cos (Known a) -> Known (Value.cos a)
        | Sqrt (Known a) -> Known (Value.sqrt a)
        | Matmul (Known a, Known b) -> Known (Value.matmul a b)
        | Transpose (Known a) -> Known (Value.transpose a)
        | Sum { value = Known a; dims; keep_dims } -> Known (Value.sum a ~dims ~keep_dims)
        | Broadcast { value = Known a; dims } -> Known (Value.broadcast a ~dims)
        | ( Add _
          | Sub _
          | Mul _
          | Div _
          | Neg _
          | Sin _
          | Cos _
          | Sqrt _
          | Matmul _
          | Transpose _
          | Sum _
          | Broadcast _ ) as op ->
          let dims = Op.map op ~f:Partial_value.dims |> Op.infer_dims in
          let binder = fresh_var t ~dims in
          t.equations
          <- { var = binder; op = Op.map op ~f:Partial_value.to_atom } :: t.equations;
          Unknown binder
      in
      continue
        k
        (T
           { value = result
           ; type_id = Partial_value.type_id
           ; dims = Partial_value.dims result
           })
  ;;
end

(** Do we need the const argument to prevent constants from being instantiated many many times?

    arguably we could just have an eq in the jaxpr for the constant
*)
let partially_apply_expr_flat
      (inputs : Partial_value.t list)
      ~(f : Value.t list -> Value.t list * Value_tree.Def.t)
  : Partial_value.t list * Expr.t
  =
  let partial = Partial.create () in
  let outputs, out_tree_def =
    Partial.handle partial ~f:(fun () ->
      List.map inputs ~f:(fun input ->
        Value.T
          { value = input
          ; type_id = Partial_value.type_id
          ; dims = Partial_value.dims input
          })
      |> f)
  in
  let outputs = List.map outputs ~f:Partial.lift in
  let only_unknowns =
    List.filter_map ~f:(function
      | Partial_value.Known _ -> None
      | Unknown var -> Some var)
  in
  ( outputs
  , { parameters = only_unknowns inputs
    ; equations = List.rev partial.equations
    ; return_vals =
        only_unknowns outputs
        |> List.map ~f:(fun var -> Expr.Atom.Var var)
        |> Nonempty_list.of_list_exn
    ; out_tree_def
    } )
;;

let%expect_test "partially_apply_expr_flat" =
  let partial_values, expr =
    Eval.handle ~f:(fun () ->
      partially_apply_expr_flat
        [ Known (Value.of_float 2.); Unknown { name = "x"; dims = [||] } ]
        ~f:(function
          | [ x; y ] ->
            let x2 = Value.O.(x * x) in
            ( [ x2; Value.O.((x2 * y) + Value.of_float 3.); x; Value.O.((y * y) + x2) ]
            , Value.tree_def ~dims:[||] )
          | _ -> assert false))
  in
  print_s ([%sexp_of: Partial_value.t list] partial_values);
  [%expect
    {|
    ((Known (Tensor 4)) (Unknown ((name v_3) (dims ()))) (Known (Tensor 2))
     (Unknown ((name v_1) (dims ()))))
    |}];
  Expr.to_string_hum expr |> print_endline;
  [%expect
    {|
    x[] ->
    v_0[] = mul x x
    v_1[] = add v_0 (Tensor 4)
    v_2[] = mul (Tensor 4) x
    v_3[] = add v_2 (Tensor 3)
    in ( v_3, v_1 )
    |}]
;;

let linearize
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~(f : in_ -> out)
      ~(primals : in_)
  =
  let primals_tree = In.tree_of_t primals in
  let primals_tree_def = Value_tree.to_def primals_tree in
  let primals =
    Value_tree.flatten primals_tree
    |> List.map ~f:(fun value -> Partial_value.Known value)
  in
  let primals_length = List.length primals in
  let inputs =
    List.append
      primals
      (List.mapi primals ~f:(fun i primal ->
         Partial_value.Unknown
           { name = [%string "a_%{i#Int}"]; dims = Partial_value.dims primal }))
  in
  let outputs, expr =
    partially_apply_expr_flat inputs ~f:(fun inputs ->
      let primals, tangents = List.split_n inputs primals_length in
      let out_primal, out_tangent =
        jvp
          (module In)
          (module Out)
          ~f
          ~primals:(Value_tree.unflatten primals ~def:primals_tree_def |> In.t_of_tree)
          ~tangents:(Value_tree.unflatten tangents ~def:primals_tree_def |> In.t_of_tree)
      in
      let out_primal_tree = Out.tree_of_t out_primal in
      let out_tree_def = Value_tree.to_def out_primal_tree in
      let out_tangent_tree = Out.tree_of_t out_tangent in
      ( List.append
          (Value_tree.flatten out_primal_tree)
          (Value_tree.flatten out_tangent_tree)
      , out_tree_def ))
  in
  let outputs = List.take outputs primals_length in
  let output =
    List.filter_map outputs ~f:(function
      | Partial_value.Known value -> Some value
      | Unknown _ ->
        raise_s
          [%message
            "unexpected unknown primal" (outputs : Partial_value.t list) (expr : Expr.t)])
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  let f_lin (tangents : in_) =
    In.tree_of_t tangents
    |> Value_tree.flatten
    |> eval_expr_flat expr
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  output, f_lin
;;

let linearize' ~f ~primals = linearize (module Value) (module Value) ~f ~primals

let%expect_test "linearize" =
  let y, f_lin =
    Eval.handle ~f:(fun () -> linearize' ~f:Value.sin ~primals:(Value.of_float 3.))
  in
  print_s [%message "" (y : Value.t) (Float.sin 3. : float)];
  [%expect {| ((y (Tensor 0.14112000805986721)) ("Float.sin 3." 0.14112000805986721)) |}];
  let y' = Eval.handle ~f:(fun () -> f_lin (Value.of_float 1.)) in
  print_s [%message "" (y' : Value.t) (Float.cos 3. : float)];
  [%expect
    {| ((y' (Tensor -0.98999249660044542)) ("Float.cos 3." -0.98999249660044542)) |}];
  let y, f_lin =
    Eval.handle ~f:(fun () ->
      linearize'
        ~f:(fun x ->
          let y = Value.O.(Value.sin x * Value.of_float 2.) in
          Value.O.(-y + x))
        ~primals:(Value.of_float 3.))
  in
  let y' = Eval.handle ~f:(fun () -> f_lin (Value.of_float 1.)) in
  print_s [%message "" (y : Value.t) (y' : Value.t)];
  [%expect {| ((y (Tensor 2.7177599838802657)) (y' (Tensor 2.9799849932008908))) |}];
  let f a =
    let b = Value.sin a in
    let c = Value.neg b in
    c
  in
  build_expr' ~f ~in_dims:[||] |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = sin v_0
    v_2[] = neg v_1
    in ( v_2 )
    |}];
  build_expr
    (module Value.Tuple2)
    (module Value.Tuple2)
    ~f:(fun (a, b) -> jvp' ~f ~primal:a ~tangent:b)
    ~in_tree_def:(Value.Tuple2.tree_def ~dims1:[||] ~dims2:[||])
  |> Expr.to_string_hum
  |> print_endline;
  [%expect
    {|
    v_0[] v_1[] ->
    v_2[] = cos v_0
    v_3[] = mul v_2 v_1
    v_4[] = sin v_0
    v_5[] = neg v_3
    v_6[] = neg v_4
    in ( v_6, v_5 )
    |}];
  (* TODO: fix consts? *)
  build_expr'
    ~f:(fun x ->
      let y, _f_lin = linearize' ~f ~primals:x in
      y)
    ~in_dims:[||]
  |> Expr.to_string_hum
  |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = cos v_0
    v_2[] = sin v_0
    v_3[] = neg v_2
    in ( v_3 )
    |}];
  let _y, f_lin = Eval.handle ~f:(fun () -> linearize' ~f ~primals:(Value.of_float 0.)) in
  build_expr' ~f:f_lin ~in_dims:[||] |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = mul (Tensor 1) v_0
    v_2[] = neg v_1
    in ( v_2 )
    |}]
;;

let eval_expr_transposed (expr : Expr.t) args ~cotangents =
  let accum_gradient ~ct_env var value =
    Map.update ct_env var ~f:(function
      | None -> value
      | Some existing -> Value.O.(existing + value))
  in
  let read_gradient ~ct_env var =
    (* TODO: some sort of type inference / add a new variant for "zero"? *)
    Map.find ct_env var
    |> Option.value_or_thunk ~default:(fun () ->
      Tensor.zeros ~dims:(Expr.Var.dims var) |> Value.of_tensor)
  in
  let ct_env =
    List.zip_exn (Nonempty_list.to_list expr.return_vals) cotangents
    |> List.fold ~init:Expr.Var.Map.empty ~f:(fun ct_env (return_val, cotangent) ->
      match return_val with
      | Value _ ->
        (* TODO: do we actually want to just ignore constnats? *)
        raise_s [%message "unexpected const return value" (return_val : Expr.Atom.t)]
      | Var var -> accum_gradient ~ct_env var cotangent)
  in
  let ct_env =
    List.rev expr.equations
    |> List.fold ~init:ct_env ~f:(fun ct_env { var; op } ->
      let cotangent = read_gradient ~ct_env var in
      let ct_env =
        match op with
        | Add (Var var, Value _) | Add (Value _, Var var) ->
          accum_gradient ~ct_env var cotangent
        | Add (Var v1, Var v2) ->
          let ct_env = accum_gradient ~ct_env v1 cotangent in
          accum_gradient ~ct_env v2 cotangent
        | Sub (Var var, Value _) -> accum_gradient ~ct_env var cotangent
        | Sub (Value _, Var var) -> accum_gradient ~ct_env var (Value.neg cotangent)
        | Mul (Var var, Value v) | Mul (Value v, Var var) ->
          accum_gradient ~ct_env var (Value.mul v cotangent)
        | Div (Var var, Value v) -> accum_gradient ~ct_env var (Value.div v cotangent)
        | Neg (Var var) -> accum_gradient ~ct_env var (Value.neg cotangent)
        | Sin (Var var) -> accum_gradient ~ct_env var (Value.cos cotangent)
        | Cos (Var var) -> accum_gradient ~ct_env var (Value.neg (Value.sin cotangent))
        | Matmul (Var var, Value v) ->
          accum_gradient ~ct_env var (Value.matmul cotangent (Value.transpose v))
        | Matmul (Value v, Var var) ->
          accum_gradient ~ct_env var (Value.matmul (Value.transpose v) cotangent)
        | Transpose (Var var) -> accum_gradient ~ct_env var (Value.transpose cotangent)
        | Sum { value = Var var; dims = _; keep_dims = _ } ->
          Value.broadcast cotangent ~dims:(Expr.Var.dims var)
          |> accum_gradient ~ct_env var
        | Broadcast { value = Var var; dims = to_dims } ->
          let from_dims = Expr.Var.dims var in
          let padding_length = Array.length to_dims - Array.length from_dims in
          let non_padded_broadcasts =
            Array.sub to_dims ~pos:padding_length ~len:(Array.length from_dims)
            |> Array.zip_exn from_dims
            |> Array.filter_mapi ~f:(fun i (from, to_) ->
              if from <> to_ then Some i else None)
          in
          let unpadded_cotangent =
            match padding_length with
            | 0 -> cotangent
            | _ ->
              Value.sum
                cotangent
                ~dims:(`Just (Array.init padding_length ~f:Fn.id))
                ~keep_dims:false
          in
          (match non_padded_broadcasts with
           | [||] -> unpadded_cotangent
           | _ ->
             Value.sum
               unpadded_cotangent
               ~dims:(`Just non_padded_broadcasts)
               ~keep_dims:true)
          |> accum_gradient ~ct_env var
        | Add _
        | Sub _
        | Mul _
        | Div _
        | Neg _
        | Sin _
        | Cos _
        | Sqrt _
        | Matmul _
        | Transpose _
        | Sum _
        | Broadcast _ ->
          raise_s [%message "Invalid var/val op combination" (op : Expr.Atom.t Op.t)]
      in
      ct_env)
  in
  List.map args ~f:(read_gradient ~ct_env)
;;

let vjp
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~(f : in_ -> out)
      ~(primals : in_)
  =
  let primals_tree = In.tree_of_t primals in
  let primals_tree_def = Value_tree.to_def primals_tree in
  let primals =
    Value_tree.flatten primals_tree
    |> List.map ~f:(fun value -> Partial_value.Known value)
  in
  let primals_length = List.length primals in
  let tangent_vars =
    List.mapi primals ~f:(fun i primal ->
      { Expr.Var.name = [%string "a_%{i#Int}"]; dims = Partial_value.dims primal })
  in
  let inputs =
    List.append primals (List.map tangent_vars ~f:(fun var -> Partial_value.Unknown var))
  in
  let outputs, expr =
    partially_apply_expr_flat inputs ~f:(fun inputs ->
      let primals, tangents = List.split_n inputs primals_length in
      let out_primal, out_tangent =
        jvp
          (module In)
          (module Out)
          ~f
          ~primals:(Value_tree.unflatten primals ~def:primals_tree_def |> In.t_of_tree)
          ~tangents:(Value_tree.unflatten tangents ~def:primals_tree_def |> In.t_of_tree)
      in
      let out_primal_tree = Out.tree_of_t out_primal in
      let out_tree_def = Value_tree.to_def out_primal_tree in
      let out_tangent_tree = Out.tree_of_t out_tangent in
      ( List.append
          (Value_tree.flatten out_primal_tree)
          (Value_tree.flatten out_tangent_tree)
      , out_tree_def ))
  in
  let outputs = List.take outputs primals_length in
  let output =
    List.filter_map outputs ~f:(function
      | Partial_value.Known value -> Some value
      | Unknown _ ->
        raise_s
          [%message
            "unexpected unknown primal" (outputs : Partial_value.t list) (expr : Expr.t)])
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  let f_vjp (cotangents : out) =
    eval_expr_transposed
      expr
      tangent_vars
      ~cotangents:(Out.tree_of_t cotangents |> Value_tree.flatten)
    |> Value_tree.unflatten ~def:primals_tree_def
    |> In.t_of_tree
  in
  output, f_vjp
;;

let vjp' ~f ~primal = vjp (module Value) (module Value) ~f ~primals:primal

let grad
      (type in_)
      (module In : Treeable_intf.S with type t = in_)
      ~(f : in_ -> Value.t)
      ~x
  =
  let _y, f_vjp = vjp (module In) (module Value) ~f ~primals:x in
  f_vjp (Value.of_float 1.)
;;

let grad' ~f ~x = grad (module Value) ~f ~x

let%expect_test "grad" =
  let y, f_vjp =
    Eval.handle ~f:(fun () -> vjp' ~f:Value.sin ~primal:(Value.of_float 3.))
  in
  let y' = Eval.handle ~f:(fun () -> f_vjp (Value.of_float 1.)) in
  print_s [%message "" (y : Value.t) (y' : Value.t)];
  [%expect {| ((y (Tensor 0.14112000805986721)) (y' (Tensor -0.98999249660044542))) |}];
  Eval.handle ~f:(fun () ->
    grad'
      ~f:(fun x ->
        let y = Value.O.(Value.sin x * Value.of_float 2.) in
        Value.O.(-y + x))
      ~x:(Value.of_float 3.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 2.9799849932008908) |}];
  Eval.handle ~f:(fun () ->
    grad'
      ~f:(Value.sum ~keep_dims:false)
      ~x:(Value.of_tensor (Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ])))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor ((1 1) (1 1)) (dims (2 2))) |}]
;;

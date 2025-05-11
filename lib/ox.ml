open! Core
open! Effect
open! Effect.Deep
module Expr = Expr
module Value = Value

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
    | effect Ox_effect.Op op, k ->
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
  let to_value t : Value.t = T (t, type_id)
end

module Jvp = struct
  type t = { id : Id.t }

  let create () = { id = Id.create () }
  let dual_number t ~primal ~tangent : Dual_number.t = { primal; tangent; id = t.id }

  let lift t (T (x, id) as value : Value.t) : Dual_number.t =
    match Type_equal.Id.same_witness id Dual_number.type_id with
    | Some T ->
      if Id.equal t.id x.id
      then x
      else dual_number t ~primal:value ~tangent:(Value.of_float 0.)
    | None -> dual_number t ~primal:value ~tangent:(Value.of_float 0.)
  ;;

  let handle t ~f =
    try f () with
    | effect Ox_effect.Op op, k ->
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

  let fresh_var t : Expr.Var.t =
    let name = [%string "v_%{t.name_counter#Int}"] in
    t.name_counter <- t.name_counter + 1;
    Var name
  ;;

  let handle t ~f =
    try f () with
    | effect Ox_effect.Op op, k ->
      let binder = fresh_var t in
      t.equations <- { var = binder; op = Op.map op ~f:Expr.Atom.of_value } :: t.equations;
      continue k (T (binder, Expr.Var.type_id))
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
  let arguments = Value_tree.Def.length in_tree_def in
  let staging = Staging.create () in
  let parameters = List.init arguments ~f:(fun _ -> Staging.fresh_var staging) in
  let f, out_tree_def =
    flatten_function (module In) (module Out) ~f ~in_tree_def ~here:[%here]
  in
  let f = Staged.unstage f in
  let result =
    Staging.handle staging ~f:(fun () ->
      List.map parameters ~f:(fun parameter -> Value.T (parameter, Expr.Var.type_id)) |> f)
  in
  { parameters
  ; equations = List.rev staging.equations
  ; return_vals =
      Nonempty_list.of_list_exn result |> Nonempty_list.map ~f:Expr.Atom.of_value
  ; out_tree_def
  }
;;

let build_expr' ~f : Expr.t =
  build_expr (module Value) (module Value) ~f ~in_tree_def:Value.tree_def
;;

let%expect_test "build_expr" =
  build_expr' ~f:foo |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0 ->
    v_1 = add v_0 (Tensor 3)
    v_2 = mul v_0 v_1
    in ( v_2 )
    |}]
;;

let%expect_test "build_expr2" =
  build_expr' ~f:(fun _x -> Value.O.(Value.of_float 2. * Value.of_float 2.))
  |> Expr.to_string_hum
  |> print_endline;
  [%expect
    {|
    v_0 ->
    v_1 = mul (Tensor 2) (Tensor 2)
    in ( v_1 )
    |}]
;;

let eval_expr
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      (expr : Expr.t)
      (input : in_)
  : out
  =
  let values = In.tree_of_t input |> Value_tree.flatten in
  let output =
    let eval_atom (atom : Expr.Atom.t) ~env =
      match atom with
      | Var var -> Map.find_exn env var
      | Value value -> value
    in
    let env =
      List.fold
        expr.equations
        ~init:(List.zip_exn expr.parameters values |> Expr.Var.Map.of_alist_exn)
        ~f:(fun env eq ->
          let result = Op.map eq.op ~f:(eval_atom ~env) |> Op.eval (module Value) in
          Map.add_exn env ~key:eq.var ~data:result)
    in
    Nonempty_list.map expr.return_vals ~f:(eval_atom ~env) |> Nonempty_list.to_list
  in
  let out_tree_def = Set_once.get_exn expr.out_tree_def [%here] in
  Value_tree.unflatten output ~def:out_tree_def |> Out.t_of_tree
;;

let eval_expr' = eval_expr (module Value) (module Value)

let%expect_test "eval_expr" =
  Eval.handle ~f:(fun () -> eval_expr' (build_expr' ~f:foo) (Value.of_float 2.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 10) |}]
;;

let%expect_test "jvp and eval_expr" =
  Eval.handle ~f:(fun () ->
    jvp'
      ~f:(fun x -> eval_expr' (build_expr' ~f:foo) x)
      ~primal:(Value.of_float 2.)
      ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10) (Tensor 7)) |}]
;;

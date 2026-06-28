open! Core
open! Fox_core

let foo x = Value.O.(x * (x + Value.of_float 3.))

let%expect_test "build_expr" =
  let expr = build_expr' ~f:foo ~in_dims:[::] in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 3 Float)
    v_1[]: float = add v_0 c_0;
    v_2[]: float = mul v_0 v_1;
    ( v_2 )
    |}]
;;

let%expect_test "build_expr2" =
  (* Distinct [of_float 2.] calls are distinct values, so they are not shared. *)
  let expr =
    build_expr'
      ~f:(fun _x -> Value.O.(Value.of_float 2. * Value.of_float 2.))
      ~in_dims:[::]
  in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 2 Float)
      c_1[]: float = (Tensor 2 Float)
    v_1[]: float = mul c_1 c_0;
    ( v_1 )
    |}]
;;

let%expect_test "[Expr.map ~f:(fun _ -> ())] is a value-free structural key" =
  let structure f = build_expr' ~f ~in_dims:[::] |> Expr.map ~f:(fun _ -> ()) in
  let a = structure (fun x -> Value.O.((x * Value.of_float 3.) + Value.of_float 5.)) in
  let b = structure (fun x -> Value.O.((x * Value.of_float 7.) + Value.of_float 9.)) in
  let c = structure (fun x -> Value.O.(x * x)) in
  (* The constants are erased to (), so [a] (constants 3, 5) and [b] (constants 7, 9) are
     structurally equal, while [c] is a different program. *)
  print_endline (Expr.to_string_hum a ~value_to_string:Unit.to_string);
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = ()
      c_1[]: float = ()
    v_1[]: float = mul v_0 c_0;
    v_2[]: float = add v_1 c_1;
    ( v_2 )
    |}];
  let equal = [%compare.equal: unit Expr.t] in
  let same_hash x y = [%hash: unit Expr.t] x = [%hash: unit Expr.t] y in
  print_s [%message (equal a b : bool) (same_hash a b : bool) (equal a c : bool)];
  [%expect {| (("equal a b" true) ("same_hash a b" true) ("equal a c" false)) |}]
;;

let%expect_test "a constant returned directly is hoisted into consts" =
  (* The returned constant occurs only in the return values, not in any equation, so
     [build_expr] must resolve the return values (which is what hoists the constant)
     before snapshotting [consts] - otherwise [c_0] would be referenced without appearing
     in the consts map. *)
  let expr = build_expr' ~f:(fun _x -> Value.of_float 5.) ~in_dims:[::] in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 5 Float)

    ( c_0 )
    |}]
;;

let%expect_test "shared constants are hoisted and deduplicated" =
  (* [three] is a single value reused in two ops; it is hoisted into one shared const var
     [c_0] rather than being embedded inline at each use. *)
  let three = Value.of_float 3. in
  let expr =
    build_expr' ~f:(fun x -> Value.O.((x + three) * (x + three))) ~in_dims:[::]
  in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 3 Float)
    v_1[]: float = add v_0 c_0;
    v_2[]: float = add v_0 c_0;
    v_3[]: float = mul v_2 v_1;
    ( v_3 )
    |}]
;;

let%expect_test "nth_order_derivative build_expr" =
  let print ~n =
    let expr =
      build_expr' ~f:(fun x -> nth_order_derivative ~n ~f:foo ~x) ~in_dims:[::]
    in
    Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline
  in
  print ~n:0;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 3 Float)
    v_1[]: float = add v_0 c_0;
    v_2[]: float = mul v_0 v_1;
    ( v_2 )
    |}];
  print ~n:1;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 0 Float)
      c_1[]: float = (Tensor 1 Float)
      c_2[]: float = (Tensor 3 Float)
    v_1[]: float = add c_1 c_0;
    v_2[]: float = add v_0 c_2;
    v_3[]: float = mul v_0 v_1;
    v_4[]: float = mul c_1 v_2;
    v_5[]: float = add v_4 v_3;
    v_6[]: float = mul v_0 v_2;
    ( v_5 )
    |}];
  print ~n:2;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 0 Float)
      c_1[]: float = (Tensor 0 Float)
      c_2[]: float = (Tensor 0 Float)
      c_3[]: float = (Tensor 1 Float)
      c_4[]: float = (Tensor 0 Float)
      c_5[]: float = (Tensor 1 Float)
      c_6[]: float = (Tensor 3 Float)
      c_7[]: float = (Tensor 0 Float)
    v_1[]: float = add c_1 c_0;
    v_2[]: float = add c_3 c_2;
    v_3[]: float = add c_5 c_4;
    v_4[]: float = add v_0 c_6;
    v_5[]: float = mul v_0 v_1;
    v_6[]: float = mul c_5 v_2;
    v_7[]: float = add v_6 v_5;
    v_8[]: float = mul v_0 v_2;
    v_9[]: float = mul c_3 v_3;
    v_10[]: float = mul c_7 v_4;
    v_11[]: float = add v_10 v_9;
    v_12[]: float = mul c_3 v_4;
    v_13[]: float = add v_11 v_7;
    v_14[]: float = add v_12 v_8;
    v_15[]: float = mul v_0 v_3;
    v_16[]: float = mul c_5 v_4;
    v_17[]: float = add v_16 v_15;
    v_18[]: float = mul v_0 v_4;
    ( v_13 )
    |}]
;;

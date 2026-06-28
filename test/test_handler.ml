open! Core
open! Fox_core

let foo x = Value.O.(x * (x + Value.of_float 3.))

let%expect_test "eval_expr" =
  eval ~f:(fun () ->
    let expr = build_expr' ~f:foo ~in_dims:[::] in
    eval_expr' expr (Value.of_float 2.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 10 Float) |}]
;;

let%expect_test "jvp and eval_expr" =
  eval ~f:(fun () ->
    jvp'
      ~f:(fun x ->
        let expr = build_expr' ~f:foo ~in_dims:[::] in
        eval_expr' expr x)
      ~primal:(Value.of_float 2.)
      ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10 Float) (Tensor 7 Float)) |}]
;;

let%expect_test "linearize" =
  let y, f_lin =
    eval ~f:(fun () -> linearize' ~f:Value.sin ~primals:(Value.of_float 3.))
  in
  print_s [%message "" (y : Value.t) (Float.sin 3. : float)];
  [%expect
    {| ((y (Tensor 0.14112000805986721 Float)) ("Float.sin 3." 0.14112000805986721)) |}];
  let y' = eval ~f:(fun () -> f_lin (Value.of_float 1.)) in
  print_s [%message "" (y' : Value.t) (Float.cos 3. : float)];
  [%expect
    {|
    ((y' (Tensor -0.98999249660044542 Float))
     ("Float.cos 3." -0.98999249660044542))
    |}];
  let y, f_lin =
    eval ~f:(fun () ->
      linearize'
        ~f:(fun x ->
          let y = Value.O.(Value.sin x * Value.of_float 2.) in
          Value.O.(-y + x))
        ~primals:(Value.of_float 3.))
  in
  let y' = eval ~f:(fun () -> f_lin (Value.of_float 1.)) in
  print_s [%message "" (y : Value.t) (y' : Value.t)];
  [%expect
    {|
    ((y (Tensor 2.7177599838802657 Float))
     (y' (Tensor 2.9799849932008908 Float)))
    |}];
  let f a =
    let b = Value.sin a in
    let c = Value.neg b in
    c
  in
  let expr = build_expr' ~f ~in_dims:[::] in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    v_1[]: float = sin v_0;
    v_2[]: float = neg v_1;
    ( v_2 )
    |}];
  let expr =
    build_expr
      (module Value.Tuple2)
      (module Value.Tuple2)
      ~f:(fun (a, b) -> jvp' ~f ~primal:a ~tangent:b)
      ~in_tree_def:(Value.Tuple2.tree_def ~dims1:[::] ~dims2:[::])
  in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float v_1[]: float ->
    v_2[]: float = cos v_0;
    v_3[]: float = mul v_2 v_1;
    v_4[]: float = sin v_0;
    v_5[]: float = neg v_3;
    v_6[]: float = neg v_4;
    ( v_6, v_5 )
    |}];
  let expr =
    build_expr'
      ~f:(fun x ->
        let y, _f_lin = linearize' ~f ~primals:x in
        y)
      ~in_dims:[::]
  in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    v_1[]: float = cos v_0;
    v_2[]: float = sin v_0;
    v_3[]: float = neg v_2;
    ( v_3 )
    |}];
  let _y, f_lin = eval ~f:(fun () -> linearize' ~f ~primals:(Value.of_float 0.)) in
  let expr = build_expr' ~f:f_lin ~in_dims:[::] in
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    v_0[]: float ->
    consts:
      c_0[]: float = (Tensor 1 Float)
    v_1[]: float = mul c_0 v_0;
    v_2[]: float = neg v_1;
    ( v_2 )
    |}]
;;

let%expect_test "grad" =
  let y, f_vjp = eval ~f:(fun () -> vjp' ~f:Value.sin ~primal:(Value.of_float 3.)) in
  let y' = eval ~f:(fun () -> f_vjp (Value.of_float 1.)) in
  print_s [%message "" (y : Value.t) (y' : Value.t)];
  [%expect
    {|
    ((y (Tensor 0.14112000805986721 Float))
     (y' (Tensor -0.98999249660044542 Float)))
    |}];
  eval ~f:(fun () -> grad' ~f:(fun x -> Value.O.(x * x)) ~x:(Value.of_float 3.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 6 Float) |}];
  eval ~f:(fun () ->
    grad'
      ~f:(fun x ->
        let y = Value.O.(Value.sin x * Value.of_float 2.) in
        Value.O.(-y + x))
      ~x:(Value.of_float 3.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 2.9799849932008908 Float) |}];
  eval ~f:(fun () ->
    grad'
      ~f:(Value.sum ~keep_dims:false)
      ~x:(Value.of_tensor (Tensor.of_list2_exn Float [ [ 1.; 2. ]; [ 3.; 4. ] ])))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor ((1 1) (1 1)) (dims (2 2)) (type_ Float)) |}];
  eval ~f:(fun () ->
    grad'
      ~f:(fun x ->
        Value.broadcast x ~dims:[: 3; 4 :] |> Value.sum ~dims:(`Just [ 1 ]) |> Value.mean)
      ~x:(Value.of_typed_tensor (Tensor.Typed.arange 4)))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor (1 1 1 1) (dims (4)) (type_ Float)) |}]
;;

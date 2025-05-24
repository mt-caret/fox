open! Core
open! Fox_core

let f x =
  let y = Value.O.(Value.sin x * Value.of_float 2.) in
  let z = Value.O.(-y + x) in
  z
;;

let%expect_test "eval" =
  Eval.handle ~f:(fun () -> f (Value.of_float 3.)) |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 2.7177599838802657) |}]
;;

let%expect_test "jvp" =
  Eval.handle ~f:(fun () ->
    jvp' ~f:Value.sin ~primal:(Value.of_float 3.) ~tangent:(Value.of_float 1.)
    |> snd
    |> [%sexp_of: Value.t]
    |> print_s;
    Value.cos (Value.of_float 3.) |> [%sexp_of: Value.t] |> print_s);
  [%expect
    {|
    (Tensor -0.98999249660044542)
    (Tensor -0.98999249660044542)
    |}];
  Eval.handle ~f:(fun () ->
    jvp' ~f ~primal:(Value.of_float 3.) ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 2.7177599838802657) (Tensor 2.9799849932008908)) |}];
  Eval.handle ~f:(fun () ->
    let deriv ~n =
      nth_order_derivative ~n ~f:Value.sin ~x:(Value.of_float 3.)
      |> [%sexp_of: Value.t]
      |> print_s
    in
    deriv ~n:1;
    deriv ~n:2;
    deriv ~n:3;
    deriv ~n:4);
  [%expect
    {|
    (Tensor -0.98999249660044542)
    (Tensor -0.14112000805986721)
    (Tensor 0.98999249660044542)
    (Tensor 0.14112000805986721)
    |}]
;;

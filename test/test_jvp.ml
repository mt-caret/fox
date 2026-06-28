open! Core
open! Fox_core

(* - foo(x) = x(x + 3) = x^2 + 3x
   - foo'(x) = 2x + 3
   - foo''(x) = 2
   - foo'''(x) = 0
*)
let foo x = Value.O.(x * (x + Value.of_float 3.))

let%expect_test "foo" =
  eval ~f:(fun () -> foo (Value.of_float 2.)) |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 10 Float) |}]
;;

let%expect_test "jvp'" =
  eval ~f:(fun () ->
    jvp
      (module Value)
      (module Value)
      ~f:foo
      ~primals:(Value.of_float 2.)
      ~tangents:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10 Float) (Tensor 7 Float)) |}]
;;

let%expect_test "jvp" =
  eval ~f:(fun () -> jvp' ~f:foo ~primal:(Value.of_float 2.) ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 10 Float) (Tensor 7 Float)) |}];
  eval ~f:(fun () ->
    jvp'
      ~f:(fun x ->
        let _, tangent = jvp' ~f:foo ~primal:x ~tangent:(Value.of_float 1.) in
        tangent)
      ~primal:(Value.of_float 2.)
      ~tangent:(Value.of_float 1.))
  |> [%sexp_of: Value.t * Value.t]
  |> print_s;
  [%expect {| ((Tensor 7 Float) (Tensor 2 Float)) |}]
;;

let%expect_test "nth_order_derivative" =
  let print ~n =
    eval ~f:(fun () -> nth_order_derivative ~n ~f:foo ~x:(Value.of_float 2.))
    |> [%sexp_of: Value.t]
    |> print_s
  in
  print ~n:0;
  [%expect {| (Tensor 10 Float) |}];
  print ~n:1;
  [%expect {| (Tensor 7 Float) |}];
  print ~n:2;
  [%expect {| (Tensor 2 Float) |}];
  print ~n:3;
  [%expect {| (Tensor 0 Float) |}];
  print ~n:4;
  [%expect {| (Tensor 0 Float) |}]
;;

let%expect_test "pertubation confusion avoidance" =
  let f x =
    let g (_y : Value.t) = x in
    let should_be_zero = derivative ~f:g ~x:(Value.of_float 0.) in
    Value.O.(x * should_be_zero)
  in
  eval ~f:(fun () -> derivative ~f ~x:(Value.of_float 0.))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect {| (Tensor 0 Float) |}]
;;

open! Core
open! Fox_core
open For_testing

let%expect_test "partially_apply_expr_flat" =
  let partial_values, expr =
    eval ~f:(fun () ->
      partially_apply_expr_flat
        [ Known (Value.of_float 2.)
        ; Unknown { name = "x"; shape = { dims = [::]; type_ = T Float } }
        ]
        ~f:(function
          | [ x; y ] ->
            let x2 = Value.O.(x * x) in
            ( [ x2; Value.O.((x2 * y) + Value.of_float 3.); x; Value.O.((y * y) + x2) ]
            , Value.tree_def ~dims:[::] )
          | _ -> assert false))
  in
  print_s ([%sexp_of: Partial_value.t list] partial_values);
  [%expect
    {|
    ((Known (Tensor 4 Float))
     (Unknown ((name p_3) (shape ((dims ()) (type_ Float)))))
     (Known (Tensor 2 Float))
     (Unknown ((name p_1) (shape ((dims ()) (type_ Float))))))
    |}];
  Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
  [%expect
    {|
    x[]: float ->
    p_0[]: float = mul x x;
    p_1[]: float = add p_0 (Tensor 4 Float);
    p_2[]: float = mul (Tensor 4 Float) x;
    p_3[]: float = add p_2 (Tensor 3 Float);
    ( p_3, p_1 )
    |}]
;;

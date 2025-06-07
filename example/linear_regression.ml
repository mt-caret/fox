open! Core
open! Fox_core

let%expect_test "linear regression" =
  let num_features = 2 in
  let num_examples = 10 in
  let num_iters = 100 in
  let learning_rate = 0.01 in
  let rng = Splittable_random.of_int 0 in
  let x = Tensor.normal ~dims:[| num_examples; num_features |] ~rng () in
  let true_params = Tensor.normal ~dims:[| num_features |] ~rng () in
  let y =
    Tensor.O.(
      Tensor.matmul x true_params
      + Tensor.normal ~dims:[| num_examples |] ~std:1e-2 ~rng ())
  in
  let loss weights =
    let open Value.O in
    let error = Value.of_tensor y - Value.matmul (Value.of_tensor x) weights in
    Value.mean (error * error)
  in
  let weights = ref (Tensor.normal ~dims:[| num_features |] ~rng ()) in
  let loss_grad x = grad' ~f:loss ~x in
  build_expr' ~f:loss ~in_dims:[| num_features |] |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0[2] ->
    v_1[10] = matmul (Tensor((0.39995642633665462 1.1602368073797789)(1.1461698444484156 -0.27508260258159217)(-0.79429206930935448 -0.79691620100672833)(0.37183586790086887 0.08252530331332697)(-0.640823054843121 -0.972395618004264)(-0.24531465496199806 0.32943595058180369)(-0.49745709501288782 -0.5186552221834726)(-0.49202679675483579 -0.30011198307839476)(-2.2219543337312779 0.66654181134893775)(0.458720357930648 0.0010728286252209105))(dims(10 2))) v_0;
    v_2[10] = sub (Tensor(-0.0064666378598043162 1.9665677929469698 -0.82202018980048819 0.55321043273971726 -0.47887754772981445 -0.59419529415798567 -0.479101132934855 -0.60615974619803537 -3.888855769581165 0.72374784193750175)(dims(10))) v_1;
    v_3[10] = mul v_2 v_2;
    v_4[] = sum v_3 dims=all keep_dims=false;
    v_5[] = broadcast (Tensor 10) dims=[];
    v_6[] = div v_4 v_5;
    ( v_6 )
    |}];
  let _y, f_jvp =
    Eval.handle ~f:(fun () -> vjp' ~f:loss ~primal:(Value.of_tensor !weights))
  in
  build_expr' ~f:f_jvp ~in_dims:[||] |> Expr.to_string_hum |> print_endline;
  [%expect
    {|
    v_0[] ->
    v_1[] = div v_0 (Tensor 100);
    v_2[] = mul (Tensor 10) v_1;
    v_3[10] = broadcast v_2 dims=[10];
    v_4[10] = mul (Tensor(-0.22987507367862317 4.3463761730905182 -1.6427703619487461 1.1811642145933288 -0.86466781807436588 -1.3305171979260781 -0.976724348558953 -1.2769362756602236 -8.614129411878265 1.5829382308162989)(dims(10))) v_3;
    v_5[10] = mul (Tensor(-0.22987507367862317 4.3463761730905182 -1.6427703619487461 1.1811642145933288 -0.86466781807436588 -1.3305171979260781 -0.976724348558953 -1.2769362756602236 -8.614129411878265 1.5829382308162989)(dims(10))) v_3;
    v_6[10] = add v_4 v_5;
    v_7[10] = neg v_6;
    v_8[2,10] = transpose (Tensor((0.39995642633665462 1.1602368073797789)(1.1461698444484156 -0.27508260258159217)(-0.79429206930935448 -0.79691620100672833)(0.37183586790086887 0.08252530331332697)(-0.640823054843121 -0.972395618004264)(-0.24531465496199806 0.32943595058180369)(-0.49745709501288782 -0.5186552221834726)(-0.49202679675483579 -0.30011198307839476)(-2.2219543337312779 0.66654181134893775)(0.458720357930648 0.0010728286252209105))(dims(10 2)));
    v_9[2] = matmul v_8 v_7;
    ( v_9 )
    |}];
  for i = 0 to num_iters do
    let grads =
      Eval.handle ~f:(fun () -> loss_grad (Value.of_tensor !weights))
      |> Value.to_tensor_exn
    in
    weights := Tensor.sub !weights (Tensor.scale grads learning_rate);
    let loss =
      Eval.handle ~f:(fun () -> loss (Value.of_tensor !weights)) |> Value.to_float_exn
    in
    if i mod 10 = 0 then print_s [%message "" (i : int) (loss : float)]
  done;
  [%expect
    {|
    ((i 0) (loss 10.154729604578765))
    ((i 10) (loss 7.38631852925343))
    ((i 20) (loss 5.3906108447873917))
    ((i 30) (loss 3.9491952116052551))
    ((i 40) (loss 2.905804008806244))
    ((i 50) (loss 2.1485718499399744))
    ((i 60) (loss 1.5973678981829491))
    ((i 70) (loss 1.1947496743966037))
    ((i 80) (loss 0.89950194008054518))
    ((i 90) (loss 0.68202100327261561))
    ((i 100) (loss 0.52101656302408539))
    |}]
;;

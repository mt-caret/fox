open! Core
open! Fox_core
open! Base_quickcheck

module Dims = struct
  module T = struct
    type t = int array [@@deriving quickcheck, sexp_of, compare]
  end

  include T
  include Comparable.Make_plain (T)

  let quickcheck_generator =
    Generator.fixed_point (fun quickcheck_generator ->
      let open Generator.Let_syntax in
      let%bind total_elements = Generator.size in
      match total_elements with
      | 0 -> return [||]
      | _ ->
        let%bind this_dim = Generator.int_inclusive 1 total_elements in
        (match total_elements / this_dim with
         | 0 | 1 -> return [| this_dim |]
         | remaining_size ->
           let%map dims = Generator.with_size quickcheck_generator ~size:remaining_size in
           Array.append dims [| this_dim |]))
  ;;
end

module Tensor = struct
  include Tensor

  let quickcheck_generator_with_dims ~dims =
    let open Generator.Let_syntax in
    let total_elements = Array.fold dims ~init:1 ~f:( * ) in
    let%map values =
      List.init total_elements ~f:(fun _ -> Generator.float) |> Generator.all
    in
    Tensor.of_list values |> Tensor.reshape ~dims
  ;;

  let quickcheck_generator =
    let open Generator.Let_syntax in
    let%bind dims = Dims.quickcheck_generator in
    quickcheck_generator_with_dims ~dims
  ;;

  let tensor_equal tensor1 tensor2 =
    [%test_eq: int array] (Tensor.dims tensor1) (Tensor.dims tensor2);
    let is_equal = ref true in
    Tensor.iteri tensor1 ~f:(fun index value1 ->
      let value2 = Tensor.get tensor2 index in
      is_equal
      := !is_equal
         && (Float.robustly_compare value1 value2 = 0
             || (Float.is_nan value1 && Float.is_nan value2)));
    !is_equal
  ;;

  let quickcheck_shrinker =
    Shrinker.create (fun tensor ->
      let dims = Tensor.dims tensor in
      let sliced_tensors =
        if Array.length dims > 0
        then
          Sequence.init dims.(0) ~f:(fun i ->
            Bigarray.Genarray.slice_left (Tensor.Private.to_bigarray tensor) [| i |]
            |> Tensor.Private.of_bigarray)
        else Sequence.empty
      in
      let smaller_tensors =
        Sequence.unfold ~init:tensor ~f:(fun tensor ->
          let tensor' = Tensor.map tensor ~f:(fun x -> x /. 2.) in
          if tensor_equal tensor tensor' then None else Some (tensor', tensor'))
      in
      Sequence.interleave (Sequence.of_list [ sliced_tensors; smaller_tensors ]))
  ;;

  let quickcheck_observer =
    Observer.of_hash_fold (fun hash_state tensor ->
      let hash_state = ref hash_state in
      Tensor.iter tensor ~f:(fun value ->
        hash_state := Float.hash_fold_t !hash_state value);
      !hash_state)
  ;;
end

module Op = struct
  module Unary = struct
    type t = Op.Unary.t =
      | Neg
      | Sin
      | Cos
      | Sqrt
    [@@deriving quickcheck]
  end

  module Binary = struct
    type t = Op.Binary.t =
      | Add
      | Sub
      | Mul
      | Div
    [@@deriving quickcheck]
  end

  type 'value t = 'value Op.t =
    | Unary of Unary.t * 'value
    | Binary of Binary.t * 'value * 'value
    | Matmul of 'value * 'value
    | Transpose of 'value
    | Sum of
        { value : 'value
        ; dims : [ `Just of int Nonempty_list.t | `All ]
        ; keep_dims : bool
        }
    | Broadcast of
        { value : 'value
        ; dims : Dims.t
        }
  [@@deriving quickcheck]
end

let choose elements =
  assert (not (List.is_empty elements));
  let open Generator.Let_syntax in
  let length = List.length elements in
  let%map index = Generator.int_inclusive 0 (length - 1) in
  List.nth_exn elements index
;;

let generator_filter_bind generator ~f =
  let rec loop ~size ~random =
    let x = Generator.generate generator ~size ~random in
    match Generator.generate (f x) ~size ~random with
    | Some y -> y
    | None -> loop ~size:(size + 1) ~random
  in
  Generator.create loop
;;

let op_generator ~values_by_dims =
  let open Generator.Let_syntax in
  Op.quickcheck_generator Generator.unit
  |> generator_filter_bind ~f:(function
    | Unary (kind, ()) ->
      let%map value =
        choose (Map.keys values_by_dims) >>| Map.find_exn values_by_dims >>= choose
      in
      Some (Op.Unary (kind, value))
    | Binary (kind, (), ()) ->
      let%bind values =
        choose (Map.keys values_by_dims) >>| Map.find_exn values_by_dims
      in
      let%map value1 = choose values
      and value2 = choose values in
      Some (Op.Binary (kind, value1, value2))
    | Matmul ((), ()) ->
      (match
         let%bind.Option all_dims =
           match
             Map.keys values_by_dims |> List.filter ~f:(fun dims -> Array.length dims = 2)
           with
           | [] -> None
           | all_dims -> Some all_dims
         in
         match
           List.concat_map all_dims ~f:(fun lhs_dims ->
             List.filter_map all_dims ~f:(fun rhs_dims ->
               match lhs_dims, rhs_dims with
               | [| _; m |], [| m' |] | [| _; m |], [| m'; _ |] ->
                 Option.some_if (m = m') (lhs_dims, rhs_dims)
               | _ -> None))
         with
         | [] -> None
         | dim_pairs -> Some dim_pairs
       with
       | None -> return None
       | Some dim_pairs ->
         let%bind lhs_dims, rhs_dims = choose dim_pairs in
         let%map lhs = Map.find_exn values_by_dims lhs_dims |> choose
         and rhs = Map.find_exn values_by_dims rhs_dims |> choose in
         Some (Op.Matmul (lhs, rhs)))
    | Transpose () ->
      (match
         Map.keys values_by_dims |> List.filter ~f:(fun dims -> Array.length dims = 2)
       with
       | [] -> return None
       | all_dims ->
         let%map value = choose all_dims >>| Map.find_exn values_by_dims >>= choose in
         Some (Op.Transpose value))
    | Sum { value = (); dims; keep_dims = _ } ->
      (match dims with
       | `All ->
         let%map value =
           choose (Map.keys values_by_dims) >>| Map.find_exn values_by_dims >>= choose
         in
         Some (Op.Sum { value; dims = `All; keep_dims = true })
       | `Just dims ->
         let max_index =
           Nonempty_list.map dims ~f:(fun dim -> if dim < 0 then -dim - 1 else dim)
           |> Nonempty_list.max_elt' ~compare:Int.compare
         in
         (match
            Map.keys values_by_dims
            |> List.filter ~f:(fun dims -> Array.length dims > max_index)
          with
          | [] -> return None
          | all_dims ->
            let%map value = choose all_dims >>| Map.find_exn values_by_dims >>= choose in
            Some (Op.Sum { value; dims = `Just dims; keep_dims = true })))
    | Broadcast { value = (); dims } ->
      (match
         Map.keys values_by_dims
         |> List.filter ~f:(fun input_dims ->
           Fox_core.Op.infer_dims (Broadcast { value = input_dims; dims })
           |> Or_error.is_ok)
       with
       | [] -> return None
       | all_dims ->
         let%map value = choose all_dims >>| Map.find_exn values_by_dims >>= choose in
         Some (Op.Broadcast { value; dims })))
;;

let expr_generator ~op_nums =
  let open Generator.Let_syntax in
  let%bind dims = Dims.quickcheck_generator in
  let arg : Expr.Var.t = { name = "arg"; dims } in
  let%map _values_by_dims, equations =
    List.range 0 op_nums
    |> List.fold
         ~init:(return (Dims.Map.singleton dims [ Expr.Atom.Var arg ], []))
         ~f:(fun accum i ->
           let%bind values_by_dims, equations = accum in
           let%map op = op_generator ~values_by_dims in
           let dims =
             Fox_core.Op.map op ~f:Expr.Atom.dims |> Fox_core.Op.infer_dims_exn
           in
           let var : Expr.Var.t = { name = [%string "v_%{i#Int}"]; dims } in
           ( Map.add_multi values_by_dims ~key:dims ~data:(Expr.Atom.Var var)
           , { Expr.Eq.var; op } :: equations ))
  in
  let out = List.hd_exn equations |> Expr.Eq.var in
  Expr.create
    ~parameters:[ arg ]
    ~equations:(List.rev equations)
    ~return_vals:[ Expr.Atom.Var out ]
    ~out_tree_def:(Value_tree.Def.leaf ~dims:out.dims)
;;

let%expect_test "expr_generator" =
  let random = Splittable_random.of_int 0 in
  for i = 1 to 5 do
    let expr = Generator.generate (expr_generator ~op_nums:i) ~size:6 ~random in
    Expr.to_string_hum expr |> print_endline;
    print_endline "--------------------------------"
  done;
  [%expect
    {|
    arg[4] ->
    v_0[4] = broadcast arg dims=[4];
    ( v_0 )
    --------------------------------
    arg[2,2] ->
    v_0[2,2] = add arg arg;
    v_1[2,2] = sin v_0;
    ( v_1 )
    --------------------------------
    arg[5] ->
    v_0[5] = sin arg;
    v_1[5] = sub v_0 v_0;
    v_2[5] = div arg arg;
    ( v_2 )
    --------------------------------
    arg[5] ->
    v_0[5] = mul arg arg;
    v_1[5] = div v_0 arg;
    v_2[1] = sum v_1 dims=all keep_dims=true;
    v_3[2,1,1,3,1] = broadcast v_2 dims=[2, 1, 1, 3, 1];
    ( v_3 )
    --------------------------------
    arg[4] ->
    v_0[4] = div arg arg;
    v_1[4] = cos arg;
    v_2[4] = neg v_0;
    v_3[1] = sum arg dims=all keep_dims=true;
    v_4[4] = sqrt v_0;
    ( v_4 )
    --------------------------------
    |}]
;;

let fun_generator ~op_nums =
  let open Generator.Let_syntax in
  let%bind expr = expr_generator ~op_nums in
  match Expr.parameters expr with
  | [ var ] ->
    let%map value =
      Tensor.quickcheck_generator_with_dims ~dims:var.dims >>| Value.of_tensor
    in
    value, expr
  | _ -> failwith "fun_generator: expected 1 parameter"
;;

let%expect_test "eval expr vs xla" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  Quickcheck.test
    (fun_generator ~op_nums:1)
    ~trials:300
    ~sexp_of:(fun (value, expr) ->
      [%sexp { value : Value.t; expr : string = Expr.to_string_hum expr }])
    ~f:(fun (value, expr) ->
      let f value = eval_expr' expr value in
      let eval_result = Eval.handle ~f:(fun () -> f value) in
      let xla_result = Fox_jit.jit' ~f ~x:value in
      assert (
        Tensor.tensor_equal
          (Value.to_tensor_exn eval_result)
          (Value.to_tensor_exn xla_result)))
;;

let%expect_test "grad+jit vs grad+eval" =
  let f value = grad' ~f:(fun value -> Value.O.(value * value)) ~x:value in
  Eval.handle ~f:(fun () -> f (Value.of_float 1.)) |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 2) |}];
  Fox_jit.jit' ~f ~x:(Value.of_float 1.) |> [%sexp_of: Value.t] |> print_s;
  String.substr_replace_all [%expect.output] ~pattern:"\\n" ~with_:" " |> print_endline;
  [%expect {| (Tensor 2) |}]
;;

let%expect_test "eval grad expr vs xla" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  Expect_test_helpers_core.require_does_raise [%here] (fun () ->
    Quickcheck.test
      (fun_generator ~op_nums:1)
      ~trials:300
      ~sexp_of:(fun (value, expr) ->
        [%sexp { value : Value.t; expr : string = Expr.to_string_hum expr }])
      ~f:(fun (value, expr) ->
        let f value =
          grad' ~f:(fun value -> eval_expr' expr value |> Value.sum) ~x:value
        in
        let eval_result = Eval.handle ~f:(fun () -> f value) in
        let xla_result = Fox_jit.jit' ~f ~x:value in
        assert (
          Tensor.tensor_equal
            (Value.to_tensor_exn eval_result)
            (Value.to_tensor_exn xla_result))));
  String.substr_replace_all [%expect.output] ~pattern:"\\n" ~with_:" " |> print_endline;
  [%expect
    {|
    ("Base_quickcheck.Test.run: test failed"
      (input (
        (value (
          Tensor
          (-2.2765450197646858E-274
           126.14432978630066
           2.716326520622772E+60
           1.852679443830472E-17
           -0.38238309169044982
           3.7564530697704668E+60
           -1)
          (dims (7))))
        (expr "arg[7] -> v_0[7] = div arg arg; ( v_0 )")))
      (error (
        "Invalid var/val op combination"
        (op (
          Binary Sub (Var ((name p_1) (dims (7)))) (Var ((name p_0) (dims (7))))))
        (expr
         "a_0[7] -> p_0[7] = mul (Tensor(-2.2765450197646858E-274 126.14432978630066 2.716326520622772E+60 1.852679443830472E-17 -0.38238309169044982 3.7564530697704668E+60 -1)(dims(7))) a_0; p_1[7] = mul a_0 (Tensor(-2.2765450197646858E-274 126.14432978630066 2.716326520622772E+60 1.852679443830472E-17 -0.38238309169044982 3.7564530697704668E+60 -1)(dims(7))); p_2[7] = sub p_1 p_0; p_3[7] = div p_2 (Tensor(0 15912.391937234979 7.3784297666386151E+120 3.4324211215919869E-34 0.14621682881074696 1.4110939665387963E+121 1)(dims(7))); p_4[] = sum p_3 dims=all keep_dims=false; ( p_4 )"))))
    |}]
;;

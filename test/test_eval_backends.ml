open! Core
open! Fox_core
open! Base_quickcheck

module Dims = struct
  type t = int iarray [@@deriving sexp_of, compare]

  (* The generator builds a valid (positive, exactly-multiplying) dims array and freezes
     it; the observer and shrinker defer to the [int array] instances. *)
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
    |> Generator.map ~f:Iarray.of_array
  ;;

  let quickcheck_observer =
    Observer.unmap [%quickcheck.observer: int array] ~f:Iarray.to_array
  ;;

  let quickcheck_shrinker =
    Shrinker.map
      [%quickcheck.shrinker: int array]
      ~f:Iarray.of_array
      ~f_inverse:Iarray.to_array
  ;;
end

module Tensor = struct
  include Tensor

  let quickcheck_generator_with_dims ~dims =
    let open Generator.Let_syntax in
    let total_elements = Iarray.fold dims ~init:1 ~f:( * ) in
    let%map values =
      List.init total_elements ~f:(fun _ -> Generator.float_inclusive (-100.) 100.)
      |> Generator.all
    in
    Tensor.of_list Float values |> Tensor.reshape ~dims
  ;;

  let quickcheck_generator =
    let open Generator.Let_syntax in
    let%bind dims = Dims.quickcheck_generator in
    quickcheck_generator_with_dims ~dims
  ;;

  let quickcheck_shrinker =
    Shrinker.create (fun (Tensor.T tensor) ->
      let dims = Tensor.Typed.dims tensor in
      let sliced_tensors =
        if Iarray.length dims > 0
        then
          Sequence.init dims.:(0) ~f:(fun i ->
            T (Tensor.Typed.left_slice tensor ~indices:[: i :]))
        else Sequence.empty
      in
      let smaller_tensors =
        match Tensor.Typed.type_ tensor with
        | Float ->
          Sequence.unfold ~init:tensor ~f:(fun tensor ->
            let tensor' = Tensor.Typed.map Float tensor ~f:(fun x -> x /. 2.) in
            if Tensor.Typed.allclose ~equal_nan:true tensor tensor'
            then None
            else Some (T tensor', tensor'))
        | Bool ->
          (* TODO: consider shrinking bool tensors *)
          Sequence.empty
      in
      Sequence.interleave (Sequence.of_list [ sliced_tensors; smaller_tensors ]))
  ;;

  let quickcheck_observer =
    Observer.of_hash_fold (fun hash_state (T tensor) ->
      let hash_state = ref hash_state in
      Tensor.Typed.iter tensor ~f:(fun value ->
        hash_state
        := match Tensor.Typed.type_ tensor with
           | Float -> Float.hash_fold_t !hash_state value
           | Bool -> Bool.hash_fold_t !hash_state value);
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
      | Exp
      | Log
      | Sigmoid
    [@@deriving quickcheck]
  end

  module Binary = struct
    type t = Op.Binary.t =
      | Add
      | Sub
      | Mul
      | Div
      | Eq
      | Gt
      | Lt
    [@@deriving quickcheck]
  end

  let reduce_dims_generator =
    [%quickcheck.generator: int Nonempty_list.t]
    |> Generator.filter ~f:(fun dims ->
      [%equal: int list]
        (Nonempty_list.to_list dims |> List.stable_dedup ~compare:Int.compare)
        (Nonempty_list.to_list dims))
  ;;

  type 'value t = 'value Op.t =
    | Unary of Unary.t * 'value
    | Binary of Binary.t * 'value * 'value
    | Matmul of 'value * 'value
    | Transpose of 'value
    | Sum of
        { value : 'value
        ; dims :
            [ `Just of (int Nonempty_list.t[@quickcheck.generator reduce_dims_generator])
            | `All
            ]
        ; keep_dims : bool
        }
    | Broadcast of
        { value : 'value
        ; dims : Dims.t
        }
    | Reshape of
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

let op_generator ~values_by_shape =
  let open Generator.Let_syntax in
  Op.quickcheck_generator Generator.unit
  |> generator_filter_bind ~f:(function
    | Unary (kind, ()) ->
      let valid_shapes =
        Map.keys values_by_shape
        |> List.filter ~f:(fun shape ->
          Fox_core.Op.infer_shape (Op.Unary (kind, shape)) |> Or_error.is_ok)
      in
      let%map value = choose valid_shapes >>| Map.find_exn values_by_shape >>= choose in
      Some (Op.Unary (kind, value))
    | Binary (kind, (), ()) ->
      let valid_shapes =
        Map.keys values_by_shape
        |> List.filter ~f:(fun shape ->
          Fox_core.Op.infer_shape (Op.Binary (kind, shape, shape)) |> Or_error.is_ok)
      in
      let%bind values = choose valid_shapes >>| Map.find_exn values_by_shape in
      let%map value1 = choose values
      and value2 = choose values in
      Some (Op.Binary (kind, value1, value2))
    | Matmul ((), ()) ->
      (match
         let%bind.Option all_shapes =
           match
             Map.keys values_by_shape
             |> List.filter ~f:(fun shape ->
               Iarray.length (Shape.dims shape) = 2
               &&
               match Shape.type_ shape with
               | T Float -> true
               | T Bool -> false)
           with
           | [] -> None
           | all_shapes -> Some all_shapes
         in
         match
           List.concat_map all_shapes ~f:(fun lhs_shape ->
             List.filter_map all_shapes ~f:(fun rhs_shape ->
               match lhs_shape, rhs_shape with
               | { dims = [: _; m :]; type_ = _ }, { dims = [: m' :]; type_ = _ }
               | { dims = [: _; m :]; type_ = _ }, { dims = [: m'; _ :]; type_ = _ } ->
                 Option.some_if (m = m') (lhs_shape, rhs_shape)
               | _ -> None))
         with
         | [] -> None
         | dim_pairs -> Some dim_pairs
       with
       | None -> return None
       | Some dim_pairs ->
         let%bind lhs_dims, rhs_dims = choose dim_pairs in
         let%map lhs = Map.find_exn values_by_shape lhs_dims |> choose
         and rhs = Map.find_exn values_by_shape rhs_dims |> choose in
         Some (Op.Matmul (lhs, rhs)))
    | Transpose () ->
      (match
         Map.keys values_by_shape
         |> List.filter ~f:(fun shape -> Iarray.length (Shape.dims shape) = 2)
       with
       | [] -> return None
       | all_dims ->
         let%map value = choose all_dims >>| Map.find_exn values_by_shape >>= choose in
         Some (Op.Transpose value))
    | Sum { value = (); dims; keep_dims } ->
      (* Filter to operand shapes for which the sum is actually valid - this rejects bool
         shapes (sum is float-only) and, for [`Just], out-of-bounds reduction dims. *)
      let valid_shapes =
        Map.keys values_by_shape
        |> List.filter ~f:(fun shape ->
          Fox_core.Op.infer_shape (Op.Sum { value = shape; dims; keep_dims })
          |> Or_error.is_ok)
      in
      (match valid_shapes with
       | [] -> return None
       | valid_shapes ->
         let%map value =
           choose valid_shapes >>| Map.find_exn values_by_shape >>= choose
         in
         Some (Op.Sum { value; dims; keep_dims }))
    | (Broadcast { value = (); dims = _ } | Reshape { value = (); dims = _ }) as op ->
      (match
         Map.keys values_by_shape
         |> List.filter ~f:(fun input_shape ->
           Fox_core.Op.map op ~f:(fun () -> input_shape)
           |> Fox_core.Op.infer_shape
           |> Or_error.is_ok)
       with
       | [] -> return None
       | all_dims ->
         let%map value = choose all_dims >>| Map.find_exn values_by_shape >>= choose in
         Some (Fox_core.Op.map op ~f:(fun () -> value))))
;;

let expr_generator ~op_nums =
  let open Generator.Let_syntax in
  let%bind dims = Dims.quickcheck_generator in
  let arg : Expr.Var.t = { name = "arg"; shape = { dims; type_ = T Float } } in
  let%map _values_by_shape, equations =
    List.range 0 op_nums
    |> List.fold
         ~init:
           (return
              (Shape.Map.singleton { dims; type_ = T Float } [ Expr.Atom.Var arg ], []))
         ~f:(fun accum i ->
           let%bind values_by_shape, equations = accum in
           let%map op = op_generator ~values_by_shape in
           let shape =
             Fox_core.Op.map op ~f:Expr.Atom.shape |> Fox_core.Op.infer_shape_exn
           in
           let var : Expr.Var.t = { name = [%string "v_%{i#Int}"]; shape } in
           ( Map.add_multi values_by_shape ~key:shape ~data:(Expr.Atom.Var var)
           , { Expr.Eq.var; op } :: equations ))
  in
  let out = List.hd_exn equations |> Expr.Eq.var in
  Expr.create
    ~parameters:[ arg ]
    ~consts:Expr.Var.Map.empty
    ~equations:(List.rev equations)
    ~return_vals:[ Expr.Atom.Var out ]
    ~out_tree_def:(Value_tree.Def.leaf ~dims:(Expr.Var.dims out))
;;

let%expect_test "expr_generator" =
  let random = Splittable_random.of_int 0 in
  for i = 1 to 5 do
    let expr = Generator.generate (expr_generator ~op_nums:i) ~size:6 ~random in
    Expr.to_string_hum expr ~value_to_string:Value.to_string |> print_endline;
    print_endline "--------------------------------"
  done;
  [%expect
    {|
    arg[4]: float ->
    v_0[4]: float = broadcast arg dims=[4];
    ( v_0 )
    --------------------------------
    arg[2,2]: float ->
    v_0[2,2]: float = sub arg arg;
    v_1[2,2]: float = sin v_0;
    ( v_1 )
    --------------------------------
    arg[5]: float ->
    v_0[5]: float = div arg arg;
    v_1[5]: bool = lt v_0 v_0;
    v_2[5]: float = cos arg;
    ( v_2 )
    --------------------------------
    arg[4]: float ->
    v_0[4]: float = sub arg arg;
    v_1[4]: bool = lt arg arg;
    v_2[4]: bool = gt arg arg;
    v_3[4]: bool = lt arg arg;
    ( v_3 )
    --------------------------------
    arg[4]: float ->
    v_0[4]: bool = gt arg arg;
    v_1[4]: float = sin arg;
    v_2[4]: bool = reshape v_0 dims=[4];
    v_3[4]: float = div v_1 v_1;
    v_4[4]: bool = lt arg v_1;
    ( v_4 )
    --------------------------------
    |}]
;;

let fun_generator ~op_nums =
  let open Generator.Let_syntax in
  let%bind expr = expr_generator ~op_nums in
  match Expr.parameters expr with
  | [ var ] ->
    let%map tensor = Tensor.quickcheck_generator_with_dims ~dims:(Expr.Var.dims var) in
    tensor, expr
  | _ -> failwith "fun_generator: expected 1 parameter"
;;

(* XLA leaks a few threads per compiled executable; without a periodic [Gc.full_major ()]
   to run finalizers, pthread_create eventually fails with EAGAIN. *)
let gc_every = 50

(* The eager and XLA backends accumulate reductions in different orders, so
   large-magnitude results (e.g. [sum (exp x)]) legitimately differ by a few ULPs.
   [Tensor.allclose] uses [Float.robustly_compare], an *absolute* 1e-7 tolerance, which
   those differences blow past once values are large. Compare with a relative tolerance
   instead. [Tensor.allclose] still handles bool tensors, nan (via [equal_nan]), and
   matching infinities. *)
let agree a b =
  Tensor.allclose ~equal_nan:true a b
  ||
  match Tensor.type_ a, Tensor.type_ b with
  | T Float, T Float when [%equal: int iarray] (Tensor.dims a) (Tensor.dims b) ->
    let flat t = Tensor.reshape t ~dims:[: Tensor.length t :] in
    let a = flat a
    and b = flat b in
    List.for_all
      (List.init (Tensor.length a) ~f:Fn.id)
      ~f:(fun i ->
        let x = Tensor.get_exn Float a [: i :] in
        let y = Tensor.get_exn Float b [: i :] in
        Float.( <= ) (Float.abs (x -. y)) (1e-6 +. (1e-5 *. Float.abs y)))
  | _, _ -> false
;;

let%expect_test "eval expr vs xla" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  let trial = ref 0 in
  (* Multi-op programs (not just single ops) exercise compositions across both backends. *)
  List.iter [ 1; 2; 3; 4 ] ~f:(fun op_nums ->
    Quickcheck.test
      (fun_generator ~op_nums)
      ~trials:100
      ~sexp_of:(fun (tensor, expr) ->
        [%sexp
          { op_nums : int
          ; tensor : Tensor.t
          ; expr : string = Expr.to_string_hum expr ~value_to_string:Value.to_string
          }])
      ~f:(fun (tensor, expr) ->
        incr trial;
        if !trial % gc_every = 0 then Gc.full_major ();
        let f value = eval_expr' expr value in
        let value = Value.of_tensor tensor in
        let eval_result = eval ~f:(fun () -> f value) in
        let xla_result = Staged.unstage (Fox_jit.jit' ~f ()) value in
        assert (agree (Value.to_tensor_exn eval_result) (Value.to_tensor_exn xla_result))))
;;

let%expect_test "grad+jit vs grad+eval" =
  let test ~f ~x =
    eval ~f:(fun () -> f x) |> [%sexp_of: Value.t] |> print_s;
    Staged.unstage (Fox_jit.jit' ~f ()) x |> [%sexp_of: Value.t] |> print_s
  in
  test
    ~f:(fun value -> grad' ~f:(fun value -> Value.O.(value * value)) ~x:value)
    ~x:(Value.of_float 1.);
  [%expect {|
    (Tensor 2 Float)
    (Tensor 2 Float)
    |}];
  test
    ~f:(fun value ->
      grad' ~f:(fun value -> Value.O.(value / value) |> Value.sum) ~x:value)
    ~x:(Value.of_tensor (Tensor.of_list Float [ 1.; 1. ]));
  [%expect
    {|
    (Tensor (0 0) (dims (2)) (type_ Float))
    (Tensor (0 0) (dims (2)) (type_ Float))
    |}];
  test
    ~f:(fun value -> grad' ~f:(fun value -> Value.sqrt value |> Value.sum) ~x:value)
    ~x:(Value.of_tensor (Tensor.of_list Float [ 1.; 1. ]));
  [%expect
    {|
    (Tensor (0.5 0.5) (dims (2)) (type_ Float))
    (Tensor (0.5 0.5) (dims (2)) (type_ Float))
    |}]
;;

let%expect_test "eval grad expr vs xla" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  let trial = ref 0 in
  List.iter [ 1; 2; 3; 4 ] ~f:(fun op_nums ->
    Quickcheck.test
      (fun_generator ~op_nums)
      ~trials:100
      ~sexp_of:(fun (tensor, expr) ->
        [%sexp
          { op_nums : int
          ; tensor : Tensor.t
          ; expr : string = Expr.to_string_hum expr ~value_to_string:Value.to_string
          }])
      ~f:(fun (tensor, expr) ->
        incr trial;
        if !trial % gc_every = 0 then Gc.full_major ();
        let f value =
          grad'
            ~f:(fun value ->
              let expr_result = eval_expr' expr value in
              let expr_result_shape = Value.shape expr_result in
              match Shape.type_ expr_result_shape with
              | T Float -> Value.sum expr_result
              | T Bool ->
                (* TODO: somehow prevent return type from being a boolean? *)
                (* We can't really differentiate bool tensors, so we just sum the input
                   instead. *)
                Value.sum value)
            ~x:value
        in
        let value = Value.of_tensor tensor in
        let eval_result = eval ~f:(fun () -> f value) in
        let xla_result = Staged.unstage (Fox_jit.jit' ~f ()) value in
        assert (agree (Value.to_tensor_exn eval_result) (Value.to_tensor_exn xla_result))))
;;

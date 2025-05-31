open! Core
open! Fox_core

let tensor_of_xla_literal literal =
  Xla.Literal.to_bigarray literal ~kind:Bigarray.float64 |> Tensor.Private.of_bigarray
;;

let tensor_to_xla_literal tensor =
  Tensor.Private.to_bigarray tensor |> Xla.Literal.of_bigarray
;;

let xla_subcomp
      ({ parameters; equations; return_vals; out_tree_def = _ } : Expr.t)
      arguments
      ~builder
  =
  let env = List.zip_exn parameters arguments |> Expr.Var.Map.of_alist_exn in
  let read_atom (atom : Expr.Atom.t) ~env =
    match atom with
    | Var var -> Map.find_exn env var
    | Value value ->
      let tensor = Value.to_tensor_exn value in
      let op = tensor_to_xla_literal tensor |> Xla.Op.constant ~builder in
      op, Tensor.dims tensor
  in
  let env =
    List.fold equations ~init:env ~f:(fun env { var; op } ->
      let op = Op.map op ~f:(read_atom ~env) in
      let xla_op =
        match op with
        | Unary (kind, (a, _)) ->
          let f =
            match kind with
            | Neg -> Xla.Op.neg
            | Sin -> Xla.Op.sin
            | Cos -> Xla.Op.cos
            | Sqrt -> Xla.Op.sqrt
          in
          f a
        | Binary (kind, (a, _), (b, _)) ->
          let f =
            match kind with
            | Add -> Xla.Op.add
            | Sub -> Xla.Op.sub
            | Mul -> Xla.Op.mul
            | Div -> Xla.Op.div
          in
          f a b
        | Matmul ((a, _), (b, _)) -> Xla.Op.matmul a b
        | Transpose (a, _) ->
          (* TODO: support arbitrary dimensions *)
          Xla.Op.transpose a ~dim_indexes:[| 1; 0 |]
        | Sum { value = value, in_dims; dims; keep_dims } ->
          let dims =
            match dims with
            | `All -> Array.init (Array.length in_dims) ~f:Fn.id
            | `Just dims -> Nonempty_list.to_list dims |> Array.of_list
          in
          Xla.Op.reduce_sum value ~dims ~keep_dims
        | Broadcast { value = value, in_dims; dims = out_dims } ->
          let padding_length = Array.length out_dims - Array.length in_dims in
          Xla.Op.broadcast_in_dim
            value
            ~out_dims
            ~broadcast_dims:(Array.mapi in_dims ~f:(fun i _ -> padding_length + i))
      in
      let dims = Op.map op ~f:snd |> Op.infer_dims in
      Map.add_exn env ~key:var ~data:(xla_op, dims))
  in
  Nonempty_list.map return_vals ~f:(fun atom -> read_atom atom ~env |> fst)
  |> Nonempty_list.to_list
  |> Xla.Op.tuple ~builder
;;

let xla_callable (expr : Expr.t) =
  let xla_builder = Xla.Builder.create ~name:"xla_call" in
  let xla_params =
    List.mapi expr.parameters ~f:(fun i { name; dims } ->
      (* TODO: right now we assume all parameters are single-element tensors. *)
      Xla.Op.parameter name ~id:i ~ty:F64 ~dims ~builder:xla_builder, dims)
  in
  let out = xla_subcomp expr xla_params ~builder:xla_builder in
  let xla_client = Xla.Client.cpu () in
  let xla_device = Xla.Client.addressable_devices xla_client |> List.hd_exn in
  let xla_exe = Xla.Computation.build ~root:out |> Xla.Executable.compile xla_client in
  Staged.stage (fun inputs ->
    let inputs =
      List.map inputs ~f:(fun tensor ->
        tensor_to_xla_literal tensor |> Xla.Buffer.of_host_literal ~device:xla_device)
      |> List.to_array
    in
    let buffers = Xla.Executable.execute_b xla_exe inputs in
    let buffer = buffers.(0).(0) in
    Xla.Buffer.to_literal_sync buffer
    |> Xla.Literal.decompose_tuple
    |> Array.to_list
    |> Nonempty_list.of_list_exn
    |> Nonempty_list.map ~f:tensor_of_xla_literal)
;;

(* TODO: One crucial difference between the implementation here and autodidax is
   that the compilation is not cached. One approach would be to have
   [jit] take an [in_tree_def] argument, and return a staged function of type
   [in_ -> out] (with some validation that the tree def matches up, or
   alternatively recompile if it doesn't?).
*)
let jit
      (type in_ out)
      (module In : Treeable_intf.S with type t = in_)
      (module Out : Treeable_intf.S with type t = out)
      ~f
      (input : in_)
  : out
  =
  let input_tree = In.tree_of_t input in
  let input_tree_def = Value_tree.to_def input_tree in
  let flattened_input_tensors =
    Value_tree.flatten input_tree |> List.map ~f:Value.to_tensor_exn
  in
  let expr = build_expr (module In) (module Out) ~f ~in_tree_def:input_tree_def in
  let xla_callable = xla_callable expr |> Staged.unstage in
  let output =
    xla_callable flattened_input_tensors
    |> Nonempty_list.to_list
    |> List.map ~f:Value.of_tensor
  in
  Value_tree.unflatten output ~def:expr.out_tree_def |> Out.t_of_tree
;;

let jit' ~f ~x = jit (module Value) (module Value) ~f x

let%expect_test "jit'" =
  (* Filters out noisy XLA log message *)
  let print output =
    String.split_lines output
    |> List.filter ~f:(Fn.non (String.is_substring ~substring:"TfrtCpuClient created"))
    |> String.concat
    |> print_endline
  in
  jit' ~f:foo ~x:(Value.of_float 2.) |> [%sexp_of: Value.t] |> print_s;
  print [%expect.output];
  [%expect {| (Tensor 10) |}];
  (* Two-argument function *)
  jit
    (module Treeable.Tuple2 (Value) (Value))
    (module Value)
    ~f:(fun (a, b) -> Value.O.(Value.sin a * Value.cos b))
    (Value.of_float 2., Value.of_float 3.)
  |> [%sexp_of: Value.t]
  |> print_s;
  print [%expect.output];
  [%expect {| (Tensor -0.90019762973551742) |}]
;;

let%expect_test "jit and matmul" =
  let print output =
    String.split_lines output
    |> List.filter ~f:(Fn.non (String.is_substring ~substring:"TfrtCpuClient created"))
    |> String.concat
    |> print_endline
  in
  let a = Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] |> Value.of_tensor in
  let b = Tensor.of_list2_exn [ [ 5.; 6. ]; [ 7.; 8. ] ] |> Value.of_tensor in
  jit
    (module Treeable.Tuple2 (Value) (Value))
    (module Value)
    ~f:(fun (a, b) -> Value.matmul a b)
    (a, b)
  |> [%sexp_of: Value.t]
  |> print_s;
  print [%expect.output];
  [%expect {| (Tensor ((19 22) (43 50)) (dims (2 2))) |}]
;;

let%expect_test "jit and sum" =
  let print output =
    String.split_lines output
    |> List.filter ~f:(Fn.non (String.is_substring ~substring:"TfrtCpuClient created"))
    |> String.concat
    |> print_endline
  in
  jit'
    ~f:(fun x -> Value.sum x)
    ~x:(Value.of_tensor (Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  print [%expect.output];
  [%expect {| (Tensor 10) |}]
;;

let%expect_test "jit and broadcast" =
  let print output =
    String.split_lines output
    |> List.filter ~f:(Fn.non (String.is_substring ~substring:"TfrtCpuClient created"))
    |> String.concat
    |> print_endline
  in
  jit'
    ~f:(fun x -> Value.broadcast x ~dims:[| 2; 2; 2 |])
    ~x:(Value.of_tensor (Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  print [%expect.output];
  [%expect {| (Tensor (((1 2) (3 4)) ((1 2) (3 4))) (dims (2 2 2))) |}]
;;

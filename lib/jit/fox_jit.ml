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
      let dims = Op.map op ~f:snd |> Op.infer_dims_exn in
      Map.add_exn env ~key:var ~data:(xla_op, dims))
  in
  Nonempty_list.map return_vals ~f:(fun atom -> read_atom atom ~env |> fst)
  |> Nonempty_list.to_list
  |> Xla.Op.tuple ~builder
;;

let xla_builder = lazy (Xla.Builder.create ~name:"xla_call")

let xla_callable ?(print_hlo = false) (expr : Expr.t) =
  let xla_builder = Lazy.force xla_builder in
  let xla_params =
    List.mapi expr.parameters ~f:(fun i { name; dims } ->
      (* TODO: right now we assume all parameters are single-element tensors. *)
      Xla.Op.parameter name ~id:i ~ty:F64 ~dims ~builder:xla_builder, dims)
  in
  let out = xla_subcomp expr xla_params ~builder:xla_builder in
  let xla_client = Xla.Client.cpu () in
  let xla_device = Xla.Client.addressable_devices xla_client |> List.hd_exn in
  let computation = Xla.Computation.build ~root:out in
  if print_hlo
  then
    Xla.Computation.proto computation
    |> Xla.Hlo_module_proto.to_string
    |> String.strip
    |> print_endline;
  let xla_exe = Xla.Executable.compile xla_client computation in
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
      ?print_hlo
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
  let xla_callable = xla_callable ?print_hlo expr |> Staged.unstage in
  let output =
    xla_callable flattened_input_tensors
    |> Nonempty_list.to_list
    |> List.map ~f:Value.of_tensor
  in
  Value_tree.unflatten output ~def:expr.out_tree_def |> Out.t_of_tree
;;

let jit' ?print_hlo ~f x = jit (module Value) (module Value) ?print_hlo ~f x

let%expect_test "jit'" =
  (* Suppresses noisy XLA log message *)
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  jit' ~print_hlo:true ~f:foo (Value.of_float 2.) |> [%sexp_of: Value.t] |> print_s;
  [%expect
    {|
    HloModule xla_call.6, entry_computation_layout={(f64[])->(f64[])}

    ENTRY %xla_call.6 (v_0.1: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %constant.2 = f64[] constant(3)
      %add.3 = f64[] add(f64[] %v_0.1, f64[] %constant.2)
      %multiply.4 = f64[] multiply(f64[] %v_0.1, f64[] %add.3)
      ROOT %tuple.5 = (f64[]) tuple(f64[] %multiply.4)
    }
    (Tensor 10)
    |}];
  (* Two-argument function *)
  jit
    ~print_hlo:true
    (module Treeable.Tuple2 (Value) (Value))
    (module Value)
    ~f:(fun (a, b) -> Value.O.(Value.sin a * Value.cos b))
    (Value.of_float 2., Value.of_float 3.)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.13, entry_computation_layout={(f64[],f64[])->(f64[])}

    ENTRY %xla_call.13 (v_0.7: f64[], v_1.8: f64[]) -> (f64[]) {
      %v_0.7 = f64[] parameter(0)
      %sine.10 = f64[] sine(f64[] %v_0.7)
      %v_1.8 = f64[] parameter(1)
      %cosine.9 = f64[] cosine(f64[] %v_1.8)
      %multiply.11 = f64[] multiply(f64[] %sine.10, f64[] %cosine.9)
      ROOT %tuple.12 = (f64[]) tuple(f64[] %multiply.11)
    }
    (Tensor -0.90019762973551742)
    |}]
;;

let%expect_test "jit and matmul" =
  let a = Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] |> Value.of_tensor in
  let b = Tensor.of_list2_exn [ [ 5.; 6. ]; [ 7.; 8. ] ] |> Value.of_tensor in
  jit
    ~print_hlo:true
    (module Treeable.Tuple2 (Value) (Value))
    (module Value)
    ~f:(fun (a, b) -> Value.matmul a b)
    (a, b)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.18, entry_computation_layout={(f64[2,2]{1,0},f64[2,2]{1,0})->(f64[2,2]{1,0})}

    ENTRY %xla_call.18 (v_0.14: f64[2,2], v_1.15: f64[2,2]) -> (f64[2,2]) {
      %v_0.14 = f64[2,2]{1,0} parameter(0)
      %v_1.15 = f64[2,2]{1,0} parameter(1)
      %dot.16 = f64[2,2]{1,0} dot(f64[2,2]{1,0} %v_0.14, f64[2,2]{1,0} %v_1.15), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %tuple.17 = (f64[2,2]{1,0}) tuple(f64[2,2]{1,0} %dot.16)
    }
    (Tensor ((19 22) (43 50)) (dims (2 2)))
    |}]
;;

let%expect_test "jit and sum" =
  jit'
    ~print_hlo:true
    ~f:(fun x -> Value.sum x)
    (Value.of_tensor (Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.27, entry_computation_layout={(f64[2,2]{1,0})->(f64[])}

    %sum.21 (x.22: f64[], y.23: f64[]) -> f64[] {
      %x.22 = f64[] parameter(0)
      %y.23 = f64[] parameter(1)
      ROOT %add.24 = f64[] add(f64[] %x.22, f64[] %y.23)
    }

    ENTRY %xla_call.27 (v_0.19: f64[2,2]) -> (f64[]) {
      %v_0.19 = f64[2,2]{1,0} parameter(0)
      %constant.20 = f64[] constant(0)
      %reduce.25 = f64[] reduce(f64[2,2]{1,0} %v_0.19, f64[] %constant.20), dimensions={0,1}, to_apply=%sum.21
      ROOT %tuple.26 = (f64[]) tuple(f64[] %reduce.25)
    }
    (Tensor 10)
    |}]
;;

let%expect_test "jit and broadcast" =
  jit'
    ~print_hlo:true
    ~f:(fun x -> Value.broadcast x ~dims:[| 2; 2; 2 |])
    (Value.of_tensor (Tensor.of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.31, entry_computation_layout={(f64[2,2]{1,0})->(f64[2,2,2]{2,1,0})}

    ENTRY %xla_call.31 (v_0.28: f64[2,2]) -> (f64[2,2,2]) {
      %v_0.28 = f64[2,2]{1,0} parameter(0)
      %broadcast.29 = f64[2,2,2]{2,1,0} broadcast(f64[2,2]{1,0} %v_0.28), dimensions={1,2}
      ROOT %tuple.30 = (f64[2,2,2]{2,1,0}) tuple(f64[2,2,2]{2,1,0} %broadcast.29)
    }
    (Tensor (((1 2) (3 4)) ((1 2) (3 4))) (dims (2 2 2)))
    |}]
;;

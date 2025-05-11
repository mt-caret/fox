open! Core
open! Ox_core

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
      Value.to_tensor_exn value |> tensor_to_xla_literal |> Xla.Op.constant ~builder
  in
  let env =
    List.fold equations ~init:env ~f:(fun env { var; op } ->
      let xla_op =
        match Op.map op ~f:(read_atom ~env) with
        | Add (a, b) -> Xla.Op.add a b
        | Sub (a, b) -> Xla.Op.sub a b
        | Mul (a, b) -> Xla.Op.mul a b
        | Neg a -> Xla.Op.neg a
        | Sin a -> Xla.Op.sin a
        | Cos a -> Xla.Op.cos a
      in
      Map.add_exn env ~key:var ~data:xla_op)
  in
  Nonempty_list.map return_vals ~f:(read_atom ~env)
  |> Nonempty_list.to_list
  |> Xla.Op.tuple ~builder
;;

let xla_callable (expr : Expr.t) =
  let xla_builder = Xla.Builder.create ~name:"xla_call" in
  let xla_params =
    List.mapi expr.parameters ~f:(fun i (Var name) ->
      (* TODO: right now we assume all parameters are single-element tensors. *)
      Xla.Op.parameter name ~id:i ~ty:F64 ~dims:[||] ~builder:xla_builder)
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
  let expr = build_expr (module In) (module Out) ~f ~in_tree_def:input_tree_def in
  let xla_callable = xla_callable expr |> Staged.unstage in
  let output =
    Value_tree.flatten input_tree
    |> List.map ~f:Value.to_tensor_exn
    |> xla_callable
    |> Nonempty_list.to_list
    |> List.map ~f:Value.of_tensor
  in
  Value_tree.unflatten output ~def:expr.out_tree_def |> Out.t_of_tree
;;

let jit' ~f ~x = jit (module Value) (module Value) ~f x

let%expect_test "jit'" =
  jit' ~f:foo ~x:(Value.of_float 2.) |> [%sexp_of: Value.t] |> print_s;
  String.split_lines [%expect.output]
  |> List.filter ~f:(Fn.non (String.is_substring ~substring:"TfrtCpuClient created"))
  |> String.concat
  |> print_endline;
  [%expect {| (Tensor 10) |}]
;;

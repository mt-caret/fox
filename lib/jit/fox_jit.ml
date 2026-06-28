open! Core
open! Fox_core

let tensor_of_xla_literal literal ~shape =
  match Shape.type_ shape with
  | T Float ->
    Xla.Literal.to_bigarray literal ~kind:Bigarray.float64
    |> Tensor.Private.of_float_bigarray
    |> Tensor.of_typed
  | T Bool ->
    Xla.Literal.to_bigarray literal ~kind:Bigarray.char
    |> Tensor.Private.of_char_bigarray
    |> Tensor.of_typed
;;

let tensor_to_xla_literal tensor =
  Tensor.Private.to_bigarray tensor |> Xla.Literal.of_bigarray
;;

let xla_subcomp
  ({ parameters; consts = _; equations; return_vals; out_tree_def = _ } : Value.t Expr.t)
  arguments
  ~const_ops
  ~builder
  =
  let env =
    List.zip_exn parameters arguments
    |> Expr.Var.Map.of_alist_exn
    |> Map.merge_disjoint_exn const_ops
  in
  let read_atom (atom : Value.t Expr.Atom.t) ~env =
    match atom with
    | Var var -> Map.find_exn env var
    | Value value ->
      let tensor = Value.to_typed_tensor_exn Float value in
      let op = tensor_to_xla_literal tensor |> Xla.Op.constant ~builder in
      op, Tensor.Typed.shape tensor
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
            | Exp -> Xla.Op.exp
            | Log -> Xla.Op.log
            | Sigmoid ->
              fun x ->
                let one =
                  Tensor.Typed.ones ~dims:(Iarray.of_array (Xla.Op.dims x))
                  |> tensor_to_xla_literal
                  |> Xla.Op.constant ~builder
                in
                let is_non_negative = Xla.Op.ge x (Xla.Op.zeros_like x) in
                Xla.Op.select
                  ~mask:is_non_negative
                  ~on_true:(Xla.Op.div one (Xla.Op.add one (Xla.Op.exp (Xla.Op.neg x))))
                  ~on_false:(Xla.Op.div (Xla.Op.exp x) (Xla.Op.add one (Xla.Op.exp x)))
          in
          f a
        | Binary (kind, (a, _), (b, _)) ->
          let f =
            match kind with
            | Add -> Xla.Op.add
            | Sub -> Xla.Op.sub
            | Mul -> Xla.Op.mul
            | Div -> Xla.Op.div
            | Eq -> Xla.Op.eq
            | Gt -> Xla.Op.gt
            | Lt -> Xla.Op.lt
          in
          f a b
        | Matmul ((a, _), (b, _)) -> Xla.Op.matmul a b
        | Transpose (a, _) ->
          (* TODO: support arbitrary dimensions *)
          Xla.Op.transpose a ~dim_indexes:[| 1; 0 |]
        | Sum { value = value, in_shape; dims; keep_dims } ->
          let in_dims = Iarray.to_array (Shape.dims in_shape) in
          let dims =
            match dims with
            | `All -> Array.init (Array.length in_dims) ~f:Fn.id
            | `Just dims -> Nonempty_list.to_list dims |> Array.of_list
          in
          Xla.Op.reduce_sum value ~dims ~keep_dims
        | Broadcast { value = value, in_shape; dims = out_dims } ->
          let in_dims = Iarray.to_array (Shape.dims in_shape) in
          let out_dims = Iarray.to_array out_dims in
          let padding_length = Array.length out_dims - Array.length in_dims in
          Xla.Op.broadcast_in_dim
            value
            ~out_dims
            ~broadcast_dims:(Array.mapi in_dims ~f:(fun i _ -> padding_length + i))
        | Reshape { value = value, in_shape; dims } ->
          (* [dims] may contain a -1 placeholder; XLA requires concrete dims (passing -1
             aborts the process via a C++ CHECK), so resolve it the same way the eager
             backend and [Op.infer_shape] do. *)
          let out_dims =
            Op.infer_shape_exn (Reshape { value = in_shape; dims })
            |> Shape.dims
            |> Iarray.to_array
          in
          Xla.Op.reshape value ~dims:out_dims
      in
      let shape = Op.map op ~f:snd |> Op.infer_shape_exn in
      Map.add_exn env ~key:var ~data:(xla_op, shape))
  in
  let ops, shapes =
    Nonempty_list.map return_vals ~f:(fun atom -> read_atom atom ~env)
    |> Nonempty_list.to_list
    |> List.unzip
  in
  Xla.Op.tuple ops ~builder, shapes
;;

let xla_callable ?(print_hlo = false) (expr : Value.t Expr.t) =
  (* A fresh builder per compilation. An XLA builder records its first error and replays
     it from every subsequent [build], so a single process-global builder would let one
     failed compilation poison every later (even unrelated, valid) one. *)
  let xla_builder = Xla.Builder.create ~name:"xla_call" in
  let xla_params =
    List.mapi expr.parameters ~f:(fun i { name; shape = { dims; type_ } as shape } ->
      (* TODO: right now we assume all parameters are single-element tensors. *)
      ( Xla.Op.parameter
          name
          ~id:i
          ~ty:
            (match type_ with
             | T Float -> F64
             | T Bool -> Pred)
          ~dims:(Iarray.to_array dims)
          ~builder:xla_builder
      , shape ))
  in
  (* Hoisted constants become XLA parameters (after the real parameters), supplied per
     call at execution time rather than baked in, so one compiled executable serves any
     constant values of the same structure. Their shapes come from the const vars, so
     compilation depends only on the structure, not the values. *)
  let num_params = List.length expr.parameters in
  (* CR: bool constants are unsupported here - the parameter type is hard-coded to [F64]
     (and [jit] below feeds const values via [to_typed_tensor_exn Float], and
     [tensor_to_xla_literal] only handles float bigarrays). A jitted program that closes
     over a bool constant therefore raises (a bool const compared against a float-typed
     param gives "compare with different element types: pred vs f64"). Eager handles such
     programs fine. See [test_robustness.ml] "jit: bool constant unsupported". *)
  let const_ops =
    Map.to_alist expr.consts
    |> List.mapi ~f:(fun i (var, _value) ->
      let shape = Expr.Var.shape var in
      let op =
        Xla.Op.parameter
          (Expr.Var.name var)
          ~id:(num_params + i)
          ~ty:F64
          ~dims:(Iarray.to_array (Shape.dims shape))
          ~builder:xla_builder
      in
      var, (op, shape))
    |> Expr.Var.Map.of_alist_exn
  in
  let out, out_shapes = xla_subcomp expr xla_params ~const_ops ~builder:xla_builder in
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
  Staged.stage (fun inputs const_values ->
    let inputs =
      List.map (inputs @ const_values) ~f:(fun tensor ->
        tensor_to_xla_literal tensor |> Xla.Buffer.of_host_literal ~device:xla_device)
      |> List.to_array
    in
    let buffers = Xla.Executable.execute_b xla_exe inputs in
    let buffer = buffers.(0).(0) in
    let out_literals =
      Xla.Buffer.to_literal_sync buffer |> Xla.Literal.decompose_tuple |> Array.to_list
    in
    List.zip_exn out_literals out_shapes
    |> Nonempty_list.of_list_exn
    |> Nonempty_list.map ~f:(fun (literal, shape) -> tensor_of_xla_literal literal ~shape))
;;

module Structure = struct
  (* The value-free structure of a traced program - the compilation cache key. *)
  type t = unit Expr.t [@@deriving compare, hash, sexp_of]
end

(* [jit] returns a reusable function that traces [f] on every call and reuses a compiled
   executable cached by the program's [Structure.t] (the traced [expr] with all values
   erased). Because constants are fed as runtime parameters, one executable serves every
   call that shares a structure, and each call supplies its own constants - so even a
   closure whose captured constants change reuses the executable and still gets correct
   results.

   This is the autodidax model (re-trace every call, cache only the compilation), rather
   than keying on input shapes to skip re-tracing (real JAX). The structure key stays
   sound even if value-dependent control flow is ever added, since distinct programs get
   distinct keys. *)
let jit
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ?print_hlo
  ~f
  ()
  : (in_ -> out) Staged.t
  =
  let cache = Hashtbl.create (module Structure) in
  Staged.stage (fun input ->
    let input_tree = In.tree_of_t input in
    let input_tree_def = Value_tree.to_def input_tree in
    let flattened_input_tensors =
      Value_tree.flatten input_tree |> List.map ~f:(Value.to_typed_tensor_exn Float)
    in
    let expr = build_expr (module In) (module Out) ~f ~in_tree_def:input_tree_def in
    let callable =
      Hashtbl.find_or_add
        cache
        (Expr.map expr ~f:(fun _ -> ()))
        ~default:(fun () -> xla_callable ?print_hlo expr |> Staged.unstage)
    in
    let const_values =
      Map.data expr.consts |> List.map ~f:(Value.to_typed_tensor_exn Float)
    in
    let output =
      callable flattened_input_tensors const_values
      |> Nonempty_list.to_list
      |> List.map ~f:Value.of_tensor
    in
    Value_tree.unflatten output ~def:expr.out_tree_def |> Out.t_of_tree)
;;

let jit' ?print_hlo ~f () = jit (module Value) (module Value) ?print_hlo ~f ()
let foo x = Value.O.(x * (x + Value.of_float 3.))

let%expect_test "jit'" =
  (* Suppresses noisy XLA log message *)
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  Staged.unstage (jit' ~print_hlo:true ~f:foo ()) (Value.of_float 2.)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.6, entry_computation_layout={(f64[],f64[])->(f64[])}

    ENTRY %xla_call.6 (v_0.1: f64[], c_0.2: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %c_0.2 = f64[] parameter(1)
      %add.3 = f64[] add(f64[] %v_0.1, f64[] %c_0.2)
      %multiply.4 = f64[] multiply(f64[] %v_0.1, f64[] %add.3)
      ROOT %tuple.5 = (f64[]) tuple(f64[] %multiply.4)
    }
    (Tensor 10 Float)
    |}];
  (* Two-argument function *)
  Staged.unstage
    (jit
       ~print_hlo:true
       (module Treeable.Tuple2 (Value) (Value))
       (module Value)
       ~f:(fun (a, b) -> Value.O.(Value.sin a * Value.cos b))
       ())
    (Value.of_float 2., Value.of_float 3.)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.7, entry_computation_layout={(f64[],f64[])->(f64[])}

    ENTRY %xla_call.7 (v_0.1: f64[], v_1.2: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %sine.4 = f64[] sine(f64[] %v_0.1)
      %v_1.2 = f64[] parameter(1)
      %cosine.3 = f64[] cosine(f64[] %v_1.2)
      %multiply.5 = f64[] multiply(f64[] %sine.4, f64[] %cosine.3)
      ROOT %tuple.6 = (f64[]) tuple(f64[] %multiply.5)
    }
    (Tensor -0.90019762973551742 Float)
    |}]
;;

let%expect_test "jit and matmul" =
  let a = Tensor.of_list2_exn Float [ [ 1.; 2. ]; [ 3.; 4. ] ] |> Value.of_tensor in
  let b = Tensor.of_list2_exn Float [ [ 5.; 6. ]; [ 7.; 8. ] ] |> Value.of_tensor in
  Staged.unstage
    (jit
       ~print_hlo:true
       (module Treeable.Tuple2 (Value) (Value))
       (module Value)
       ~f:(fun (a, b) -> Value.matmul a b)
       ())
    (a, b)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.5, entry_computation_layout={(f64[2,2]{1,0},f64[2,2]{1,0})->(f64[2,2]{1,0})}

    ENTRY %xla_call.5 (v_0.1: f64[2,2], v_1.2: f64[2,2]) -> (f64[2,2]) {
      %v_0.1 = f64[2,2]{1,0} parameter(0)
      %v_1.2 = f64[2,2]{1,0} parameter(1)
      %dot.3 = f64[2,2]{1,0} dot(f64[2,2]{1,0} %v_0.1, f64[2,2]{1,0} %v_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %tuple.4 = (f64[2,2]{1,0}) tuple(f64[2,2]{1,0} %dot.3)
    }
    (Tensor ((19 22) (43 50)) (dims (2 2)) (type_ Float))
    |}]
;;

let%expect_test "jit and sum" =
  Staged.unstage
    (jit' ~print_hlo:true ~f:(fun x -> Value.sum x) ())
    (Value.of_tensor (Tensor.of_list2_exn Float [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.9, entry_computation_layout={(f64[2,2]{1,0})->(f64[])}

    %sum.3 (x.4: f64[], y.5: f64[]) -> f64[] {
      %x.4 = f64[] parameter(0)
      %y.5 = f64[] parameter(1)
      ROOT %add.6 = f64[] add(f64[] %x.4, f64[] %y.5)
    }

    ENTRY %xla_call.9 (v_0.1: f64[2,2]) -> (f64[]) {
      %v_0.1 = f64[2,2]{1,0} parameter(0)
      %constant.2 = f64[] constant(0)
      %reduce.7 = f64[] reduce(f64[2,2]{1,0} %v_0.1, f64[] %constant.2), dimensions={0,1}, to_apply=%sum.3
      ROOT %tuple.8 = (f64[]) tuple(f64[] %reduce.7)
    }
    (Tensor 10 Float)
    |}]
;;

let%expect_test "jit and broadcast" =
  Staged.unstage
    (jit' ~print_hlo:true ~f:(fun x -> Value.broadcast x ~dims:[: 2; 2; 2 :]) ())
    (Value.of_tensor (Tensor.of_list2_exn Float [ [ 1.; 2. ]; [ 3.; 4. ] ]))
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.4, entry_computation_layout={(f64[2,2]{1,0})->(f64[2,2,2]{2,1,0})}

    ENTRY %xla_call.4 (v_0.1: f64[2,2]) -> (f64[2,2,2]) {
      %v_0.1 = f64[2,2]{1,0} parameter(0)
      %broadcast.2 = f64[2,2,2]{2,1,0} broadcast(f64[2,2]{1,0} %v_0.1), dimensions={1,2}
      ROOT %tuple.3 = (f64[2,2,2]{2,1,0}) tuple(f64[2,2,2]{2,1,0} %broadcast.2)
    }
    (Tensor (((1 2) (3 4)) ((1 2) (3 4))) (dims (2 2 2)) (type_ Float))
    |}]
;;

let%expect_test "jit caches the compiled executable per input structure" =
  let jitted = Staged.unstage (jit' ~print_hlo:true ~f:(fun x -> Value.O.(x * x)) ()) in
  (* The first call traces and compiles, printing the HLO. *)
  let a = jitted (Value.of_float 2.) in
  [%expect
    {|
    HloModule xla_call.4, entry_computation_layout={(f64[])->(f64[])}

    ENTRY %xla_call.4 (v_0.1: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %multiply.2 = f64[] multiply(f64[] %v_0.1, f64[] %v_0.1)
      ROOT %tuple.3 = (f64[]) tuple(f64[] %multiply.2)
    }
    |}];
  (* A second call with the same input structure reuses the compiled executable - nothing
     is printed. *)
  let b = jitted (Value.of_float 5.) in
  [%expect {| |}];
  (* A different input structure is a separate cache entry, so it compiles again. *)
  let c = jitted (Value.of_tensor (Tensor.of_list Float [ 1.; 2. ])) in
  [%expect
    {|
    HloModule xla_call.4, entry_computation_layout={(f64[2]{0})->(f64[2]{0})}

    ENTRY %xla_call.4 (v_0.1: f64[2]) -> (f64[2]) {
      %v_0.1 = f64[2]{0} parameter(0)
      %multiply.2 = f64[2]{0} multiply(f64[2]{0} %v_0.1, f64[2]{0} %v_0.1)
      ROOT %tuple.3 = (f64[2]{0}) tuple(f64[2]{0} %multiply.2)
    }
    |}];
  print_s [%message (a : Value.t) (b : Value.t) (c : Value.t)];
  [%expect
    {|
    ((a (Tensor 4 Float)) (b (Tensor 25 Float))
     (c (Tensor (1 4) (dims (2)) (type_ Float))))
    |}]
;;

let%expect_test "distinct constants are fed as separate runtime parameters" =
  (* [x * 3 + 5] hoists two constants, so the HLO has two const parameters. Each must
     receive its own value: for x = 2 the result is 11, whereas feeding the constants in a
     swapped order ([x * 5 + 3]) would give 13. *)
  Staged.unstage
    (jit'
       ~print_hlo:true
       ~f:(fun x -> Value.O.((x * Value.of_float 3.) + Value.of_float 5.))
       ())
    (Value.of_float 2.)
  |> [%sexp_of: Value.t]
  |> print_s;
  [%expect
    {|
    HloModule xla_call.7, entry_computation_layout={(f64[],f64[],f64[])->(f64[])}

    ENTRY %xla_call.7 (v_0.1: f64[], c_0.2: f64[], c_1.3: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %c_0.2 = f64[] parameter(1)
      %multiply.4 = f64[] multiply(f64[] %v_0.1, f64[] %c_0.2)
      %c_1.3 = f64[] parameter(2)
      %add.5 = f64[] add(f64[] %multiply.4, f64[] %c_1.3)
      ROOT %tuple.6 = (f64[]) tuple(f64[] %add.5)
    }
    (Tensor 11 Float)
    |}]
;;

let%expect_test "one executable serves a closure whose captured constant changes" =
  let c = ref 3. in
  let jitted =
    Staged.unstage (jit' ~print_hlo:true ~f:(fun x -> Value.O.(x * Value.of_float !c)) ())
  in
  (* The first call compiles (HLO printed), feeding the current constant as a parameter. *)
  let r1 = jitted (Value.of_float 2.) in
  [%expect
    {|
    HloModule xla_call.5, entry_computation_layout={(f64[],f64[])->(f64[])}

    ENTRY %xla_call.5 (v_0.1: f64[], c_0.2: f64[]) -> (f64[]) {
      %v_0.1 = f64[] parameter(0)
      %c_0.2 = f64[] parameter(1)
      %multiply.3 = f64[] multiply(f64[] %v_0.1, f64[] %c_0.2)
      ROOT %tuple.4 = (f64[]) tuple(f64[] %multiply.3)
    }
    |}];
  (* The captured constant changes, but the structure does not: the executable is reused
     (no HLO) and the new value is fed in - so r1 uses 3 (= 6) and r2 uses 10 (= 20). *)
  c := 10.;
  let r2 = jitted (Value.of_float 2.) in
  [%expect {| |}];
  print_s [%message (r1 : Value.t) (r2 : Value.t)];
  [%expect {| ((r1 (Tensor 6 Float)) (r2 (Tensor 20 Float))) |}]
;;

let%expect_test "a cache hit feeds the call's own (multiple) constants, in order" =
  let c1 = ref 3.
  and c2 = ref 5. in
  let jitted =
    Staged.unstage
      (jit' ~f:(fun x -> Value.O.((x * Value.of_float !c1) + Value.of_float !c2)) ())
  in
  let r1 = jitted (Value.of_float 2.) in
  c1 := 10.;
  c2 := 100.;
  (* The reused executable is fed [c1 = 10], [c2 = 100] in the right slots: 2*10 + 100 =
     120 (a swapped order would give 2*100 + 10 = 210). *)
  let r2 = jitted (Value.of_float 2.) in
  print_s [%message (r1 : Value.t) (r2 : Value.t)];
  [%expect {| ((r1 (Tensor 11 Float)) (r2 (Tensor 120 Float))) |}]
;;

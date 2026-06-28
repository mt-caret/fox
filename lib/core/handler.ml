open! Core
module Partial_value = Partial_eval.Partial_value

let partially_apply_expr_flat = Partial_eval.partially_apply_expr_flat

let flatten_function
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ~(f : in_ -> out)
  ~in_tree_def
  ~here
  =
  let out_tree_def = Set_once.create () in
  ( Staged.stage (fun values ->
      let in_tree = Value_tree.unflatten values ~def:in_tree_def in
      let out = f (In.t_of_tree in_tree) in
      let out_tree = Out.tree_of_t out in
      Set_once.set_exn out_tree_def ~here (Value_tree.to_def out_tree);
      Value_tree.flatten out_tree)
  , out_tree_def )
;;

let eval ~f =
  Fox_effect.handle ~f ~handle:(fun op ->
    Op.map op ~f:Value.to_tensor_exn
    |> Op.eval (module Tensor : Operators_intf.S with type t = Tensor.t)
    |> Value.of_tensor)
;;

let jvp
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ~f
  ~(primals : in_)
  ~(tangents : in_)
  =
  let jvp = Jvp.create () in
  let primals_tree, tangents_tree = In.tree_of_t primals, In.tree_of_t tangents in
  let primals_tree_def, tangents_tree_def =
    Value_tree.to_def primals_tree, Value_tree.to_def tangents_tree
  in
  [%test_eq: Value_tree.Def.t] primals_tree_def tangents_tree_def;
  let inputs =
    List.zip_exn (Value_tree.flatten primals_tree) (Value_tree.flatten tangents_tree)
    |> List.map ~f:(fun (primal, tangent) ->
      Jvp.Dual_number.to_value (Jvp.dual_number jvp ~primal ~tangent:(Some tangent)))
  in
  let f, out_tree_def =
    flatten_function
      (module In)
      (module Out)
      ~f
      ~in_tree_def:primals_tree_def
      ~here:[%here]
  in
  let f = Staged.unstage f in
  let primals, tangents =
    Jvp.handle jvp ~f:(fun () -> f inputs)
    |> List.map ~f:(Jvp.lift jvp)
    |> List.map ~f:(fun dual_number ->
      ( Jvp.Dual_number.primal dual_number
      , Option.value_exn
          ~message:"None tangent not supported in jvp"
          (Jvp.Dual_number.tangent dual_number) ))
    |> List.unzip
  in
  let out_tree_def = Set_once.get_exn out_tree_def in
  ( Out.t_of_tree (Value_tree.unflatten primals ~def:out_tree_def)
  , Out.t_of_tree (Value_tree.unflatten tangents ~def:out_tree_def) )
;;

let jvp' ~f ~primal ~tangent =
  jvp (module Value) (module Value) ~f ~primals:primal ~tangents:tangent
;;

let derivative ~f ~x =
  let (_primal : Value.t), tangent = jvp' ~f ~primal:x ~tangent:(Value.of_float 1.) in
  tangent
;;

let rec nth_order_derivative ~n ~f ~x =
  match n with
  | 0 -> f x
  | _ -> derivative ~f:(fun x -> nth_order_derivative ~n:(n - 1) ~f ~x) ~x
;;

(* TODO: could [Expr.t] instead be something like [(in_, out) Expr.t], storing the modules
   internally? *)
let build_expr
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ~f
  ~in_tree_def
  =
  let staging = Staging.create () in
  let parameters =
    Value_tree.Def.flatten in_tree_def
    |> List.map ~f:(fun dims ->
      (* TODO: support arbitrary types here. *)
      Staging.fresh_var staging ~shape:{ dims; type_ = T Float })
  in
  let f, out_tree_def =
    flatten_function (module In) (module Out) ~f ~in_tree_def ~here:[%here]
  in
  let f = Staged.unstage f in
  let result =
    Staging.handle staging ~f:(fun () ->
      List.map parameters ~f:(fun parameter ->
        Value.create
          ~value:parameter
          ~type_id:Expr.Var.type_id
          ~shape:(Expr.Var.shape parameter))
      |> f)
  in
  let return_vals =
    Nonempty_list.of_list_exn result
    |> Nonempty_list.map ~f:(Staging.intern_value staging)
  in
  Expr.create
    ~parameters
    ~consts:(Staging.consts_map staging)
    ~equations:(Staging.equations staging)
    ~return_vals
    ~out_tree_def:(Set_once.get_exn out_tree_def)
;;

let build_expr' ~f ~in_dims =
  build_expr (module Value) (module Value) ~f ~in_tree_def:(Value.tree_def ~dims:in_dims)
;;

let eval_expr_flat (expr : Value.t Expr.t) (input : Value.t list) =
  let eval_atom (atom : Value.t Expr.Atom.t) ~env =
    match atom with
    | Var var -> Map.find_exn env var
    | Value value -> value
  in
  let env =
    List.zip_exn expr.parameters input
    |> Expr.Var.Map.of_alist_exn
    |> Map.merge_disjoint_exn expr.consts
  in
  let env =
    List.fold expr.equations ~init:env ~f:(fun env eq ->
      let result = Op.map eq.op ~f:(eval_atom ~env) |> Op.eval (module Value) in
      Map.add_exn env ~key:eq.var ~data:result)
  in
  Nonempty_list.map expr.return_vals ~f:(eval_atom ~env) |> Nonempty_list.to_list
;;

let eval_expr
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  (expr : Value.t Expr.t)
  (input : in_)
  : out
  =
  In.tree_of_t input
  |> Value_tree.flatten
  |> eval_expr_flat expr
  |> Value_tree.unflatten ~def:expr.out_tree_def
  |> Out.t_of_tree
;;

let eval_expr' = eval_expr (module Value) (module Value)

let linearize
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ~(f : in_ -> out)
  ~(primals : in_)
  =
  let primals_tree = In.tree_of_t primals in
  let primals_tree_def = Value_tree.to_def primals_tree in
  let primals =
    Value_tree.flatten primals_tree
    |> List.map ~f:(fun value -> Partial_value.Known value)
  in
  let primals_length = List.length primals in
  let inputs =
    List.append
      primals
      (List.mapi primals ~f:(fun i primal ->
         Partial_value.Unknown
           { name = [%string "a_%{i#Int}"]; shape = Partial_value.shape primal }))
  in
  let outputs, expr =
    partially_apply_expr_flat inputs ~f:(fun inputs ->
      let primals, tangents = List.split_n inputs primals_length in
      let out_primal, out_tangent =
        jvp
          (module In)
          (module Out)
          ~f
          ~primals:(Value_tree.unflatten primals ~def:primals_tree_def |> In.t_of_tree)
          ~tangents:(Value_tree.unflatten tangents ~def:primals_tree_def |> In.t_of_tree)
      in
      let out_primal_tree = Out.tree_of_t out_primal in
      let out_tree_def = Value_tree.to_def out_primal_tree in
      let out_tangent_tree = Out.tree_of_t out_tangent in
      [%test_eq: Value_tree.Def.t] out_tree_def (Value_tree.to_def out_tangent_tree);
      ( List.append
          (Value_tree.flatten out_primal_tree)
          (Value_tree.flatten out_tangent_tree)
      , out_tree_def ))
  in
  let outputs = List.take outputs (List.length outputs / 2) in
  let output =
    List.filter_map outputs ~f:(function
      | Partial_value.Known value -> Some value
      | Unknown _ ->
        raise_s
          [%message
            "unexpected unknown primal"
              (outputs : Partial_value.t list)
              ~expr:(Expr.to_string_hum expr ~value_to_string:Value.to_string)])
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  let f_lin (tangents : in_) =
    In.tree_of_t tangents
    |> Value_tree.flatten
    |> eval_expr_flat expr
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  output, f_lin
;;

let linearize' ~f ~primals = linearize (module Value) (module Value) ~f ~primals

let eval_expr_transposed (expr : Value.t Expr.t) args ~cotangents =
  let accum_gradient ~ct_env var value =
    Map.update ct_env var ~f:(function
      | None -> value
      | Some existing -> Value.O.(existing + value))
  in
  let read_gradient ~ct_env var =
    (* TODO: some sort of type inference / add a new variant for "zero"? *)
    Map.find ct_env var
    |> Option.value_or_thunk ~default:(fun () ->
      Tensor.Typed.zeros ~dims:(Expr.Var.dims var) |> Value.of_typed_tensor)
  in
  let ct_env =
    List.zip_exn (Nonempty_list.to_list expr.return_vals) cotangents
    |> List.fold ~init:Expr.Var.Map.empty ~f:(fun ct_env (return_val, cotangent) ->
      match return_val with
      | Value _ ->
        (* TODO: do we actually want to just ignore constnats? *)
        raise_s
          [%message "unexpected const return value" (return_val : Value.t Expr.Atom.t)]
      | Var var -> accum_gradient ~ct_env var cotangent)
  in
  let ct_env =
    List.rev expr.equations
    |> List.fold ~init:ct_env ~f:(fun ct_env { var; op } ->
      let cotangent = read_gradient ~ct_env var in
      let ct_env =
        match op with
        | Unary (Neg, Var var) -> accum_gradient ~ct_env var (Value.neg cotangent)
        | Unary (Sin, Var var) -> accum_gradient ~ct_env var (Value.cos cotangent)
        | Unary (Cos, Var var) ->
          accum_gradient ~ct_env var (Value.neg (Value.sin cotangent))
        | Unary (Exp, Var var) -> accum_gradient ~ct_env var (Value.exp cotangent)
        | Binary (Add, Var var, Value _) | Binary (Add, Value _, Var var) ->
          accum_gradient ~ct_env var cotangent
        | Binary (Add, Var v1, Var v2) ->
          let ct_env = accum_gradient ~ct_env v1 cotangent in
          accum_gradient ~ct_env v2 cotangent
        | Binary (Sub, Var var, Value _) -> accum_gradient ~ct_env var cotangent
        | Binary (Sub, Value _, Var var) ->
          accum_gradient ~ct_env var (Value.neg cotangent)
        | Binary (Sub, Var v1, Var v2) ->
          let ct_env = accum_gradient ~ct_env v1 cotangent in
          accum_gradient ~ct_env v2 (Value.neg cotangent)
        | Binary (Mul, Var var, Value v) | Binary (Mul, Value v, Var var) ->
          accum_gradient ~ct_env var (Value.mul v cotangent)
        | Binary (Div, Var var, Value v) ->
          accum_gradient ~ct_env var (Value.div cotangent v)
        | Matmul (Var var, Value v) ->
          (* [y = x @ v]. For a matrix [v] this is [x_ct = y_ct @ v^T]. When [v] is a
             vector [m] the forward op is [x[n,m] @ v[m] = y[n]], whose transpose is the
             outer product [x_ct[n,m] = y_ct[n] (x) v[m]]; [transpose] only handles rank
             2, so express it as [y_ct[n,1] @ v[1,m]] instead. *)
          let grad =
            match Value.dims v with
            | [: m :] ->
              let n = (Value.dims cotangent).:(0) in
              Value.matmul
                (Value.reshape cotangent ~dims:[: n; 1 :])
                (Value.reshape v ~dims:[: 1; m :])
            | _ -> Value.matmul cotangent (Value.transpose v)
          in
          accum_gradient ~ct_env var grad
        | Matmul (Value v, Var var) ->
          accum_gradient ~ct_env var (Value.matmul (Value.transpose v) cotangent)
        | Transpose (Var var) -> accum_gradient ~ct_env var (Value.transpose cotangent)
        | Sum { value = Var var; dims; keep_dims } ->
          let var_shape = Expr.Var.shape var in
          (match keep_dims with
           | true -> cotangent
           | false ->
             (* When dims aren't kept, there are situations where broadcasting to the
                input dimension doesn't work e.g. a sum s.t. [ 2; 3 ] -> [ 2 ] *)
             let shape_if_dims_were_kept =
               Op.infer_shape_exn (Op.Sum { value = var_shape; dims; keep_dims = true })
             in
             Value.reshape cotangent ~dims:(Shape.dims shape_if_dims_were_kept))
          |> Value.broadcast ~dims:(Shape.dims var_shape)
          |> accum_gradient ~ct_env var
        | Broadcast { value = Var var; dims = to_dims } ->
          let from_dims = Expr.Var.dims var in
          let padding_length = Iarray.length to_dims - Iarray.length from_dims in
          let non_padded_broadcasts =
            Iarray.sub to_dims ~pos:padding_length ~len:(Iarray.length from_dims)
            |> Iarray.zip_exn from_dims
            |> Iarray.to_list
            |> List.filter_mapi ~f:(fun i (from, to_) ->
              if from <> to_ then Some i else None)
          in
          let unpadded_cotangent =
            match padding_length with
            | 0 -> cotangent
            | _ ->
              Value.sum
                cotangent
                ~dims:(`Just (Nonempty_list.init padding_length ~f:Fn.id))
                ~keep_dims:false
          in
          (match Nonempty_list.of_list non_padded_broadcasts with
           | None -> unpadded_cotangent
           | Some non_padded_broadcasts ->
             Value.sum
               unpadded_cotangent
               ~dims:(`Just non_padded_broadcasts)
               ~keep_dims:true)
          |> accum_gradient ~ct_env var
        | Reshape { value = Var var; dims = _ } ->
          Value.reshape cotangent ~dims:(Expr.Var.dims var) |> accum_gradient ~ct_env var
        | Unary ((Neg | Sin | Cos | Sqrt | Exp | Log | Sigmoid), _)
        | Binary ((Add | Sub | Mul | Div | Eq | Gt | Lt), _, _)
        | Matmul _ | Transpose _ | Sum _ | Broadcast _ | Reshape _ ->
          raise_s
            [%message
              "Invalid var/val op combination"
                (op : Value.t Expr.Atom.t Op.t)
                ~expr:(Expr.to_string_hum expr ~value_to_string:Value.to_string)]
      in
      ct_env)
  in
  List.map args ~f:(read_gradient ~ct_env)
;;

let vjp
  (type in_ out)
  (module In : Treeable_intf.S with type t = in_)
  (module Out : Treeable_intf.S with type t = out)
  ~(f : in_ -> out)
  ~(primals : in_)
  =
  let primals_tree = In.tree_of_t primals in
  let primals_tree_def = Value_tree.to_def primals_tree in
  let primals =
    Value_tree.flatten primals_tree
    |> List.map ~f:(fun value -> Partial_value.Known value)
  in
  let primals_length = List.length primals in
  let tangent_vars =
    List.mapi primals ~f:(fun i primal ->
      { Expr.Var.name = [%string "a_%{i#Int}"]; shape = Partial_value.shape primal })
  in
  let inputs =
    List.append primals (List.map tangent_vars ~f:(fun var -> Partial_value.Unknown var))
  in
  let outputs, expr =
    partially_apply_expr_flat inputs ~f:(fun inputs ->
      let primals, tangents = List.split_n inputs primals_length in
      let out_primal, out_tangent =
        jvp
          (module In)
          (module Out)
          ~f
          ~primals:(Value_tree.unflatten primals ~def:primals_tree_def |> In.t_of_tree)
          ~tangents:(Value_tree.unflatten tangents ~def:primals_tree_def |> In.t_of_tree)
      in
      let out_primal_tree = Out.tree_of_t out_primal in
      let out_tree_def = Value_tree.to_def out_primal_tree in
      let out_tangent_tree = Out.tree_of_t out_tangent in
      [%test_eq: Value_tree.Def.t] out_tree_def (Value_tree.to_def out_tangent_tree);
      ( List.append
          (Value_tree.flatten out_primal_tree)
          (Value_tree.flatten out_tangent_tree)
      , out_tree_def ))
  in
  let outputs = List.take outputs (List.length outputs / 2) in
  let output =
    List.filter_map outputs ~f:(function
      | Partial_value.Known value -> Some value
      | Unknown _ ->
        raise_s
          [%message
            "unexpected unknown primal"
              (outputs : Partial_value.t list)
              ~expr:(Expr.to_string_hum expr ~value_to_string:Value.to_string)])
    |> Value_tree.unflatten ~def:expr.out_tree_def
    |> Out.t_of_tree
  in
  let f_vjp (cotangents : out) =
    match
      let cotangents_tree = Out.tree_of_t cotangents in
      [%test_result: Value_tree.Def.t]
        (Value_tree.to_def cotangents_tree)
        ~expect:expr.out_tree_def;
      eval_expr_transposed
        expr
        tangent_vars
        ~cotangents:(Value_tree.flatten cotangents_tree)
      |> Value_tree.unflatten ~def:primals_tree_def
      |> In.t_of_tree
    with
    | in_ -> in_
    | exception exn ->
      Exn.reraise
        exn
        (Sexp.to_string_hum
           [%message
             (exn : exn) ~expr:(Expr.to_string_hum expr ~value_to_string:Value.to_string)])
  in
  output, f_vjp
;;

let vjp' ~f ~primal = vjp (module Value) (module Value) ~f ~primals:primal

let grad_and_value
  (type in_)
  (module In : Treeable_intf.S with type t = in_)
  ~(f : in_ -> Value.t)
  ~x
  =
  let y, f_vjp = vjp (module In) (module Value) ~f ~primals:x in
  y, f_vjp (Value.of_float 1.)
;;

let grad_and_value' ~f ~x = grad_and_value (module Value) ~f ~x

let grad module_ ~f ~x =
  let _y, grad = grad_and_value module_ ~f ~x in
  grad
;;

let grad' ~f ~x = grad (module Value) ~f ~x

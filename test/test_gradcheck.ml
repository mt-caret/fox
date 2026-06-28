open! Core
open! Fox_core

(* Finite-difference gradient checks: compare the analytic reverse-mode gradient ([grad'],
   run on the eager backend) against a central-difference numerical gradient. This
   validates the vjp transpose rules independently of the forward-mode jvp rules. *)

let numerical_grad ?(h = 1e-4) ~f x =
  let xt = Value.to_typed_tensor_exn Float x in
  let dims = Tensor.Typed.dims xt in
  Tensor.init Float ~dims ~f:(fun index ->
    let f_at delta =
      let xt' =
        Tensor.Typed.mapi Float xt ~f:(fun i v ->
          if [%equal: int iarray] i index then v +. delta else v)
      in
      eval ~f:(fun () -> f (Value.of_typed_tensor xt')) |> Value.to_float_exn
    in
    (f_at h -. f_at (-.h)) /. (2. *. h))
;;

let max_abs_diff a b =
  let flat t = Tensor.reshape t ~dims:[: Tensor.length t :] in
  let a = flat a
  and b = flat b in
  List.init (Tensor.length a) ~f:(fun i ->
    Float.abs (Tensor.get_exn Float a [: i :] -. Tensor.get_exn Float b [: i :]))
  |> List.max_elt ~compare:Float.compare
  |> Option.value ~default:0.
;;

let close ~atol ~rtol a b =
  [%equal: int iarray] (Tensor.dims a) (Tensor.dims b)
  &&
  let flat t = Tensor.reshape t ~dims:[: Tensor.length t :] in
  let a = flat a
  and b = flat b in
  List.for_all
    (List.init (Tensor.length a) ~f:Fn.id)
    ~f:(fun i ->
      let x = Tensor.get_exn Float a [: i :] in
      let y = Tensor.get_exn Float b [: i :] in
      Float.( <= ) (Float.abs (x -. y)) (atol +. (rtol *. Float.abs y)))
;;

(* Prints the name and whether analytic == numerical. On mismatch (or a raised exception),
   prints the offending values so the diff says what broke. *)
let check ~name ~f x =
  match
    Or_error.try_with (fun () ->
      let analytic = eval ~f:(fun () -> grad' ~f ~x) |> Value.to_tensor_exn in
      let numerical = numerical_grad ~f x in
      analytic, numerical)
  with
  | Error e ->
    print_s [%message name ~grad_check:"RAISED" ~error:(Error.to_string_hum e : string)]
  | Ok (analytic, numerical) ->
    let ok = close ~atol:1e-3 ~rtol:1e-2 analytic numerical in
    print_s [%message name (ok : bool)];
    if not ok
    then
      print_s
        [%message
          "  mismatch"
            (analytic : Tensor.t)
            (numerical : Tensor.t)
            ~max_abs_diff:(max_abs_diff analytic numerical : float)]
;;

let vec l = Value.of_tensor (Tensor.of_list Float l)
let mat l = Value.of_tensor (Tensor.of_list2_exn Float l)
let m23 = mat [ [ 0.5; -0.3; 1.2 ]; [ 0.7; 0.9; -0.4 ] ]
let m32 = mat [ [ 0.5; -0.3 ]; [ 1.2; 0.7 ]; [ 0.9; -0.4 ] ]
let v3 = vec [ 0.5; -0.3; 1.2 ]
let sum = Value.sum ~keep_dims:false

let%expect_test "finite-difference gradient checks: unary ops" =
  check ~name:"neg" ~f:(fun x -> sum (Value.neg x)) v3;
  check ~name:"sin" ~f:(fun x -> sum (Value.sin x)) v3;
  check ~name:"cos" ~f:(fun x -> sum (Value.cos x)) v3;
  check ~name:"exp" ~f:(fun x -> sum (Value.exp x)) v3;
  check ~name:"sqrt" ~f:(fun x -> sum (Value.sqrt x)) (vec [ 0.5; 1.0; 2.0 ]);
  check ~name:"log" ~f:(fun x -> sum (Value.log x)) (vec [ 0.5; 1.0; 2.0 ]);
  check ~name:"sigmoid" ~f:(fun x -> sum (Value.sigmoid x)) v3;
  [%expect
    {|
    (neg (ok true))
    (sin (ok true))
    (cos (ok true))
    (exp (ok true))
    (sqrt (ok true))
    (log (ok true))
    (sigmoid (ok true))
    |}]
;;

let%expect_test "finite-difference gradient checks: binary ops with constants" =
  check ~name:"x + c" ~f:(fun x -> sum Value.O.(x + v3)) v3;
  check ~name:"x - c" ~f:(fun x -> sum Value.O.(x - v3)) v3;
  check ~name:"c - x" ~f:(fun x -> sum Value.O.(v3 - x)) v3;
  check ~name:"x * c" ~f:(fun x -> sum Value.O.(x * v3)) v3;
  check ~name:"x / c" ~f:(fun x -> sum Value.O.(x / v3)) v3;
  check ~name:"c / x" ~f:(fun x -> sum Value.O.(v3 / x)) (vec [ 0.5; 1.0; 2.0 ]);
  check ~name:"x * x" ~f:(fun x -> sum Value.O.(x * x)) v3;
  check ~name:"x / x" ~f:(fun x -> sum Value.O.(x / x)) v3;
  [%expect
    {|
    ("x + c" (ok true))
    ("x - c" (ok true))
    ("c - x" (ok true))
    ("x * c" (ok true))
    ("x / c" (ok true))
    ("c / x" (ok true))
    ("x * x" (ok true))
    ("x / x" (ok true))
    |}]
;;

let%expect_test "finite-difference gradient checks: compositions" =
  check ~name:"sum (sin (x*x))" ~f:(fun x -> sum (Value.sin Value.O.(x * x))) v3;
  check ~name:"sum (exp (sin x))" ~f:(fun x -> sum (Value.exp (Value.sin x))) v3;
  check
    ~name:"sum (log (x*x + c))"
    ~f:(fun x -> sum (Value.log Value.O.((x * x) + v3)))
    (vec [ 0.5; 1.0; 2.0 ]);
  check ~name:"sum (sigmoid (x + x))" ~f:(fun x -> sum (Value.sigmoid Value.O.(x + x))) v3;
  [%expect
    {|
    ("sum (sin (x*x))" (ok true))
    ("sum (exp (sin x))" (ok true))
    ("sum (log (x*x + c))" (ok true))
    ("sum (sigmoid (x + x))" (ok true))
    |}]
;;

let%expect_test "finite-difference gradient checks: structural ops" =
  check ~name:"sum all" ~f:(fun x -> sum x) m23;
  check
    ~name:"sum dims=[0] keep=false"
    ~f:(fun x -> sum (Value.sum x ~dims:(`Just [ 0 ]) ~keep_dims:false))
    m23;
  check
    ~name:"sum dims=[1] keep=false"
    ~f:(fun x -> sum (Value.sum x ~dims:(`Just [ 1 ]) ~keep_dims:false))
    m23;
  check
    ~name:"sum dims=[0] keep=true"
    ~f:(fun x -> sum (Value.sum x ~dims:(`Just [ 0 ]) ~keep_dims:true))
    m23;
  check ~name:"transpose" ~f:(fun x -> sum (Value.transpose x)) m23;
  check ~name:"sum(transpose * c)" ~f:(fun x -> sum Value.O.(Value.transpose x * m32)) m23;
  check ~name:"reshape" ~f:(fun x -> sum (Value.reshape x ~dims:[: 6 :])) m23;
  check
    ~name:"broadcast [3]->[2,3]"
    ~f:(fun x -> sum (Value.broadcast x ~dims:[: 2; 3 :]))
    v3;
  check
    ~name:"broadcast [1,3]->[2,3]"
    ~f:(fun x -> sum (Value.broadcast x ~dims:[: 2; 3 :]))
    (mat [ [ 0.5; -0.3; 1.2 ] ]);
  check ~name:"matmul x@c (x=[2,3])" ~f:(fun x -> sum (Value.matmul x m32)) m23;
  check ~name:"matmul c@x (x=[3,2])" ~f:(fun x -> sum (Value.matmul m23 x)) m32;
  (* matmul against a rank-1 (vector) operand: the matrix is the differentiated variable,
     so the transpose rule must form an outer product rather than transpose the vector. *)
  check
    ~name:"matmul x[2,3]@vec[3] (matrix var)"
    ~f:(fun x -> sum (Value.matmul x v3))
    m23;
  check ~name:"matmul c[2,3]@v[3] (vector var)" ~f:(fun v -> sum (Value.matmul m23 v)) v3;
  check ~name:"mean" ~f:(fun x -> Value.mean x) m23;
  [%expect
    {|
    ("sum all" (ok true))
    ("sum dims=[0] keep=false" (ok true))
    ("sum dims=[1] keep=false" (ok true))
    ("sum dims=[0] keep=true" (ok true))
    (transpose (ok true))
    ("sum(transpose * c)" (ok true))
    (reshape (ok true))
    ("broadcast [3]->[2,3]" (ok true))
    ("broadcast [1,3]->[2,3]" (ok true))
    ("matmul x@c (x=[2,3])" (ok true))
    ("matmul c@x (x=[3,2])" (ok true))
    ("matmul x[2,3]@vec[3] (matrix var)" (ok true))
    ("matmul c[2,3]@v[3] (vector var)" (ok true))
    (mean (ok true))
    |}]
;;

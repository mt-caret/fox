open! Core
open! Fox_core

(* Robustness regressions surfaced by the differential / multi-agent bug hunt: two JIT
   bugs that are now fixed, and two limitations that are documented (with CR notes at the
   code) and tracked here until fixed. *)

let jit f x =
  Or_error.try_with (fun () ->
    Staged.unstage (Fox_jit.jit' ~f ()) x |> Value.to_tensor_exn)
;;

let vec l = Value.of_tensor (Tensor.of_list Float l)
let mat l = Value.of_tensor (Tensor.of_list2_exn Float l)

let%expect_test "jit: reshape with -1 resolves (regression: previously aborted the \
                 process)"
  =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  let m = mat [ [ 1.; 2.; 3. ]; [ 4.; 5.; 6. ] ] in
  print_s
    [%message
      "reshape [-1]"
        ~r:(jit (fun v -> Value.reshape v ~dims:[: -1 :]) m : Tensor.t Or_error.t)];
  print_s
    [%message
      "reshape [-1;2]"
        ~r:(jit (fun v -> Value.reshape v ~dims:[: -1; 2 :]) m : Tensor.t Or_error.t)];
  [%expect
    {|
    ("reshape [-1]" (r (Ok (1 2 3 4 5 6))))
    ("reshape [-1;2]" (r (Ok ((1 2) (3 4) (5 6)))))
    |}]
;;

let%expect_test "jit: a failed compilation does not poison later compilations" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  (* A bool constant becomes an F64 XLA parameter, so [eq (pred, f64)] fails to build (see
     the bool-constant limitation below). Before the fix this poisoned the shared global
     builder and broke every subsequent compilation; now each compile gets a fresh
     builder. *)
  let mask = Value.of_tensor (Tensor.of_list Bool [ true; false; true ]) in
  let zero3 = Value.of_tensor (Tensor.zeros ~dims:[: 3 :]) in
  let v = vec [ 1.; -2.; 3. ] in
  let failed =
    jit (fun x -> Value.O.(Value.O.(x > zero3) = mask)) v |> Or_error.is_error
  in
  print_s [%message "first compile failed (as expected)" (failed : bool)];
  print_s
    [%message
      "unrelated valid jit still works"
        ~r:(jit (fun x -> Value.O.(x * x)) v : Tensor.t Or_error.t)];
  [%expect
    {|
    ("first compile failed (as expected)" (failed true))
    ("unrelated valid jit still works" (r (Ok (1 4 9))))
    |}]
;;

let%expect_test "jit: bool constant unsupported (documented limitation)" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  (* Eager handles a returned bool constant fine; the XLA backend hard-codes constants to
     F64. Tracked by the CR note in fox_jit.ml. *)
  let mask = Value.of_tensor (Tensor.of_list Bool [ true; false; true ]) in
  let x = vec [ 1.; 2.; 3. ] in
  print_s
    [%message
      "eager" ~r:(eval ~f:(fun () -> (fun _ -> mask) x) |> Value.to_tensor_exn : Tensor.t)];
  print_s [%message "jit" ~r:(jit (fun _ -> mask) x : Tensor.t Or_error.t)];
  [%expect
    {|
    (eager (r (true false true)))
    (jit
     (r
      (Error
       ("Type_equal.Id.same_witness_exn got different ids"
        ((Tensor Float) (Tensor Bool))))))
    |}]
;;

let%expect_test "grad of an output independent of the input (documented limitation)" =
  (* Forward mode yields a zero tangent, so the gradient should be zero; reverse mode
     instead raises because the traced tangent program has no input-dependent outputs.
     Tracked by the CR note in handler.ml. *)
  let jvp_tangent =
    eval ~f:(fun () ->
      jvp'
        ~f:(fun _x -> Value.of_float 5.)
        ~primal:(Value.of_float 3.)
        ~tangent:(Value.of_float 1.)
      |> snd)
    |> Value.to_tensor_exn
  in
  print_s [%message "jvp tangent (correct: 0)" (jvp_tangent : Tensor.t)];
  let grad_result =
    Or_error.try_with (fun () ->
      eval ~f:(fun () -> grad' ~f:(fun _x -> Value.of_float 5.) ~x:(Value.of_float 3.))
      |> Value.to_tensor_exn)
  in
  print_s
    [%message
      "grad result (currently raises; should be 0)" (grad_result : Tensor.t Or_error.t)];
  [%expect
    {|
    ("jvp tangent (correct: 0)" (jvp_tangent 0))
    ("grad result (currently raises; should be 0)"
     (grad_result
      (Error
       ("Failed to create expr" (exn "Nonempty_list.of_list_exn: empty list")
        (inputs
         ((Known (Tensor 3 Float))
          (Unknown ((name a_0) (shape ((dims ()) (type_ Float)))))))))))
    |}]
;;

open! Core
open! Fox_core

(* Unit tests for [Op.infer_shape]: the single source of truth for shape/type checking
   that both backends rely on. *)

let infer op = Op.infer_shape op |> [%sexp_of: Shape.t Or_error.t] |> print_s
let f dims : Shape.t = { dims; type_ = T Float }
let b dims : Shape.t = { dims; type_ = T Bool }

let%expect_test "sum: reduction-dim bounds (regression: bounds check was a no-op)" =
  (* Out-of-bounds positive and negative reduction dims must be rejected, not silently
     treated as no-ops. *)
  infer (Sum { value = f [: 3 :]; dims = `Just [ 5 ]; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_dims: dims out of bounds"
      (op
       (Sum (value ((dims (3)) (type_ Float))) (dims (Just (5)))
        (keep_dims false)))))
    |}];
  infer (Sum { value = f [: 3 :]; dims = `Just [ -5 ]; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_dims: dims out of bounds"
      (op
       (Sum (value ((dims (3)) (type_ Float))) (dims (Just (-5)))
        (keep_dims false)))))
    |}];
  infer (Sum { value = f [: 2; 2 :]; dims = `Just [ 2 ]; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_dims: dims out of bounds"
      (op
       (Sum (value ((dims (2 2)) (type_ Float))) (dims (Just (2)))
        (keep_dims false)))))
    |}];
  (* In-bounds dims (incl. valid negatives) still work. *)
  infer (Sum { value = f [: 2; 3 :]; dims = `Just [ 1 ]; keep_dims = false });
  [%expect {| (Ok ((dims (2)) (type_ Float))) |}];
  infer (Sum { value = f [: 2; 3 :]; dims = `Just [ -1 ]; keep_dims = false });
  [%expect {| (Ok ((dims (2)) (type_ Float))) |}]
;;

let%expect_test "sum: dims, keep_dims, duplicates" =
  infer (Sum { value = f [: 2; 3 :]; dims = `Just [ 0; 1 ]; keep_dims = false });
  [%expect {| (Ok ((dims ()) (type_ Float))) |}];
  infer (Sum { value = f [: 2; 3 :]; dims = `Just [ 0; 1 ]; keep_dims = true });
  [%expect {| (Ok ((dims (1 1)) (type_ Float))) |}];
  (* Duplicate reduction dims are rejected. *)
  infer (Sum { value = f [: 2; 3 :]; dims = `Just [ 0; 0 ]; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_dims: Sum: duplicate reduction dimension"
      (op
       (Sum (value ((dims (2 3)) (type_ Float))) (dims (Just (0 0)))
        (keep_dims false)))))
    |}];
  (* Sign-aliased duplicates (same axis written positive and negative) are duplicates
     too - the dedup runs after negative-index normalization. *)
  infer (Sum { value = f [: 2; 3; 4 :]; dims = `Just [ 0; -3 ]; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_dims: Sum: duplicate reduction dimension"
      (op
       (Sum (value ((dims (2 3 4)) (type_ Float))) (dims (Just (0 -3)))
        (keep_dims false)))))
    |}];
  (* A distinct positive/negative mix is fine. *)
  infer (Sum { value = f [: 2; 3; 4 :]; dims = `Just [ 0; -1 ]; keep_dims = false });
  [%expect {| (Ok ((dims (3)) (type_ Float))) |}];
  infer (Sum { value = f [: 2; 3 :]; dims = `All; keep_dims = false });
  [%expect {| (Ok ((dims ()) (type_ Float))) |}];
  (* Sum of a bool tensor is unsupported. *)
  infer (Sum { value = b [: 3 :]; dims = `All; keep_dims = false });
  [%expect
    {|
    (Error
     ("infer_shape: Sum: Bool not supported"
      (op (Sum (value ((dims (3)) (type_ Bool))) (dims All) (keep_dims false)))))
    |}]
;;

let%expect_test "broadcast: rank padding and dim stretching" =
  infer (Broadcast { value = f [: 3 :]; dims = [: 2; 3 :] });
  [%expect {| (Ok ((dims (2 3)) (type_ Float))) |}];
  infer (Broadcast { value = f [: 1 :]; dims = [: 2; 3 :] });
  [%expect {| (Ok ((dims (2 3)) (type_ Float))) |}];
  infer (Broadcast { value = f [: 1; 3 :]; dims = [: 2; 3 :] });
  [%expect {| (Ok ((dims (2 3)) (type_ Float))) |}];
  (* A non-1, non-equal dim cannot be stretched. *)
  infer (Broadcast { value = f [: 2 :]; dims = [: 2; 3 :] });
  [%expect
    {|
    (Error
     ("infer_dims: can't broadcast"
      (op (Broadcast (value ((dims (2)) (type_ Float))) (dims (2 3))))))
    |}];
  (* Cannot broadcast to a smaller rank. *)
  infer (Broadcast { value = f [: 2; 3 :]; dims = [: 3 :] });
  [%expect
    {|
    (Error
     ("infer_dims: can't broadcast to a larger rank"
      (op (Broadcast (value ((dims (2 3)) (type_ Float))) (dims (3))))))
    |}];
  (* Target dims must be positive. *)
  infer (Broadcast { value = f [: 3 :]; dims = [: 0; 3 :] });
  [%expect
    {|
    (Error
     ("infer_dims: dims must be positive"
      (op (Broadcast (value ((dims (3)) (type_ Float))) (dims (0 3))))))
    |}]
;;

let%expect_test "reshape: explicit and inferred (-1) dims" =
  infer (Reshape { value = f [: 2; 3 :]; dims = [: 6 :] });
  [%expect {| (Ok ((dims (6)) (type_ Float))) |}];
  infer (Reshape { value = f [: 2; 3 :]; dims = [: -1 :] });
  [%expect {| (Ok ((dims (6)) (type_ Float))) |}];
  infer (Reshape { value = f [: 2; 3 :]; dims = [: -1; 2 :] });
  [%expect {| (Ok ((dims (3 2)) (type_ Float))) |}];
  (* More than one -1 is ambiguous. *)
  infer (Reshape { value = f [: 2; 3 :]; dims = [: -1; -1 :] });
  [%expect
    {|
    (Error
     ("infer_dims: more than one -1 in dims"
      (op (Reshape (value ((dims (2 3)) (type_ Float))) (dims (-1 -1))))))
    |}];
  (* Element count must match. *)
  infer (Reshape { value = f [: 2; 3 :]; dims = [: 4 :] });
  [%expect
    {|
    (Error
     ("infer_dims: can't reshape"
      (op (Reshape (value ((dims (2 3)) (type_ Float))) (dims (4))))))
    |}];
  (* -1 with no integer solution. *)
  infer (Reshape { value = f [: 2; 3 :]; dims = [: -1; 4 :] });
  [%expect
    {|
    (Error
     ("infer_dims: no valid implicit dimension"
      (op (Reshape (value ((dims (2 3)) (type_ Float))) (dims (-1 4))))))
    |}]
;;

let%expect_test "matmul: rank and contraction-dim checks" =
  infer (Matmul (f [: 2; 3 :], f [: 3; 4 :]));
  [%expect {| (Ok ((dims (2 4)) (type_ Float))) |}];
  infer (Matmul (f [: 2; 3 :], f [: 3 :]));
  [%expect {| (Ok ((dims (2)) (type_ Float))) |}];
  (* Contraction dim mismatch. *)
  infer (Matmul (f [: 2; 3 :], f [: 4 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Matmul: dims mismatch"
      (op (Matmul ((dims (2 3)) (type_ Float)) ((dims (4)) (type_ Float))))))
    |}];
  infer (Matmul (f [: 2; 3 :], f [: 4; 5 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Matmul: dims mismatch"
      (op (Matmul ((dims (2 3)) (type_ Float)) ((dims (4 5)) (type_ Float))))))
    |}];
  (* A rank-1 left operand is unsupported. *)
  infer (Matmul (f [: 3 :], f [: 3 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Matmul: unsupported matmul dimensions"
      (op (Matmul ((dims (3)) (type_ Float)) ((dims (3)) (type_ Float))))))
    |}];
  (* Rank-3 is unsupported (no batching). *)
  infer (Matmul (f [: 2; 3; 4 :], f [: 4; 5 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Matmul: unsupported matmul dimensions"
      (op (Matmul ((dims (2 3 4)) (type_ Float)) ((dims (4 5)) (type_ Float))))))
    |}];
  (* Bool operands are unsupported. *)
  infer (Matmul (b [: 2; 3 :], f [: 3; 4 :]));
  [%expect
    {|
    (Error
     ("infer_shape: Matmul: type mismatch"
      (op (Matmul ((dims (2 3)) (type_ Bool)) ((dims (3 4)) (type_ Float))))))
    |}]
;;

let%expect_test "transpose: rank-2 only" =
  infer (Transpose (f [: 2; 3 :]));
  [%expect {| (Ok ((dims (3 2)) (type_ Float))) |}];
  infer (Transpose (f [: 3 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Transpose: unsupported transpose dimensions"
      (op (Transpose ((dims (3)) (type_ Float))))))
    |}];
  infer (Transpose (f [::]));
  [%expect
    {|
    (Error
     ("infer_dims: Transpose: unsupported transpose dimensions"
      (op (Transpose ((dims ()) (type_ Float))))))
    |}];
  infer (Transpose (f [: 2; 3; 4 :]));
  [%expect
    {|
    (Error
     ("infer_dims: Transpose: unsupported transpose dimensions"
      (op (Transpose ((dims (2 3 4)) (type_ Float))))))
    |}]
;;

let%expect_test "elementwise: rank-0, dim agreement, and bool typing" =
  infer (Unary (Sin, f [::]));
  [%expect {| (Ok ((dims ()) (type_ Float))) |}];
  infer (Unary (Neg, b [: 2 :]));
  [%expect
    {|
    (Error
     ("infer_shape: Unary: Bool not supported"
      (op (Unary Neg ((dims (2)) (type_ Bool))))))
    |}];
  infer (Binary (Add, f [: 2 :], f [: 3 :]));
  [%expect
    {|
    (Error
     ("infer_dims: dims mismatch"
      (op (Binary Add ((dims (2)) (type_ Float)) ((dims (3)) (type_ Float))))))
    |}];
  (* Comparisons produce bool; eq supports bool operands but gt/lt do not. *)
  infer (Binary (Gt, f [: 2 :], f [: 2 :]));
  [%expect {| (Ok ((dims (2)) (type_ Bool))) |}];
  infer (Binary (Eq, b [: 2 :], b [: 2 :]));
  [%expect {| (Ok ((dims (2)) (type_ Bool))) |}];
  infer (Binary (Gt, b [: 2 :], b [: 2 :]));
  [%expect
    {|
    (Error
     ("infer_shape: Binary: type mismatch"
      (op (Binary Gt ((dims (2)) (type_ Bool)) ((dims (2)) (type_ Bool))))))
    |}];
  infer (Binary (Add, f [: 2 :], b [: 2 :]));
  [%expect
    {|
    (Error
     ("infer_shape: Binary: type mismatch"
      (op (Binary Add ((dims (2)) (type_ Float)) ((dims (2)) (type_ Bool))))))
    |}]
;;

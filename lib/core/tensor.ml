open! Core

type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

let dims (t : t) = Bigarray.Genarray.dims t
let num_dims (t : t) = Bigarray.Genarray.num_dims t
let length (t : t) = dims t |> Array.fold ~init:1 ~f:( * )

let item (t : t) =
  match dims t with
  | [||] -> Bigarray.Genarray.get t [||]
  | dims -> raise_s [%message "Tensor.item: dims > 0" (dims : int array)]
;;

let get (t : t) index = Bigarray.Genarray.get t index
let set (t : t) index value = Bigarray.Genarray.set t index value
let fill (t : t) value = Bigarray.Genarray.fill t value

let rec sexp_of_t t =
  match dims t with
  | [||] -> item t |> [%sexp_of: float]
  | [| n |] -> List.init n ~f:(fun i -> get t [| i |]) |> [%sexp_of: float list]
  | dims ->
    let first_dim = dims.(0) in
    List.init first_dim ~f:(fun i -> Bigarray.Genarray.slice_left t [| i |] |> sexp_of_t)
    |> [%sexp_of: Sexp.t list]
;;

(* TODO: I'm pretty sure this doesn't give you a meaningful total order, but
   that's fine, we just use it for a binary search tree. *)
let rec compare t1 t2 =
  if [%compare.equal: int array] (dims t1) (dims t2)
  then Comparable.lift [%compare: int array] ~f:dims t1 t2
  else (
    let compare =
      match dims t1 with
      | [||] -> Comparable.lift [%compare: float] ~f:item
      | [| n |] ->
        Comparable.lift [%compare: float list] ~f:(fun t ->
          List.init n ~f:(fun i -> get t [| i |]))
      | dims ->
        let first_dim = dims.(0) in
        List.init first_dim ~f:(fun i ->
          Comparable.lift compare ~f:(fun t -> Bigarray.Genarray.slice_left t [| i |]))
        |> Comparable.lexicographic
    in
    compare t1 t2)
;;

let type_id = Type_equal.Id.create ~name:"Tensor" [%sexp_of: t]

let create_uninitialized dims =
  Bigarray.Genarray.create Bigarray.float64 Bigarray.c_layout dims
;;

let reshape t ~dims = Bigarray.reshape t dims

(* TODO: write a ppx that allows writing [5t] or [5.5t] which expands to
   [Tensor.of_float (Int.to_float 5)] and [Tensor.of_float 5.5]. See
   janestreet/ppx_fixed_literal for prior art. *)
let of_float f =
  let t = create_uninitialized [||] in
  set t [||] f;
  t
;;

let%expect_test "of_float" =
  let t = of_float 5. in
  [%sexp_of: t] t |> print_s;
  [%expect {| 5 |}]
;;

let of_list l =
  let t = create_uninitialized [| List.length l |] in
  List.iteri l ~f:(fun i x -> set t [| i |] x);
  t
;;

let%expect_test "of_list" =
  let t = of_list [ 1.; 2.; 3. ] in
  [%sexp_of: t] t |> print_s;
  [%expect {| (1 2 3) |}]
;;

let of_list2_exn l =
  match List.map ~f:List.length l |> List.dedup_and_sort ~compare:Int.compare with
  | [ row_length ] ->
    let t = create_uninitialized [| List.length l; row_length |] in
    List.iteri l ~f:(fun i row -> List.iteri row ~f:(fun j x -> set t [| i; j |] x));
    t
  | row_lengths ->
    raise_s [%message "of_list2_exn: non-rectangular list" (row_lengths : int list)]
;;

let%expect_test "of_list2_exn" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  [%sexp_of: t] t |> print_s;
  [%expect {| ((1 2) (3 4)) |}]
;;

let create ~dims value =
  let t = create_uninitialized dims in
  fill t value;
  t
;;

let zeros ~dims = create ~dims 0.
let ones ~dims = create ~dims 1.

let arange n =
  let t = create_uninitialized [| n |] in
  for i = 0 to n - 1 do
    Bigarray.Genarray.set t [| i |] (Int.to_float i)
  done;
  t
;;

let%expect_test "arange" =
  arange 12 |> sexp_of_t |> print_s;
  [%expect {| (0 1 2 3 4 5 6 7 8 9 10 11) |}];
  arange 12 |> reshape ~dims:[| 6; 2 |] |> sexp_of_t |> print_s;
  [%expect {| ((0 1) (2 3) (4 5) (6 7) (8 9) (10 11)) |}];
  arange 12 |> reshape ~dims:[| 3; 4 |] |> sexp_of_t |> print_s;
  [%expect {| ((0 1 2 3) (4 5 6 7) (8 9 10 11)) |}]
;;

let map t ~f =
  match num_dims t with
  | 0 -> get t [||] |> f |> of_float
  | _ ->
    let t' = create_uninitialized [| length t |] in
    for i = 0 to length t - 1 do
      set t' [| i |] (f (get t [| i |]))
    done;
    reshape t' ~dims:(dims t)
;;

let map2 t1 t2 ~f =
  let dims1 = dims t1 in
  let dims2 = dims t2 in
  if not ([%compare.equal: int array] dims1 dims2)
  then
    raise_s
      [%message "Tensor.map2: dims mismatch" (dims1 : int array) (dims2 : int array)];
  match num_dims t1 with
  | 0 -> f (get t1 [||]) (get t2 [||]) |> of_float
  | _ ->
    let t = create_uninitialized [| length t1 |] in
    for i = 0 to length t1 - 1 do
      set t [| i |] (f (get t1 [| i |]) (get t2 [| i |]))
    done;
    reshape t ~dims:(dims t1)
;;

let add t1 t2 = map2 t1 t2 ~f:( +. )
let sub t1 t2 = map2 t1 t2 ~f:( -. )
let mul t1 t2 = map2 t1 t2 ~f:( *. )
let neg t = map t ~f:( ~-. )
let sin t = map t ~f:Float.sin
let cos t = map t ~f:Float.cos

(* TODO: support more than just 2D tensors for matmuls and transposes *)
let matmul t1 t2 =
  match dims t1, dims t2 with
  | [| n; m |], [| m'; k |] ->
    [%test_eq: int] m m';
    let t = create_uninitialized [| n; k |] in
    for i = 0 to n - 1 do
      for j = 0 to k - 1 do
        let acc = ref 0. in
        for l = 0 to m - 1 do
          acc := !acc +. (get t1 [| i; l |] *. get t2 [| l; j |])
        done;
        set t [| i; j |] !acc
      done
    done;
    t
  | t1_dims, t2_dims ->
    raise_s
      [%message
        "matmul: unsupported dimensions" (t1_dims : int array) (t2_dims : int array)]
;;

let%expect_test "matmul" =
  let t1 = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  let t2 = of_list2_exn [ [ 5.; 6. ]; [ 7.; 8. ] ] in
  matmul t1 t2 |> sexp_of_t |> print_s;
  [%expect {| ((19 22) (43 50)) |}]
;;

let transpose t =
  match dims t with
  | [| n; m |] ->
    let t' = create_uninitialized [| m; n |] in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        set t' [| j; i |] (get t [| i; j |])
      done
    done;
    t'
  | dims -> raise_s [%message "transpose: unsupported dimensions" (dims : int array)]
;;

let%expect_test "transpose" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  transpose t |> sexp_of_t |> print_s;
  [%expect {| ((1 3) (2 4)) |}]
;;

module O = struct
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul
  let ( ~- ) = neg
end

module With_shape = struct
  type nonrec t = t

  let sexp_of_t t = [%sexp { dims : int array = dims t; tensor : t = t }]
end

module Private = struct
  let to_bigarray = Fn.id
  let of_bigarray = Fn.id
end

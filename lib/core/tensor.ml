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

let init ~dims ~f = Bigarray.Genarray.init Bigarray.float64 Bigarray.c_layout dims f
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

let mapi t ~f = init ~dims:(dims t) ~f:(fun index -> f index (get t index))
let map t ~f = mapi t ~f:(fun _index value -> f value)

let map2 t1 t2 ~f =
  let dims1 = dims t1 in
  let dims2 = dims t2 in
  if not ([%compare.equal: int array] dims1 dims2)
  then
    raise_s
      [%message "Tensor.map2: dims mismatch" (dims1 : int array) (dims2 : int array)];
  init ~dims:dims1 ~f:(fun index -> f (get t1 index) (get t2 index))
;;

let iteri t ~f =
  (* TODO: figure out a way to iterate over indices and get rid of this hack. *)
  let (_ : t) =
    mapi t ~f:(fun index value ->
      f index value;
      0.)
  in
  ()
;;

let iter t ~f = iteri t ~f:(fun _index value -> f value)

let sum_single_axis t ~axis ~keep_dim =
  let dims = dims t in
  let dims_length = Array.length dims in
  if axis < 0 || axis >= dims_length
  then raise_s [%message "sum_single_axis: axis out of bounds" (axis : int)];
  let dims_left = Array.subo dims ~len:axis in
  let dims_right = Array.subo dims ~pos:(axis + 1) ~len:(dims_length - axis - 1) in
  let result_dims =
    Array.concat [ dims_left; (if keep_dim then [| 1 |] else [||]); dims_right ]
  in
  init ~dims:result_dims ~f:(fun index ->
    let index =
      if keep_dim
      then Array.copy index
      else
        Array.init dims_length ~f:(fun i ->
          match Ordering.of_int (Int.compare i axis) with
          | Less -> index.(i)
          | Equal -> 0
          | Greater -> index.(i - 1))
    in
    let acc = ref 0. in
    for i = 0 to dims.(axis) - 1 do
      index.(axis) <- i;
      acc := !acc +. get t index
    done;
    !acc)
;;

let%expect_test "sum_single_axis" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  sum_single_axis t ~axis:0 ~keep_dim:false |> sexp_of_t |> print_s;
  [%expect {| (4 6) |}];
  sum_single_axis t ~axis:0 ~keep_dim:true |> sexp_of_t |> print_s;
  [%expect {| ((4 6)) |}];
  sum_single_axis t ~axis:1 ~keep_dim:false |> sexp_of_t |> print_s;
  [%expect {| (3 7) |}];
  sum_single_axis t ~axis:1 ~keep_dim:true |> sexp_of_t |> print_s;
  [%expect {| ((3) (7)) |}]
;;

include Op.Make_operators (struct
    type value = t [@@deriving sexp_of]

    let of_float = of_float

    let eval : t Op.t -> t = function
      | Unary (kind, t) ->
        let f =
          match kind with
          | Neg -> Float.neg
          | Sin -> Float.sin
          | Cos -> Float.cos
          | Sqrt -> Float.sqrt
        in
        map t ~f
      | Binary (kind, t1, t2) ->
        let f =
          match kind with
          | Add -> ( +. )
          | Sub -> ( -. )
          | Mul -> ( *. )
          | Div -> ( /. )
        in
        map2 t1 t2 ~f
      | Matmul (t1, t2) ->
        (* TODO: support more than just 2D tensors for matmuls and transposes *)
        (match dims t1, dims t2 with
         | [| n; m |], [| m' |] ->
           [%test_eq: int] m m';
           let t = create_uninitialized [| n |] in
           for i = 0 to n - 1 do
             let acc = ref 0. in
             for l = 0 to m - 1 do
               acc := !acc +. (get t1 [| i; l |] *. get t2 [| l |])
             done;
             set t [| i |] !acc
           done;
           t
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
               "matmul: unsupported dimensions"
                 (t1_dims : int array)
                 (t2_dims : int array)])
      | Transpose t ->
        (match dims t with
         | [| n; m |] ->
           init ~dims:[| m; n |] ~f:(fun index -> get t [| index.(1); index.(0) |])
         | dims ->
           raise_s [%message "transpose: unsupported dimensions" (dims : int array)])
      | Sum { value = t; dims = dims_to_sum; keep_dims } ->
        let dims = dims t in
        let dims_length = Array.length dims in
        let dims_to_sum =
          (match dims_to_sum with
           | `Just dims_to_sum ->
             Nonempty_list.map dims_to_sum ~f:(fun dim ->
               if dim < 0 then dims_length + dim else dim)
             |> Nonempty_list.to_list
           | `All -> List.range 0 dims_length)
          |> List.sort ~compare:(Comparable.reverse Int.compare)
        in
        List.fold dims_to_sum ~init:t ~f:(fun t axis ->
          sum_single_axis t ~axis ~keep_dim:keep_dims)
      | Broadcast { value = t; dims = to_dims } ->
        let from_dims = dims t in
        let dims_padding_length = Array.length to_dims - Array.length from_dims in
        init ~dims:to_dims ~f:(fun index ->
          let from_index =
            Array.subo index ~pos:dims_padding_length ~len:(Array.length from_dims)
            |> Array.map2_exn from_dims ~f:(fun from_dim index_dim ->
              if from_dim = 1 then 0 else index_dim)
          in
          get t from_index)
    ;;

    let dims = dims
  end)

let%expect_test "matmul" =
  let t1 = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  let t2 = of_list2_exn [ [ 5.; 6. ]; [ 7.; 8. ] ] in
  matmul t1 t2 |> sexp_of_t |> print_s;
  [%expect {| ((19 22) (43 50)) |}];
  let t2 = of_list [ 5.; 6. ] in
  matmul t1 t2 |> sexp_of_t |> print_s;
  [%expect {| (17 39) |}]
;;

let%expect_test "transpose" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  transpose t |> sexp_of_t |> print_s;
  [%expect {| ((1 3) (2 4)) |}]
;;

let%expect_test "sum" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  sum t ~keep_dims:false |> sexp_of_t |> print_s;
  [%expect {| 10 |}];
  sum t ~keep_dims:true |> sexp_of_t |> print_s;
  [%expect {| ((10)) |}]
;;

let%expect_test "mean" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  mean t ~keep_dims:false |> sexp_of_t |> print_s;
  [%expect {| 2.5 |}];
  mean t ~keep_dims:true |> sexp_of_t |> print_s;
  [%expect {| ((2.5)) |}]
;;

let%expect_test "var" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  var t ~keep_dims:false |> sexp_of_t |> print_s;
  [%expect {| 1.6666666666666667 |}];
  var t ~keep_dims:true |> sexp_of_t |> print_s;
  [%expect {| ((1.6666666666666667)) |}]
;;

let%expect_test "std" =
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  std t ~keep_dims:false |> sexp_of_t |> print_s;
  [%expect {| 1.2909944487358056 |}];
  std t ~keep_dims:true |> sexp_of_t |> print_s;
  [%expect {| ((1.2909944487358056)) |}]
;;

let%expect_test "broadcast" =
  let broadcast_and_print t ~dims:dims' =
    let t = broadcast t ~dims:dims' in
    print_s [%message "" (t : t) ~dims:(dims t : int array)]
  in
  let t = of_list2_exn [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  broadcast_and_print t ~dims:[| 1; 2; 2 |];
  [%expect {| ((t (((1 2) (3 4)))) (dims (1 2 2))) |}];
  broadcast_and_print t ~dims:[| 2; 2; 2 |];
  [%expect {| ((t (((1 2) (3 4)) ((1 2) (3 4)))) (dims (2 2 2))) |}]
;;

(* Box-Muller transform *)
let normal ?(mean = 0.) ?(std = 1.) ~dims ~rng () =
  let rec rejection_sample_unit_normal rng =
    let x1 = Splittable_random.float rng ~lo:(-1.) ~hi:1. in
    let x2 = Splittable_random.float rng ~lo:(-1.) ~hi:1. in
    let r2 = (x1 *. x1) +. (x2 *. x2) in
    match Float.O.(r2 >= 1. || r2 = 0.) with
    | true -> rejection_sample_unit_normal rng
    | false ->
      let f = Float.sqrt (-2. *. log r2 /. r2) *. std in
      (f *. x1) +. mean, (f *. x2) +. mean
  in
  let next = ref None in
  init ~dims ~f:(fun _index ->
    match !next with
    | Some value ->
      next := None;
      value
    | None ->
      let value1, value2 = rejection_sample_unit_normal rng in
      next := Some value2;
      value1)
;;

let%expect_test "normal" =
  let rng = Splittable_random.of_int 0 in
  normal ~dims:[| 2; 2 |] ~rng () |> sexp_of_t |> print_s;
  [%expect
    {|
    ((0.39995642633665462 1.1602368073797789)
     (1.1461698444484156 -0.27508260258159217))
    |}];
  let t = normal ~dims:[| 10000 |] ~rng () in
  let mean = mean t |> item in
  let std = std t |> item in
  print_s [%message "" (mean : float) (std : float)];
  [%expect {| ((mean 0.0026463860836857677) (std 0.98795664491502377)) |}]
;;

module With_shape = struct
  type nonrec t = t

  let sexp_of_t t = [%sexp { dims : int array = dims t; tensor : t = t }]
end

module Private = struct
  let to_bigarray = Fn.id
  let of_bigarray = Fn.id
end

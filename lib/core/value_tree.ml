open! Core

module General = struct
  type 'value t =
    | Leaf of 'value
    | Node of 'value t Map.M(String).t
  [@@deriving sexp, compare, variants, quickcheck]

  let rec length : _ t -> int = function
    | Leaf _ -> 1
    | Node children -> Map.data children |> List.sum (module Int) ~f:length
  ;;

  let rec map t ~f =
    match t with
    | Leaf value -> Leaf (f value)
    | Node children -> Node (Map.map children ~f:(fun t -> map t ~f))
  ;;
end

type t = Value0.Packed.t General.t [@@deriving sexp_of]

let node = General.node

let get_exn (t : t) name =
  match t with
  | Leaf _ -> raise_s [%message "Unexpected leaf node" (t : t) (name : string)]
  | Node children -> Map.find_exn children name
;;

let of_value value = General.leaf value

let to_value_exn : t -> Value0.Packed.t = function
  | Leaf value -> value
  | Node _ as t -> raise_s [%message "Unexpected non-leaf node" (t : t)]
;;

let rec flatten : 'value General.t -> 'value list = function
  | Leaf value -> [ value ]
  | Node children ->
    Map.to_alist children ~key_order:`Increasing
    |> List.concat_map ~f:(fun (_name, t) -> flatten t)
;;

module Def = struct
  type t = int array General.t [@@deriving sexp, compare, quickcheck]

  let leaf ~dims = General.leaf dims
  let node = General.node
  let create tree = General.map tree ~f:(fun (Value0.Packed.T t) -> Value0.dims t)
  let length = General.length
  let flatten = flatten
end

let to_def = Def.create

let rec unflatten' values ~(def : Def.t) ~sexp_of_value : _ General.t =
  match def with
  | Leaf dims ->
    (match values with
     | [ value ] ->
       [%test_result: int array] (Value0.Packed.dims value) ~expect:dims;
       Leaf value
     | _ -> raise_s [%message "Expected singleton leaf value" (values : value list)])
  | Node children ->
    let remaining_values, children =
      Map.to_alist children ~key_order:`Increasing
      |> List.fold_map ~init:values ~f:(fun values (name, def) ->
        let child_values, remaining_values = List.split_n values (General.length def) in
        let child = unflatten' child_values ~def ~sexp_of_value in
        remaining_values, (name, child))
    in
    (match remaining_values with
     | [] -> ()
     | _ ->
       raise_s
         [%message "Values remaining after unflattening" (remaining_values : value list)]);
    String.Map.of_alist_exn children |> node
;;

(* TODO: test flatten and unflatten roundtrip *)
let unflatten values ~def =
  unflatten' values ~def ~sexp_of_value:[%sexp_of: Value0.Packed.t]
;;

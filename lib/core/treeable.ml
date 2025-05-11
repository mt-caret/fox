open! Core
include Treeable_intf

module Conv (Treeable : S) (Conv : Conv_arg with type treeable := Treeable.t) :
  S with type t := Conv.t = struct
  include Conv

  let tree_of_t t = to_treeable t |> Treeable.tree_of_t
  let t_of_tree tree = Treeable.t_of_tree tree |> of_treeable
end

module Of_typed_fields (T : Of_typed_fields_arg) : S with type t := T.t = struct
  include T

  let tree_of_t t =
    List.map Typed_fields.Packed.all ~f:(fun { f = T field } ->
      let name = Typed_fields.name field in
      let to_tree, _ = field_treeable field in
      name, to_tree t)
    |> String.Map.of_alist_exn
    |> Value_tree.node
  ;;

  let t_of_tree (tree : Value_tree.t) =
    Typed_fields.create
      { f =
          (fun (type a) (field : a Typed_fields.t) ->
            let name = Typed_fields.name field in
            let _, (module T) = field_treeable field in
            T.t_of_tree (Value_tree.get_exn tree name))
      }
  ;;
end

module Tuple2 (A : S) (B : S) : S with type t = A.t * B.t = struct
  type t = A.t * B.t

  let tree_of_t (a, b) =
    Value_tree.node
      (String.Map.of_alist_exn [ "fst", A.tree_of_t a; "snd", B.tree_of_t b ])
  ;;

  let t_of_tree tree =
    ( A.t_of_tree (Value_tree.get_exn tree "fst")
    , B.t_of_tree (Value_tree.get_exn tree "snd") )
  ;;
end

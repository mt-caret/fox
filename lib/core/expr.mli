open! Core

module Var : sig
  type t =
    { name : string
    ; shape : Shape.t
    }
  [@@deriving compare, sexp, fields ~getters]

  val type_id : t Type_equal.Id.t
  val dims : t -> int iarray

  include Comparable.S_plain with type t := t
end

module Atom : sig
  type t =
    | Var of Var.t
    | Value of Value.t
  [@@deriving sexp_of]

  val of_value : Value.t -> vars:Var.Set.t -> t
  val shape : t -> Shape.t
  val dims : t -> int iarray
end

module Eq : sig
  type t =
    { var : Var.t
    ; op : Atom.t Op.t
    }
  [@@deriving sexp_of, fields ~getters]
end

type 'a t = private
  { parameters : Var.t list
  ; consts : 'a Var.Map.t
  ; equations : Eq.t list
  ; return_vals : Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, fields ~getters]

val create
  :  parameters:Var.t list
  -> consts:Value.t Var.Map.t
  -> equations:Eq.t list
  -> return_vals:Atom.t Nonempty_list.t
  -> out_tree_def:Value_tree.Def.t
  -> Value.t t

val map_consts : 'a t -> f:('a -> 'b) -> 'b t
val to_string_hum : Value.t t -> string

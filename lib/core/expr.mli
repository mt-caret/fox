open! Core

module Var : sig
  type t =
    { name : string
    ; shape : Shape.t
    }
  [@@deriving compare, hash, sexp, fields ~getters]

  val type_id : t Type_equal.Id.t
  val dims : t -> int iarray

  include Comparable.S_plain with type t := t
end

module Atom : sig
  type 'v t =
    | Var of Var.t
    | Value of 'v
  [@@deriving sexp_of, compare, hash]

  val of_value : Value.t -> vars:Var.Set.t -> Value.t t
  val shape : Value.t t -> Shape.t
  val dims : Value.t t -> int iarray
end

module Eq : sig
  type 'v t =
    { var : Var.t
    ; op : 'v Atom.t Op.t
    }
  [@@deriving sexp_of, compare, hash, fields ~getters]
end

type 'a t = private
  { parameters : Var.t list
  ; consts : 'a Map.M(Var).t
  ; equations : 'a Eq.t list
  ; return_vals : 'a Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, compare, hash, fields ~getters]

val create
  :  parameters:Var.t list
  -> consts:Value.t Var.Map.t
  -> equations:Value.t Eq.t list
  -> return_vals:Value.t Atom.t Nonempty_list.t
  -> out_tree_def:Value_tree.Def.t
  -> Value.t t

val map : 'a t -> f:('a -> 'b) -> 'b t
val to_string_hum : 'a t -> value_to_string:('a -> string) -> string

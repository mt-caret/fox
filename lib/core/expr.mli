open! Core

module Var : sig
  type t =
    { name : string
    ; dims : int array
    }
  [@@deriving compare, sexp, fields ~getters]

  val type_id : t Type_equal.Id.t

  include Comparable.S_plain with type t := t
end

module Atom : sig
  type t =
    | Var of Var.t
    | Value of Value.t
  [@@deriving sexp_of]

  val of_value : Value.t -> vars:Var.Set.t -> t
  val dims : t -> int array
end

module Eq : sig
  type t =
    { var : Var.t
    ; op : Atom.t Op.t
    }
  [@@deriving sexp_of, fields ~getters]
end

type t = private
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, fields ~getters]

val create
  :  parameters:Var.t list
  -> equations:Eq.t list
  -> return_vals:Atom.t Nonempty_list.t
  -> out_tree_def:Value_tree.Def.t
  -> t

val to_string_hum : t -> string

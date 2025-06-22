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
  type 'a t =
    | Var of Var.t
    | Value of 'a Value.t
  [@@deriving sexp_of]

  include Higher_kinded.S with type 'a t := 'a t

  module Packed : sig
    type 'a typed := 'a t
    type t = T : 'a typed -> t
  end

  val of_value : 'a Value.t -> vars:Var.Set.t -> 'a t
  val dims : 'a t -> int array
end

module Eq : sig
  type t =
    { var : Var.t
    ; op : Atom.higher_kinded Op.Packed.t
    }
  [@@deriving sexp_of, fields ~getters]
end

type t = private
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.Packed.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of, fields ~getters]

val create
  :  parameters:Var.t list
  -> equations:Eq.t list
  -> return_vals:Atom.Packed.t Nonempty_list.t
  -> out_tree_def:Value_tree.Def.t
  -> t

val to_string_hum : t -> string

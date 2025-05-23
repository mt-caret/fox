open! Core

module Var : sig
  type t = Var of string [@@deriving compare, sexp]

  val type_id : t Type_equal.Id.t

  include Comparable.S_plain with type t := t
end

module Atom : sig
  type t =
    | Var of
        { var : Var.t
        ; dims : int array option
        }
    | Value of Value.t
  [@@deriving sexp_of]

  val of_value : Value.t -> t
  val dims : t -> int array option
end

module Eq : sig
  type t =
    { var : Var.t
    ; op : Atom.t Op.t
    }
  [@@deriving sexp_of]
end

type t =
  { parameters : Var.t list
  ; equations : Eq.t list
  ; return_vals : Atom.t Nonempty_list.t
  ; out_tree_def : Value_tree.Def.t
  }
[@@deriving sexp_of]

val to_string_hum : t -> string

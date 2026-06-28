open! Core

module Partial_value : sig
  (** A value during partial evaluation: either fully [Known], or an [Unknown] standing
      for a traced variable. *)
  type t =
    | Known of Value.t
    | Unknown of Expr.Var.t
  [@@deriving sexp_of]

  val shape : t -> Shape.t
end

(** Partially evaluates [f] over [inputs]: [Known] inputs are computed through while
    [Unknown] inputs are traced, yielding the partial outputs and an [Expr.t] capturing
    the traced (unknown-dependent) part of the computation. *)
val partially_apply_expr_flat
  :  Partial_value.t list
  -> f:(Value.t list -> Value.t list * Value_tree.Def.t)
  -> Partial_value.t list * Value.t Expr.t

open! Core

(** The tracer behind [build_expr]. Under [handle], each [Op] effect is recorded as an
    equation over fresh variables, and the constants it touches are hoisted (deduplicated
    by identity) into named const-vars. *)
type t

val create : unit -> t
val fresh_var : t -> shape:Shape.t -> Expr.Var.t

(** Resolves a value to an atom, hoisting it into a fresh (or shared) const-var if it is a
    constant rather than one of this tracer's variables. *)
val intern_value : t -> Value.t -> Value.t Expr.Atom.t

val handle : t -> f:(unit -> 'a) -> 'a

(** The recorded equations, in evaluation order. *)
val equations : t -> Value.t Expr.Eq.t list

(** The hoisted constants, as a map from each const-var to its value. *)
val consts_map : t -> Value.t Expr.Var.Map.t

open! Core

module Dual_number : sig
  (* XCR mtakeda: Does this type need to be exposed? Could we e.g. make it abstract with
     creation functions / private? user: good call - made it abstract. The only consumer
     (the [jvp] transform) reads [primal]/[tangent] and calls [to_value], so those
     accessors are all we expose; the internal [id] tag (and the representation) are now
     hidden. No external code constructs a [Dual_number] directly - they come from
     [dual_number] / [lift] - so no creation function is needed here either. *)

  (** A primal value paired with its tangent (absent for non-differentiable values).
      Carried through a computation as a [Value.t] - see [to_value]. *)
  type t

  val primal : t -> Value.t
  val tangent : t -> Value.t option
  val to_value : t -> Value.t
end

(** The forward-mode (JVP) tracer. Under [handle], each [Op] effect is interpreted on
    [Dual_number]s, propagating tangents alongside primals. *)
type t

val create : unit -> t
val dual_number : t -> primal:Value.t -> tangent:Value.t option -> Dual_number.t

(** Views an arbitrary [Value.t] as a [Dual_number]: a value already tagged by this tracer
    keeps its tangent, anything else gets a zero (or absent, for non-float) tangent. *)
val lift : t -> Value.t -> Dual_number.t

val handle : t -> f:(unit -> 'a) -> 'a

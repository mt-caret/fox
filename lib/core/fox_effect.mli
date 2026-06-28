open! Core
open! Effect
open! Effect.Deep

type _ Effect.t += Op : Value0.t Op.t -> Value0.t t

(** Runs [f], interpreting each [Op] effect it performs with [handle] to produce the value
    the computation resumes with. *)
val handle : f:(unit -> 'a) -> handle:(Value0.t Op.t -> Value0.t) -> 'a

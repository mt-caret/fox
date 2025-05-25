open! Core

type t [@@deriving sexp_of]

val node : t String.Map.t -> t
val get_exn : t -> string -> t
val of_value : Value0.t -> t
val to_value_exn : t -> Value0.t

module Def : sig
  type t [@@deriving sexp_of, compare]

  val leaf : dims:int array -> t
  val node : t String.Map.t -> t
  val length : t -> int
  val flatten : t -> int array list
end

val to_def : t -> Def.t
val flatten : t -> Value0.t list
val unflatten : Value0.t list -> def:Def.t -> t

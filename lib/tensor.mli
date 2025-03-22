(** A multi-dimensional array of floats.*)

open! Core

type t [@@deriving sexp_of, compare]

val dims : t -> int array
val length : t -> int
val item : t -> float
val get : t -> int array -> float
val set : t -> int array -> float -> unit
val fill : t -> float -> unit
val reshape : t -> dims:int array -> t
val of_float : float -> t
val create : dims:int array -> float -> t
val zeros : dims:int array -> t
val ones : dims:int array -> t
val arange : int -> t
val map : t -> f:(float -> float) -> t
val map2 : t -> t -> f:(float -> float -> float) -> t
val ( + ) : t -> t -> t
val ( - ) : t -> t -> t
val ( * ) : t -> t -> t
val ( / ) : t -> t -> t
val ( ~- ) : t -> t
val sin : t -> t
val cos : t -> t

(** A multi-dimensional array of floats.*)

open! Core

type t [@@deriving sexp_of, compare]

val type_id : t Type_equal.Id.t
val dims : t -> int array
val num_dims : t -> int
val length : t -> int
val item : t -> float
val get : t -> int array -> float
val set : t -> int array -> float -> unit
val fill : t -> float -> unit
val left_slice : t -> indices:int array -> t
val sub_left : t -> pos:int -> len:int -> t
val reshape : t -> dims:int array -> t
val of_float : float -> t
val of_list : float list -> t

(** [of_list2_exn l] creates a tensor from a list of rows. Raises if a
    non-rectangular list of lists are provided. *)
val of_list2_exn : float list list -> t

val create : dims:int array -> float -> t
val init : dims:int array -> f:(int array -> float) -> t
val zeros : dims:int array -> t
val ones : dims:int array -> t
val arange : int -> t
val map : t -> f:(float -> float) -> t
val mapi : t -> f:(int array -> float -> float) -> t
val map2 : t -> t -> f:(float -> float -> float) -> t
val iter : t -> f:(float -> unit) -> unit
val iteri : t -> f:(int array -> float -> unit) -> unit

include Operators_intf.S with type t := t

val normal
  :  ?mean:float
  -> ?std:float
  -> dims:int array
  -> rng:Splittable_random.t
  -> unit
  -> t

module With_shape : sig
  type nonrec t = t [@@deriving sexp_of]
end

module Just_shape : sig
  type nonrec t = t [@@deriving sexp_of]
end

module Private : sig
  val to_bigarray
    :  t
    -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

  val of_bigarray
    :  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
    -> t
end

(** A multi-dimensional array of values *)

open! Core

module Typed : sig
  type 'a t [@@deriving sexp_of]

  val type_ : 'a t -> 'a Type.t
  val type_equal_id : 'a Type_equal.Id.t -> 'a t Type_equal.Id.t
  val dims : 'a t -> int array
  val shape : 'a t -> Shape.t
  val num_dims : 'a t -> int
  val length : 'a t -> int
  val item : 'a t -> 'a
  val get : 'a t -> int array -> 'a
  val set : 'a t -> int array -> 'a -> unit
  val fill : 'a t -> 'a -> unit
  val left_slice : 'a t -> indices:int array -> 'a t
  val sub_left : 'a t -> pos:int -> len:int -> 'a t
  val of_lit : 'a Type.t -> 'a -> 'a t
  val of_list : 'a Type.t -> 'a list -> 'a t

  (** [of_list2_exn l] creates a tensor from a list of rows. Raises if a
    non-rectangular list of lists are provided. *)
  val of_list2_exn : 'a Type.t -> 'a list list -> 'a t

  val create : 'a Type.t -> dims:int array -> 'a -> 'a t
  val init : 'a Type.t -> dims:int array -> f:(int array -> 'a) -> 'a t
  val zeros : dims:int array -> float t
  val ones : dims:int array -> float t
  val arange : int -> float t
  val map : 'b Type.t -> 'a t -> f:('a -> 'b) -> 'b t
  val mapi : 'b Type.t -> 'a t -> f:(int array -> 'a -> 'b) -> 'b t
  val map2 : 'c Type.t -> 'a t -> 'b t -> f:('a -> 'b -> 'c) -> 'c t
  val iter : 'a t -> f:('a -> unit) -> unit
  val iteri : 'a t -> f:(int array -> 'a -> unit) -> unit
  val allclose : ?equal_nan:bool -> 'a t -> 'a t -> bool
end

type t = T : 'a Typed.t -> t [@@deriving sexp_of]

val of_typed : 'a Typed.t -> t
val to_typed_exn : 'a Type.t -> t -> 'a Typed.t
val dims : t -> int array
val type_ : t -> Type.Packed.t
val shape : t -> Shape.t
val num_dims : t -> int
val length : t -> int
val item_exn : 'a Type.t -> t -> 'a
val get_exn : 'a Type.t -> t -> int array -> 'a
val left_slice : t -> indices:int array -> t
val sub_left : t -> pos:int -> len:int -> t
val init : 'a Type.t -> dims:int array -> f:(int array -> 'a) -> t
val of_list : 'a Type.t -> 'a list -> t
val of_list2_exn : 'a Type.t -> 'a list list -> t
val of_lit : 'a Type.t -> 'a -> t
val zeros : dims:int array -> t
val allclose : ?equal_nan:bool -> t -> t -> bool

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
    :  float Typed.t
    -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

  val of_float_bigarray
    :  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
    -> float Typed.t

  val of_char_bigarray
    :  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
    -> bool Typed.t
end

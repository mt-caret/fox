open! Core

module type S = sig
  type t

  val neg : t -> t
  val sin : t -> t
  val cos : t -> t
  val sqrt : t -> t
  val exp : t -> t
  val log : t -> t
  val sigmoid : t -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val eq : t -> t -> t
  val gt : t -> t -> t
  val lt : t -> t -> t
  val matmul : t -> t -> t
  val transpose : t -> t
  val reshape : t -> dims:int array -> t
  val sum : ?dims:[ `Just of int Nonempty_list.t | `All ] -> ?keep_dims:bool -> t -> t
  val mean : ?dims:[ `Just of int Nonempty_list.t | `All ] -> ?keep_dims:bool -> t -> t

  val var
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> ?correction:bool
    -> t
    -> t

  val std
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> ?correction:bool
    -> t
    -> t

  val softmax : dim:int -> t -> t
  val broadcast : t -> dims:int array -> t
  val scale : t -> float -> t

  module O : sig
    val ( ~- ) : t -> t
    val ( + ) : t -> t -> t
    val ( - ) : t -> t -> t
    val ( * ) : t -> t -> t
    val ( / ) : t -> t -> t
    val ( = ) : t -> t -> t
    val ( > ) : t -> t -> t
    val ( < ) : t -> t -> t
  end
end

open! Core

module type S = sig
  type 'a t

  val neg : float t -> float t
  val sin : float t -> float t
  val cos : float t -> float t
  val sqrt : float t -> float t
  val exp : float t -> float t
  val log : float t -> float t
  val sigmoid : float t -> float t
  val add : float t -> float t -> float t
  val sub : float t -> float t -> float t
  val mul : float t -> float t -> float t
  val div : float t -> float t -> float t
  val matmul : float t -> float t -> float t
  val transpose : 'a t -> 'a t
  val reshape : 'a t -> dims:int array -> 'a t

  val sum
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> float t
    -> float t

  val mean
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> float t
    -> float t

  val var
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> ?correction:bool
    -> float t
    -> float t

  val std
    :  ?dims:[ `Just of int Nonempty_list.t | `All ]
    -> ?keep_dims:bool
    -> ?correction:bool
    -> float t
    -> float t

  val softmax : dim:int -> float t -> float t
  val broadcast : 'a t -> dims:int array -> 'a t
  val scale : float t -> float -> float t

  module O : sig
    val ( ~- ) : float t -> float t
    val ( + ) : float t -> float t -> float t
    val ( - ) : float t -> float t -> float t
    val ( * ) : float t -> float t -> float t
    val ( / ) : float t -> float t -> float t
  end
end

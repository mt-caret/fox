open! Core

module type S = sig
  type t

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val neg : t -> t
  val sin : t -> t
  val cos : t -> t
  val matmul : t -> t -> t
  val transpose : t -> t
  val sum : t -> dims:int array -> keep_dims:bool -> t
  val broadcast : t -> dims:int array -> t

  module O : sig
    val ( + ) : t -> t -> t
    val ( - ) : t -> t -> t
    val ( * ) : t -> t -> t
    val ( ~- ) : t -> t
  end
end

open! Core

module type S = sig
  type t

  val tree_of_t : t -> Value_tree.t
  val t_of_tree : Value_tree.t -> t
end

module type Conv_arg = sig
  type t
  type treeable

  val to_treeable : t -> treeable
  val of_treeable : treeable -> t
end

module type Of_typed_fields_arg = sig
  type t

  module Typed_fields : Typed_fields_lib.S with type derived_on = t

  val field_treeable
    :  'a Typed_fields.t
    -> (t -> Value_tree.t) * (module S with type t = 'a)
end

module type Treeable = sig
  module type S = S

  module Conv (Treeable : S) (Conv : Conv_arg with type treeable := Treeable.t) :
    S with type t := Conv.t

  module Of_typed_fields (T : Of_typed_fields_arg) : S with type t := T.t
end

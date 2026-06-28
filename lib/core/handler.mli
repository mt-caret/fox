open! Core

(** Runs [f], evaluating each tensor op eagerly on the [Tensor] backend. *)
val eval : f:(unit -> 'a) -> 'a

(** Forward-mode AD: evaluates [f] at [primals] and its directional derivative along
    [tangents], returning both the primal and tangent outputs. *)
val jvp
  :  (module Treeable_intf.S with type t = 'in_)
  -> (module Treeable_intf.S with type t = 'out)
  -> f:('in_ -> 'out)
  -> primals:'in_
  -> tangents:'in_
  -> 'out * 'out

val jvp'
  :  f:(Value.t -> Value.t)
  -> primal:Value.t
  -> tangent:Value.t
  -> Value.t * Value.t

(** The derivative of a scalar [f] at [x] (its tangent for a unit input tangent). *)
val derivative : f:(Value.t -> Value.t) -> x:Value.t -> Value.t

val nth_order_derivative : n:int -> f:(Value.t -> Value.t) -> x:Value.t -> Value.t

(** Traces [f] into an [Expr.t]; constants it closes over are hoisted into [consts]. *)
val build_expr
  :  (module Treeable_intf.S with type t = 'in_)
  -> (module Treeable_intf.S with type t = 'out)
  -> f:('in_ -> 'out)
  -> in_tree_def:Value_tree.Def.t
  -> Value.t Expr.t

val build_expr' : f:(Value.t -> Value.t) -> in_dims:int iarray -> Value.t Expr.t

(** Evaluates a traced [Expr.t] on the surrounding effect handler. *)
val eval_expr
  :  (module Treeable_intf.S with type t = 'in_)
  -> (module Treeable_intf.S with type t = 'out)
  -> Value.t Expr.t
  -> 'in_
  -> 'out

val eval_expr' : Value.t Expr.t -> Value.t -> Value.t

(** Linearizes [f] at [primals]: returns the primal output and the linear tangent map. *)
val linearize
  :  (module Treeable_intf.S with type t = 'in_)
  -> (module Treeable_intf.S with type t = 'out)
  -> f:('in_ -> 'out)
  -> primals:'in_
  -> 'out * ('in_ -> 'out)

val linearize'
  :  f:(Value.t -> Value.t)
  -> primals:Value.t
  -> Value.t * (Value.t -> Value.t)

(** Reverse-mode AD: returns the primal output and a function mapping an output cotangent
    to the corresponding input cotangent. *)
val vjp
  :  (module Treeable_intf.S with type t = 'in_)
  -> (module Treeable_intf.S with type t = 'out)
  -> f:('in_ -> 'out)
  -> primals:'in_
  -> 'out * ('out -> 'in_)

val vjp' : f:(Value.t -> Value.t) -> primal:Value.t -> Value.t * (Value.t -> Value.t)

(** [grad]ient of a scalar-valued [f], paired with its value. *)
val grad_and_value
  :  (module Treeable_intf.S with type t = 'in_)
  -> f:('in_ -> Value.t)
  -> x:'in_
  -> Value.t * 'in_

val grad_and_value' : f:(Value.t -> Value.t) -> x:Value.t -> Value.t * Value.t

val grad
  :  (module Treeable_intf.S with type t = 'in_)
  -> f:('in_ -> Value.t)
  -> x:'in_
  -> 'in_

val grad' : f:(Value.t -> Value.t) -> x:Value.t -> Value.t

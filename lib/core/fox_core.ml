open! Core
module Expr = Expr
module Fox_effect = Fox_effect
module Op = Op
module Operators_intf = Operators_intf
module Shape = Shape
module Tensor = Tensor
module Treeable = Treeable
module Treeable_intf = Treeable_intf
module Value = Value
module Value_tree = Value_tree
include Handler

(* Internals exposed only for the test suite. *)
module For_testing = struct
  module Partial_value = Partial_eval.Partial_value

  let partially_apply_expr_flat = Partial_eval.partially_apply_expr_flat
end

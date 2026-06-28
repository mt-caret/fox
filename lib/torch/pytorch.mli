(** A PyTorch (libtorch) backend for fox: an interpreter that evaluates fox tensor
    programs on [Torch.Tensor.t]s, primarily so results can be differential-tested against
    the eager [Tensor] backend.

    fox is float64 and its element types are [Float] and [Bool]. Floats map to torch
    [Double] tensors; bools (stored as 0/1 bytes) map to torch [Uint8] tensors, and torch
    comparisons (which yield [Bool] tensors) are normalised back to fox [Bool]. *)

open! Core
open! Fox_core

(** The tensor primitives implemented over [Torch.Tensor.t]. Mirrors the eager [Tensor]
    backend op-for-op via [Op.Make_operators]. *)
module Backend : Operators_intf.S with type t = Torch.Tensor.t

(** Converts a fox eager tensor to a torch tensor via bigarrays. *)
val of_tensor : Tensor.t -> Torch.Tensor.t

(** Converts a torch tensor back to a fox eager tensor via bigarrays. *)
val to_tensor : Torch.Tensor.t -> Tensor.t

(** [handle ~f] runs [f], interpreting each fox [Op] it performs through libtorch.
    Analogous to [Handler.eval], but evaluating on torch tensors. *)
val handle : f:(unit -> 'a) -> 'a

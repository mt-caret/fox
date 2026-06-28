open! Core
open! Effect
open! Effect.Deep

type _ Effect.t += Op : Value0.t Op.t -> Value0.t t

let handle ~f ~(handle : Value0.t Op.t -> Value0.t) =
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option =
    fun eff ->
    match eff with
    | Op op -> Some (fun k -> continue k (handle op))
    | _ -> None
  in
  Effect.Deep.try_with f () { effc }
;;

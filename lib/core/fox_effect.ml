open! Core
open! Effect
open! Effect.Deep

type _ Effect.t += Op : ('a, Value0.higher_kinded) Op.t -> 'a Value0.t t

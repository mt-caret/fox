open! Core
open! Effect
open! Effect.Deep

type _ Effect.t += Op : Value0.t Op.t -> Value0.t t

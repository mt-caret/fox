(** An opaque unique identifier.*)

open! Core

type t = private int [@@deriving equal, sexp_of]

val create : unit -> t

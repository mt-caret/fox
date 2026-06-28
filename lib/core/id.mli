(** An opaque unique identifier. *)

open! Core

type t = private int [@@deriving compare, equal, sexp_of]

val create : unit -> t

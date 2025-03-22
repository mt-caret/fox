open! Core

type t = int [@@deriving equal, sexp_of]

let create =
  let counter = ref 0 in
  fun () ->
    let id = !counter in
    counter := id + 1;
    id
;;

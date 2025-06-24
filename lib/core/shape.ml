open! Core

module T = struct
  type t =
    { dims : int array
    ; type_ : Type.Packed.t
    }
  [@@deriving equal, compare, sexp, fields ~getters]
end

include T
include Comparable.Make_plain (T)

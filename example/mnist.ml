open! Core
open! Fox_core

module Dataset = struct
  type t =
    { x : Tensor.Just_shape.t
    ; y : Tensor.Just_shape.t
    }
  [@@deriving sexp_of]

  let length t = Tensor.length t.y

  let load ~header_length path =
    let contents =
      In_channel.with_file path ~f:(fun in_channel ->
        let size = Int64.to_int_exn (In_channel.length in_channel) - header_length in
        let bigstring = Bigstring.create size in
        Bigstring_unix.really_pread
          ~offset:header_length
          (Core_unix.descr_of_in_channel in_channel)
          bigstring;
        bigstring)
    in
    Tensor.init
      ~dims:[| Bigstring.length contents |]
      ~f:(fun index -> Bigstring.get contents index.(0) |> Char.to_int |> Int.to_float)
  ;;

  let load ~x ~y =
    { x = load ~header_length:16 x |> Tensor.reshape ~dims:[| -1; 28; 28 |]
    ; y = load ~header_length:8 y
    }
  ;;

  let print { x; y } ~i =
    let n = Tensor.get y [| i |] |> Int.of_float in
    print_endline [%string "Label: %{n#Int}"];
    let x = Tensor.left_slice x ~indices:[| i |] in
    let image =
      List.range 0 28
      |> List.map ~f:(fun row ->
        List.range 0 28
        |> List.map ~f:(fun col ->
          let pixel = Tensor.get x [| row; col |] in
          let pixel = 255 - Int.of_float (pixel /. 255. *. 23.) in
          [%string "\027[48;5;%{pixel#Int}m "])
        |> String.concat)
      |> String.concat ~sep:"\027[49m\n"
    in
    print_endline [%string "%{image}\027[49m"]
  ;;
end

module Model = struct
  module T = struct
    type t =
      { h1 : Value.t
      ; b1 : Value.t
      ; h2 : Value.t
      ; b2 : Value.t
      ; h3 : Value.t
      ; b3 : Value.t
      }
    [@@deriving typed_fields, sexp_of]

    let field_treeable (type a) (field : a Typed_field.t)
      : (a -> Value_tree.t) * (module Treeable.S with type t = a)
      =
      match field with
      | H1 -> Value.tree_of_t, (module Value)
      | B1 -> Value.tree_of_t, (module Value)
      | H2 -> Value.tree_of_t, (module Value)
      | B2 -> Value.tree_of_t, (module Value)
      | H3 -> Value.tree_of_t, (module Value)
      | B3 -> Value.tree_of_t, (module Value)
    ;;
  end

  include T
  include Treeable.Of_typed_fields (T)

  let create ~rng =
    { h1 = Tensor.normal ~dims:[| 784; 128 |] ~rng () |> Value.of_tensor
    ; b1 = Tensor.zeros ~dims:[| 128 |] |> Value.of_tensor
    ; h2 = Tensor.normal ~dims:[| 128; 64 |] ~rng () |> Value.of_tensor
    ; b2 = Tensor.zeros ~dims:[| 64 |] |> Value.of_tensor
    ; h3 = Tensor.normal ~dims:[| 64; 10 |] ~rng () |> Value.of_tensor
    ; b3 = Tensor.zeros ~dims:[| 10 |] |> Value.of_tensor
    }
  ;;

  let to_list { h1; b1; h2; b2; h3; b3 } = [ h1; b1; h2; b2; h3; b3 ]

  let map t ~f =
    { h1 = f t.h1; b1 = f t.b1; h2 = f t.h2; b2 = f t.b2; h3 = f t.h3; b3 = f t.b3 }
  ;;

  let map2 t1 t2 ~f =
    { h1 = f t1.h1 t2.h1
    ; b1 = f t1.b1 t2.b1
    ; h2 = f t1.h2 t2.h2
    ; b2 = f t1.b2 t2.b2
    ; h3 = f t1.h3 t2.h3
    ; b3 = f t1.b3 t2.b3
    }
  ;;

  let linear ~h ~b ~bs x =
    let open Value.O in
    let b = Value.broadcast b ~dims:(Array.append [| bs |] (Value.dims b)) in
    Value.matmul x h + b
  ;;

  let run { h1; b1; h2; b2; h3; b3 } x =
    match Value.dims x with
    | [| bs; 784 |] ->
      let x (* bs x 128 *) = linear ~h:h1 ~b:b1 ~bs x |> Value.sigmoid in
      let x (* bs x 64 *) = linear ~h:h2 ~b:b2 ~bs x |> Value.sigmoid in
      let x (* bs x 10 *) = linear ~h:h3 ~b:b3 ~bs x |> Value.softmax ~dim:1 in
      x
    | _ -> raise_s [%message "Invalid input dimensions" ~dims:(Value.dims x : int array)]
  ;;

  let cross_entropy_loss t ~x ~y =
    let open Value.O in
    let y_hat = run t x in
    let eps = 1e-9 |> Value.of_float |> Value.broadcast ~dims:(Value.dims y) in
    Value.sum ~dims:(`Just [ 1 ]) (-y * Value.log (y_hat + eps)) |> Value.mean
  ;;
end

let seed = 42
let batch_size = 128
let learning_rate = 0.01

let command =
  Command.basic ~summary:"MNIST example"
  @@
  let%map_open.Command () = Command.Param.return () in
  fun () ->
    (* Suppresses noisy XLA log message *)
    Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
    let train =
      Dataset.load ~x:"mnist/train-images-idx3-ubyte" ~y:"mnist/train-labels-idx1-ubyte"
    in
    print_s [%sexp (train : Dataset.t)];
    let test =
      Dataset.load ~x:"mnist/t10k-images-idx3-ubyte" ~y:"mnist/t10k-labels-idx1-ubyte"
    in
    print_s [%sexp (test : Dataset.t)];
    Dataset.print train ~i:0;
    let rng = Splittable_random.of_int seed in
    let model = ref (Model.create ~rng) in
    let print_dataset_loss () =
      let x =
        Tensor.reshape train.x ~dims:[| -1; 28 * 28 |]
        |> Tensor.map ~f:(fun x -> x /. 255.)
        |> Value.of_tensor
      in
      let y =
        let labels = train.y in
        Tensor.init
          ~dims:[| Tensor.length labels; 10 |]
          ~f:(fun index ->
            let label = Tensor.get labels [| index.(0) |] |> Float.to_int in
            if label = index.(1) then 1. else 0.)
        |> Value.of_tensor
      in
      let loss =
        Fox_jit.jit
          (module Model)
          (module Value)
          ~f:(fun model -> Model.cross_entropy_loss model ~x ~y)
          !model
      in
      print_s [%message "test dataset loss" (loss : Value.t)]
    in
    for i = 0 to (Dataset.length train / batch_size) - 1 do
      let x =
        Tensor.sub_left train.x ~pos:(i * batch_size) ~len:batch_size
        |> Tensor.reshape ~dims:[| -1; 28 * 28 |]
        |> Tensor.map ~f:(fun x -> x /. 255.)
        |> Value.of_tensor
      in
      let y =
        let labels = Tensor.sub_left train.y ~pos:(i * batch_size) ~len:batch_size in
        Tensor.init ~dims:[| batch_size; 10 |] ~f:(fun index ->
          let label = Tensor.get labels [| index.(0) |] |> Float.to_int in
          if label = index.(1) then 1. else 0.)
        |> Value.of_tensor
      in
      let loss, grad =
        Fox_jit.jit
          (module Model)
          (module Treeable.Tuple2 (Value) (Model))
          ~f:(fun model ->
            grad_and_value
              (module Model)
              ~f:(fun model -> Model.cross_entropy_loss model ~x ~y)
              ~x:model)
          !model
      in
      let average_grad_l2_norm =
        Eval.handle ~f:(fun () ->
          let grad_norms =
            Model.map grad ~f:(fun x -> Value.mean Value.O.(x * x)) |> Model.to_list
          in
          let sum = List.reduce_exn grad_norms ~f:Value.add in
          Value.scale sum (1. /. Float.of_int (List.length grad_norms)))
      in
      if i mod 100 = 0 then print_dataset_loss ();
      print_s [%message (loss : Value.t) (average_grad_l2_norm : Value.t)];
      model
      := Eval.handle ~f:(fun () ->
           Model.map2 !model grad ~f:(fun a b ->
             Value.O.(a - Value.scale b learning_rate)))
    done;
    print_dataset_loss ()
;;

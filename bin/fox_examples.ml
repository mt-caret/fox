open! Core

let () =
  Command_unix.run @@ Command.group ~summary:"Fox examples" [ "mnist", Mnist.command ]
;;

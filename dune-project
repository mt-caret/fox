(lang dune 3.20)

(name fox)

(generate_opam_files true)

(source
 (github mt-caret/fox))

(authors "mt-caret")

(maintainers "mt-caret")

(package
 (name fox)
 (synopsis "Autodiff and XLA JIT compilation for OCaml, inspired by JAX")
 (description
  "Fox is an OCaml library for automatic differentiation and just-in-time XLA compilation of tensor computations, inspired by JAX.")
 (allow_empty)
 (depends
  core
  core_kernel
  core_unix
  base_quickcheck
  expect_test_helpers_core
  ppx_jane
  ppx_typed_fields
  splittable_random
  xla
  (ocamlformat :with-dev-setup)
  (odoc
   (or :with-doc :with-dev-setup))
  (utop :with-dev-setup)))

(pin
 (package
  (name xla))
 (url
  "git+https://github.com/mt-caret/ocaml-xla.git#6dffd132a9e1b13b3b8c17f1abae110c40ea6f3c"))

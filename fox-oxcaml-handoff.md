# Handoff: building `fox` with OxCaml + a reproducible opam lockfile

## The goal

Get **`fox`** (https://github.com/mt-caret/fox — a JAX-like OCaml library for autodiff +
XLA JIT) building with **OxCaml** and a **reproducible opam lock file**, mirroring the
setup of **`mt-caret/orc`** (the reference to copy).

References:
- OxCaml: https://oxcaml.org/
- opam lockfiles: https://ocamlpro.com/blog/2026_01_08_opam_104_sharing_your_code/#lockdependencies
- Pattern to mirror: https://github.com/mt-caret/orc

The orc pattern = a **local `_opam` switch** on `ocaml-variants.5.2.0+ox`, `dune-project`
with `(generate_opam_files true)` + a `(package …)` stanza + `(pin …)` stanzas, a
`<pkg>.opam.template` (for `pin-depends` + `x-maintenance-intent`), a committed
`<pkg>.opam.locked`, and a GitHub Actions CI workflow. Create switches with
`opam switch create . 5.2.0+ox --repos ox=git+https://github.com/oxcaml/opam-repository.git,default --locked`.

## My (the user's) preferences

- **Jane Street / OxCaml OCaml style** per the global `~/.claude/CLAUDE.md`. Most relevant:
  - Always `dune build` **and** `dune runtest` before claiming done; `dune fmt` for
    formatting but **inspect the diff** before promoting.
  - `open Core`; prefer Core's `List`/`Option`/`String`/etc.
  - `.mli` for every non-test `.ml` except the library-name wrapper module.
  - Don't introduce single-use abstractions (2+ uses is fine — e.g. the shared effect
    handler helper below).
  - Comments must state non-obvious things and stand alone for a fresh reader — **no**
    "ported from X" / "now uses Y" history comments.
  - `[%sexp_of: T]`/`[%of_sexp: T]` extension forms, ppx_let prefix forms, etc.
- **Git: leave changes uncommitted on `main`.** Don't commit or push unless I ask.
- I already approved doing the **full** job end-to-end (scaffold → switch → deps → build/
  test → lockfile → CI/README). The blocker below (modes porting) was hit mid-way.

## What this task actually is (read this first)

It is **not** just build config. `fox`'s source was written for **vanilla OCaml 5.3 +
non-modal Jane Street libs**, but **OxCaml is 5.2-based and the JS `v0.18~preview` libs are
modal** (local/global modes). The `ox` opam repo only ships `ocaml-variants.5.2.0+ox`
(there is **no 5.3+ox**). So you must **port `fox`'s source** to compile on 5.2.0+ox + the
modal libs. Expect to touch the library's core. Raise this scope with me if it balloons.

## Config scaffolding (these all worked — recreate them)

**`dune-project`:**
```scheme
(lang dune 3.20)
(name fox)
(generate_opam_files true)
(source (github mt-caret/fox))
(authors "mt-caret")
(maintainers "mt-caret")
(package
 (name fox)
 (synopsis "Autodiff and XLA JIT compilation for OCaml, inspired by JAX")
 (description "…")
 (allow_empty)
 (depends
  core core_kernel core_unix base_quickcheck expect_test_helpers_core
  ppx_jane ppx_typed_fields splittable_random xla
  (ocamlformat :with-dev-setup)
  (odoc (or :with-doc :with-dev-setup))
  (utop :with-dev-setup)))
(pin
 (package (name xla))
 (url "git+https://github.com/mt-caret/ocaml-xla.git#main"))
```
Dep mapping from the `dune` files: `core_kernel.nonempty_list`→`core_kernel`;
`core_unix.{bigstring_unix,command_unix}`→`core_unix`; `ppx_typed_fields.typed_fields_lib`+
`base_quickcheck.ppx_quickcheck` are covered by `ppx_typed_fields`/`base_quickcheck`;
`ppx_jane` covers ppx_expect/inline_test. `xla` is the only non-JS dep.

**`fox.opam.template`:**
```
pin-depends: [
  ["xla.dev" "git+https://github.com/mt-caret/ocaml-xla.git#main"]
]
x-maintenance-intent: ["(latest)"]
```
Then `dune build ./fox.opam` regenerates `fox.opam` (the `(pin …)` stanza is accepted by
dune 3.20; `pin-depends` comes through from the template).

**`.ocamlformat`:** reduce to just `profile = janestreet`. The old file pinned
`version = 0.27.0` / `ocaml-version = 5.3`, which **conflicts** with OxCaml's
`ocamlformat 0.26.2+ox1` and breaks `dune build @fmt`. (The +ox ocamlformat DOES parse
OxCaml syntax. After porting, run `dune fmt` and inspect/`dune promote` the reflow.)

**`.gitignore`:** add `_opam`, `*.install`, `xla_extension`.

## The `xla` dependency

- `xla` is **unpublished** (not in `default` or `ox` repos). Pin
  `git+https://github.com/mt-caret/ocaml-xla.git#main` (mt-caret's fork of
  LaurentMazare/ocaml-xla; default branch `main`).
- It binds a **prebuilt XLA extension blob**. Download elixir-nx/xla **v0.4.4**:
  ```bash
  wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cpu.tar.gz
  tar -xzf xla_extension-x86_64-linux-gnu-cpu.tar.gz   # -> ./xla_extension/{include,lib/libxla_extension.so (~305MB)}
  export XLA_EXTENSION_DIR=$PWD/xla_extension
  ```
  (Other platforms: pick the matching archive.) The configurator
  (`src/config/discover.ml`) resolves it via `XLA_EXTENSION_DIR` > `$DUNE_SOURCEROOT/
  xla_extension` > `$OPAM_SWITCH_PREFIX/lib/libxla`, and bakes `-Wl,-rpath,…/lib`, so **no
  `LD_LIBRARY_PATH` is needed at runtime once built**.
- **`XLA_EXTENSION_DIR` must be set during `opam install` too** (the `xla_stubs.cpp` C++
  stubs need the headers at build time), not only for `dune build`.

## ⚠️ Biggest pitfall: `mt-caret/ocaml-xla` has broken opam metadata

Its `xla.opam` / `dune-project (depends …)` only lists
`base stdio camlzip ctypes ctypes-foreign ocaml dune`, but its dune files actually use:
- `src/wrapper/dune` libraries: **`int_repr`, `yojson`** (and `bigarray`); preprocess pps:
  **`ppx_expect ppx_sexp_conv ppx_sexp_message ppx_compare`**.
- `src/config/dune`: `dune.configurator` (→ **`dune-configurator`**).

Observed failures on a fresh install: `Library "ppx_expect" not found` (opam builds `xla`
before the ppx tree — no dependency edge), then `Library "yojson" not found` /
`int_repr` missing (nothing pulls them in).

**Why it matters for the lockfile:** `opam lock` computes the closure from *declared*
metadata, so it will **omit `yojson`/`int_repr`** → the lockfile is incomplete → a fresh
`--locked`/CI install fails. Plus a build-ordering race (xla before its ppx).

**Correct fix** (this is exactly the orc pattern — orc forks+fixes its pinned deps, e.g.
its `jsonaf` "oxcaml-portable-fix" branch): add the missing deps to
`mt-caret/ocaml-xla`'s `dune-project (depends …)` —
`int_repr yojson dune-configurator ppx_expect ppx_sexp_conv ppx_sexp_message ppx_compare`
(or just `ppx_jane` for the ppx set) — regenerate `xla.opam`, push to a branch, and pin
`fox` to that commit. This fixes both presence (lockfile completeness) and ordering.
**This touches a SEPARATE repo (mt-caret/ocaml-xla) — get my OK before pushing.**

Local-only stopgap that unblocked the build (NOT enough for clean reproducibility):
`opam install yojson int_repr -y` into the switch first, then
`opam install . --deps-only --with-test` (xla then builds because the ppx tree is present).

## Source porting needed (in order)

### 1. Effect-handler syntax (5.3 → 5.2) — DONE on the old box, redo it
`lib/core/fox_core.ml` has 4 deep handlers (`Eval`, `Jvp`, `Staging`, `Partial`) using the
OCaml-5.3 syntax `try f () with | effect Fox_effect.Op op, k -> … continue k value`, which
5.2.0+ox can't parse ("Syntax error"). There is a single effect constructor
`Fox_effect.Op : Value0.t Op.t -> Value0.t Effect.t`, and all 4 produce a `Value.t`
(`Value.t = Value0.t`). Add one shared helper and rewrite each handler through it:
```ocaml
(* [Fox_effect.Op] is the only effect performed; one shared deep handler suffices.
   [handle] turns an op into the value to resume with. *)
let handle_op ~f ~handle =
  Effect.Deep.try_with f ()
    { effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Fox_effect.Op op -> Some (fun k -> continue k (handle op))
          | _ -> None)
    }
;;
```
Each `handle [t] ~f = try f () with | effect Fox_effect.Op op, k -> <body>; continue k X`
becomes `handle [t] ~f = handle_op ~f ~handle:(fun op -> <body>; X)` (drop `continue k`,
return the value, close the lambda). `open! Effect` and `open! Effect.Deep` are already at
the top. `fox_effect.ml/.mli` (the `type _ Effect.t += Op …` extension) and `value.ml`'s
`Effect.perform (Fox_effect.Op op)` are fine on 5.2 — no change.

### 2. `Set_once` API → `[%call_pos]` — DONE on the old box, redo it
Installed sigs (`_opam/lib/core/set_once.mli`):
`set_exn : 'a t -> here:[%call_pos] -> 'a -> unit`,
`get_exn : here:[%call_pos] -> 'a t -> 'a`. In `fox_core.ml`:
- `Set_once.set_exn out_tree_def here (Value_tree.to_def out_tree)` →
  `Set_once.set_exn out_tree_def ~here (Value_tree.to_def out_tree)`.
- `Set_once.get_exn out_tree_def [%here]` (2 sites) → `Set_once.get_exn out_tree_def`
  (drop `[%here]`; call_pos auto-fills).
(The whole `~here` threading through `flatten_function` could be dropped for `[%call_pos]`,
but the minimal change above works.)

### 3. OxCaml modes / locality — UNRESOLVED, the hard part
Build stopped here. `lib/core/treeable.ml` `Of_typed_fields.t_of_tree` calls
`Typed_field.create { f = (fun (type a) (field : a Typed_field.t) -> …) }`, but the modal
`Typed_fields_lib.S.create` callback now requires the field at **local** mode:
> Error: This expression has type `'a Typed_field.t -> 'a` but … expected
> `'b Typed_field.t @ local -> 'b`.

Making `field` local (`fun (type a) (local_ field : a Typed_field.t) -> …`) **cascades**:
the body calls `field_treeable field`, and `field_treeable` is declared in
`Of_typed_fields_arg` (`treeable_intf.ml`) as `'a Typed_field.t -> …` (global). So you'd
also change that module type's `field_treeable` to accept a local field
(`'a Typed_field.t @ local -> …`) and update **every implementation** of
`Of_typed_fields_arg`.

Next steps that were about to happen:
- Read the exact `create` signature in the installed `typed_fields_lib` `.mli` and check
  whether `Typed_field.name`/`get` accept local fields.
- `grep -rn 'typed_fields\|field_treeable\|Of_typed_fields\|@@deriving typed_fields'
  lib example` to size the blast radius (the `example/` mnist code likely uses it).
- Expect **more mode errors after this one** (the build stops at the first). Iterate
  `dune build` → add `local_` / `@ local` / `@ global` annotations. If it gets deep or
  needs real local/global *design* decisions in fox's own API, stop and check with me.

## Recommended order on the fresh box

1. Recreate the config scaffolding (above).
2. **Decide the `xla` metadata fix with me first** (push fix to mt-caret/ocaml-xla + pin
   the commit) — otherwise the lockfile will be incomplete.
3. Port source: effects (1), Set_once (2), then iterate modes (3) with `dune build`.
4. Download the XLA extension; `opam switch create . 5.2.0+ox --repos ox=…,default
   --no-install -y`; then `XLA_EXTENSION_DIR=$PWD/xla_extension opam install . --deps-only
   --with-test -y`.
5. `XLA_EXTENSION_DIR=$PWD/xla_extension dune build @default @runtest`; then `dune fmt`
   (inspect diff).
6. `opam lock .` → `fox.opam.locked`. **Verify** by doing a clean
   `opam switch create <tmpdir> 5.2.0+ox --repos ox=…,default --locked` from scratch — this
   is where the ocaml-xla metadata bug bites if not fixed.
7. CI (`.github/workflows/build.yml`) + README. Leave changes **uncommitted on main**.

## CI workflow (drafted; based on orc CI + the ocaml-xla CI)

Single `ubuntu-latest` job (the xla_extension archive is linux x86_64; macOS needs the
darwin archive). Steps: `jlumbroso/free-disk-space` (the +ox JS tree is large) →
`actions/checkout` → `ocaml/setup-ocaml@v3` with `ocaml-compiler: ocaml-variants.5.2.0+ox`
and `opam-repositories: ox + default` → download xla_extension and
`echo "XLA_EXTENSION_DIR=$PWD/xla_extension" >> "$GITHUB_ENV"` → `actions/cache` on
`~/.opam` + `_opam` keyed on `hashFiles('fox.opam.locked')` →
`opam install . --deps-only --with-test --locked` → `opam exec -- dune build @default` →
`opam exec -- dune runtest` → `opam exec -- dune build @fmt`.

## README

Replace the old `DYLD_LIBRARY_PATH=… dune build` line with: note that fox targets OxCaml
(effect syntax) + pins mt-caret/ocaml-xla; one-time setup = download xla_extension + export
`XLA_EXTENSION_DIR` + `opam switch create . 5.2.0+ox --repos ox=…,default --locked`; build/
test = `XLA_EXTENSION_DIR=$PWD/xla_extension dune build @default @runtest`; regen lockfile =
`opam install . --deps-only --with-test` then `opam lock .`.

## Open questions to confirm with me

1. OK to **port fox's source to OxCaml 5.2 + modal libs** (effects + Set_once + the
   typed_fields/local-mode changes that touch fox's own `Of_typed_fields_arg` API)? The
   alternatives — wait for a 5.3-based OxCaml, or build on vanilla 5.3 + non-modal JS
   (which would **not** be OxCaml) — don't meet the stated goal.
2. OK to **push the metadata fix to `mt-caret/ocaml-xla`** and pin fox to that commit?

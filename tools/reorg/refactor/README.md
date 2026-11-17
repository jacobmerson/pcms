# pcms-refactor: the reorganization's namespace moves

The reorganization moves internal code from `pcms::` into one namespace per
sub-library (`pcms::utility`, `pcms::field`, ...) and per backend
(`pcms::backend::omega_h`, ...). `namespaces.spec` lists which files move
where; `move_namespaces.sh` applies all of it in one pass with `pcms-refactor`,
a Clang LibTooling tool, so every reference is found and checked by the
compiler, not by pattern matching.

## What a move changes

1. The top-level `namespace pcms` blocks of the listed files become
   `namespace <target>` (closing comments included). Blocks that only hold
   using-declarations stay in `pcms`.
2. Every reference to a moved declaration, anywhere in the tree, is
   rewritten. The moved declarations are found from the AST: namespace-scope
   declarations in `pcms` whose first declaration is in a listed file.
3. References are spelled as briefly as lookup allows, checked with Clang's
   lookup (`clang::tooling::replaceNestedName`) plus a check that no class
   member, inherited member or parameter hides the name:
   - Inside the target namespace: the bare name.
   - In a header: relative to the enclosing namespace
     (`localization::GridPointSearch2D` inside `pcms::transfer`).
   - In a `.cpp` file, for an internal name (declared outside `include/`,
     not in a `detail` namespace): the form it had before the move (bare, or
     `pcms::X`), and the file gets a using-declaration for it in a
     `namespace pcms { using localization::X; }` block after its includes.
     Uses inside `#if` regions are spelled relatively instead, so no
     using-declaration depends on the configuration.
   - Public and `detail` names keep a qualifier.
   - Moved code that names something already in its new namespace with a
     qualifier (`pcms::backend::omega_h::X` in code moving there) is
     shortened the same way.
4. A global namespace that a new namespace would hide gets a leading `::`
   (`mfem::Vector` inside `pcms::backend::mfem` becomes `::mfem::Vector`).

Uses that cannot be edited safely (in a macro body) are reported, not
changed. An unqualified call that depends on a template parameter and found
a moved function through argument-dependent lookup is qualified, with a note.

## Limitations

These cases are not checked by the tool; compiling the result catches most
of them, so build and test after a run.

- Hiding is checked against class members, inherited members and the
  enclosing function's parameters, not against block-scope declarations,
  lambda parameters or template parameters.
- An unqualified reference in moved code to a name that stays in `pcms` is
  left as it is. It can be hidden after the move by a declaration with the
  same name in the target namespace (for example an overload in a backend
  header the file includes later).
- When only some overloads of a name move, an unqualified dependent call to
  that name is not rewritten and no note is printed, and a using-declaration
  of the name follows the namespace of its first declaration.
- A declaration counts as moved when its first declaration in a translation
  unit is in a listed file. A forward declaration or out-of-line definition
  in an unlisted file stays in `pcms`: replace such forward declarations with
  includes before running the tool.
- Qualifying a dependent unqualified call turns off argument-dependent
  lookup for it. The tool prints a note, except for calls in files that move
  to a different namespace than the function.

## Running it

```sh
tools/reorg/refactor/build.sh <bindir>        # needs the Clang/LLVM 19 development libraries
# a configured build of the tree with every feature on, in the build's environment:
cmake -S . -B <build> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON <every PCMS_ENABLE_* option ON> ...
PCMS_REFACTOR=<bindir>/pcms-refactor \
  tools/reorg/refactor/move_namespaces.sh . <build> tools/reorg/refactor/namespaces.spec
clang-format -i $(git diff --name-only -- '*.cpp' '*.hpp' '*.h')   # clang-format 18.1.3, as CI
```

Every translation unit of the build is parsed, plus one for each header that
nothing includes. The run aborts without changing anything if the tool fails,
if a translation unit does not parse, or if two translation units disagree on
an edit; it fails after applying the edits if a marked reference is left
unconverted. Applying the spec to a tree that has already
been converted changes nothing.

## Files

- `namespaces.spec`: the moves (format described in the file).
- `pcms_refactor.cpp`: the tool. `--mode=qualify --moves=<file>` renames the
  blocks and writes every affected reference fully qualified, marked with a
  leading `::pcms::` (which PCMS code never uses); `--mode=spell` shortens
  the marked qualifiers on the moved tree. Both print edits as JSON lines.
- `apply_edits.py`: merges the edits from all translation units and applies
  them.
- `build.sh`: builds `pcms-refactor`.
- `move_namespaces.sh`: expands the spec, builds the list of translation
  units, runs both passes in parallel and applies the edits.

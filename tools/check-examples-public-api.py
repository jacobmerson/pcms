#!/usr/bin/env python3
"""Reject uses of PCMS's internals in the examples.

Usage: tools/check-examples-public-api.py [--include-dir <dir>] [<path>...]

The examples teach the public API, so they may use nothing else. The script
reads the code under each <path> (default: the examples/ directory of the
repository this script is in) and reports, with file and line:

  * `detail::` and `impl::`, the namespaces of PCMS's internals (and
    `pcms.detail` in Python);
  * an include of a header in a `detail/` directory;
  * an include of a `pcms/...` header that is not an installed public header:
    one that is not below <include-dir>/pcms/ (default: include/ of the
    repository) and is not generated at build time (pcms/configuration.hpp,
    pcms/version.hpp);
  * an include that reaches into the PCMS sources (`src/` or `..` paths) and
    does not name a file of the examples.

Includes of other libraries (the standard library, Kokkos, Omega_h, MPI) and of
a header that is in the examples (common/example_utils.hpp) are fine. C++, C,
Fortran and Python files are read whole, Markdown files only in their fenced
code blocks, so that a README can talk about `detail::` in prose. The exit
status is 1 if anything is reported.
"""

import argparse
import os
import re
import sys

CODE_SUFFIXES = (".cpp", ".cc", ".hpp", ".h", ".c", ".f90", ".py")
SKIPPED_DIRS = {".git", "__pycache__", "CMakeFiles"}
# generated into the build tree and installed next to the public headers
GENERATED = {"pcms/configuration.hpp", "pcms/version.hpp"}

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
INTERNAL_NAMESPACE = re.compile(r"\b(detail|impl)::|\bpcms\.detail\b")
FENCE = re.compile(r"^\s*(```|~~~)")


def skipped(name):
    return name in SKIPPED_DIRS or name.startswith("build") or name.startswith(
        "_build")


def code_files(paths):
    """The files to read: code files, and Markdown files, below the paths."""
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if not skipped(d))
            for name in sorted(filenames):
                if name.endswith(CODE_SUFFIXES + (".md",)):
                    yield os.path.join(dirpath, name)


def code_lines(path):
    """(line number, text) of the lines to check in a file."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not path.endswith(".md"):
        yield from enumerate(lines, 1)
        return
    inside = False
    for number, line in enumerate(lines, 1):
        if FENCE.match(line):
            inside = not inside
        elif inside:
            yield number, line


def check_include(path, number, bracket, name, include_dir, example_dirs):
    """The problem with an include, or None."""
    if re.search(r"(^|/)detail/", name):
        return f"includes the internal header <{name}>"
    if name.startswith("pcms/"):
        if (not os.path.exists(os.path.join(include_dir, name))
                and name not in GENERATED):
            return f"includes <{name}>, which is not a public header"
        return None
    if re.search(r"(^|/)src/|(^|/)\.\./", name):
        here = os.path.dirname(path)
        if not any(os.path.exists(os.path.join(d, name))
                   for d in [here] + example_dirs):
            return f"includes <{name}>, which reaches into the PCMS sources"
    return None


def check(paths, include_dir):
    problems = []
    examples = [p for p in paths if os.path.isdir(p)]
    example_dirs = [os.path.join(p, "common") for p in examples] + examples
    count = 0
    for path in code_files(paths):
        count += 1
        for number, line in code_lines(path):
            where = f"{path}:{number}"
            match = INCLUDE.match(line)
            if match:
                problem = check_include(path, number, match.group(1),
                                        match.group(2), include_dir,
                                        example_dirs)
                if problem:
                    problems.append(f"{where}: {problem}")
            for found in INTERNAL_NAMESPACE.finditer(line):
                problems.append(
                    f"{where}: uses {found.group(0)}, which is internal")
    return count, problems


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*",
                        default=[os.path.join(root, "examples")])
    parser.add_argument("--include-dir",
                        default=os.path.join(root, "include"),
                        help="the directory with the installed pcms/ headers")
    args = parser.parse_args()

    count, problems = check([os.path.abspath(p) for p in args.paths],
                            os.path.abspath(args.include_dir))
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} use(s) of PCMS internals in the examples")
        return 1
    print(f"examples: {count} files use the public API only")
    return 0


if __name__ == "__main__":
    sys.exit(main())

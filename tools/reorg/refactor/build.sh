#!/bin/bash
# build.sh <bindir>
# Builds pcms-refactor into <bindir>. Needs Clang and the Clang/LLVM 19
# development libraries (libclang-cpp and LLVM, with llvm-config on PATH or
# given as LLVM_CONFIG).
set -euo pipefail
bindir=$1
here=$(dirname "$(realpath "$0")")
llvm_config=${LLVM_CONFIG:-llvm-config}
version=$("$llvm_config" --version)
case $version in
  19.*) ;;
  *) echo "build.sh: needs LLVM 19, $llvm_config reports $version" >&2; exit 1 ;;
esac
mkdir -p "$bindir"
${CXX:-clang++} -O2 -g0 $("$llvm_config" --cxxflags) \
  "$here/pcms_refactor.cpp" -o "$bindir/pcms-refactor" \
  $("$llvm_config" --ldflags) -lclang-cpp $("$llvm_config" --libs) \
  $("$llvm_config" --system-libs)
echo "built $bindir/pcms-refactor"

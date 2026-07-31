#pragma once

#include <cstddef>

namespace pcms::transient::protocol
{

enum class Command
{
  hello = 1,
  save,
  restore,
  advance,
  get_field,
  set_boundary,
  get_scalar,
  shutdown,
  ok
};

inline constexpr std::size_t HeaderSize = 2;
inline constexpr std::size_t HelloSize = 5;

} // namespace pcms::transient::protocol

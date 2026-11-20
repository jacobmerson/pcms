#ifndef PCMS_UTILITY_COMMUNICATOR_HPP
#define PCMS_UTILITY_COMMUNICATOR_HPP

#include "pcms/utility/types.hpp"
#include <memory>
#include <type_traits>

namespace pcms
{

/// @brief A set of ranks that cooperate in collective operations.
///
/// Communicators are immutable and shared: hold them as
/// `std::shared_ptr<const Communicator>`. Two communicators are the same
/// (same ranks, same partition context) if the pointers are equal. This header
/// does not depend on MPI: Self() always exists, and communicators over MPI
/// ranks are made with CreateCommunicator in pcms/utility/mpi.hpp (builds with
/// PCMS_ENABLE_MPI).
class Communicator : public std::enable_shared_from_this<Communicator>
{
public:
  Communicator(const Communicator&) = delete;
  Communicator& operator=(const Communicator&) = delete;
  virtual ~Communicator() noexcept = default;

  /// @return the communicator of this process alone (rank 0 of 1)
  [[nodiscard]] static const std::shared_ptr<const Communicator>& Self();

  /// @return the rank of this process in the communicator
  [[nodiscard]] virtual int Rank() const noexcept = 0;

  /// @return the number of ranks
  [[nodiscard]] virtual int Size() const noexcept = 0;

  /// @brief Collective: the sum of `value` over the ranks.
  /// @tparam T Real, LO or GO
  template <typename T>
  [[nodiscard]] T SumAll(T value) const
  {
    AllReduce(Reduction::Sum, TypeOf<T>(), &value);
    return value;
  }

  /// @brief Collective: the maximum of `value` over the ranks.
  /// @tparam T Real, LO or GO
  template <typename T>
  [[nodiscard]] T MaxAll(T value) const
  {
    AllReduce(Reduction::Max, TypeOf<T>(), &value);
    return value;
  }

  /// @brief Collective: partitions the ranks as MPI_Comm_split does.
  /// @param color ranks with the same color are in the same new communicator;
  ///        a negative color gives no communicator
  /// @param key orders the ranks of a new communicator (ties by rank)
  /// @return the communicator of this rank's color, which keeps this one
  ///         alive; null for a negative color
  [[nodiscard]] std::shared_ptr<const Communicator> Split(int color,
                                                          int key) const
  {
    return DoSplit(color, key);
  }

protected:
  Communicator() = default;

  enum class Reduction
  {
    Sum,
    Max
  };

  /// Reduces the one value of type `type` at `value` over the ranks, in place.
  virtual void AllReduce(Reduction op, Type type, void* value) const = 0;

  /// The implementation of Split; a derived class passes shared_from_this()
  /// to what it creates to keep itself alive.
  [[nodiscard]] virtual std::shared_ptr<const Communicator> DoSplit(
    int color, int key) const = 0;

private:
  template <typename T>
  static constexpr Type TypeOf() noexcept
  {
    static_assert(std::is_same_v<T, Real> || std::is_same_v<T, LO> ||
                    std::is_same_v<T, GO>,
                  "reductions are for Real, LO and GO");
    if constexpr (std::is_same_v<T, Real>) {
      return Type::Real;
    } else if constexpr (std::is_same_v<T, LO>) {
      return Type::LO;
    } else {
      return Type::GO;
    }
  }
};

} // namespace pcms

#endif // PCMS_UTILITY_COMMUNICATOR_HPP

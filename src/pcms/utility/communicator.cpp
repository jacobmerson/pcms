#include "pcms/utility/communicator.hpp"

namespace pcms
{
namespace
{

// the one rank of Self(): reductions are the identity
class SerialCommunicator final : public Communicator
{
public:
  [[nodiscard]] int Rank() const noexcept override { return 0; }
  [[nodiscard]] int Size() const noexcept override { return 1; }

protected:
  void AllReduce(Reduction, Type, void*) const override {}
  [[nodiscard]] std::shared_ptr<const Communicator> DoSplit(
    int color, int) const override
  {
    return color < 0 ? nullptr : Self();
  }
};

} // namespace

const std::shared_ptr<const Communicator>& Communicator::Self()
{
  static const std::shared_ptr<const Communicator> self =
    std::make_shared<SerialCommunicator>();
  return self;
}

} // namespace pcms

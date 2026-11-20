#include "pcms/utility/mpi.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/mpi_type.hpp"

#include <string>
#include <utility>

namespace pcms
{
namespace
{

void CheckMPI(int error, const char* call)
{
  if (error != MPI_SUCCESS) {
    throw pcms_error(std::string(call) + " failed");
  }
}

// owns a duplicate of an MPI communicator (or one MPI_Comm_split made); a
// split keeps its parent alive
class MPICommunicator final : public Communicator
{
public:
  MPICommunicator(MPI_Comm owned, std::shared_ptr<const Communicator> parent)
    : comm_(owned), parent_(std::move(parent))
  {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }

  ~MPICommunicator() noexcept override
  {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) {
      MPI_Comm_free(&comm_);
    }
  }

  [[nodiscard]] int Rank() const noexcept override { return rank_; }
  [[nodiscard]] int Size() const noexcept override { return size_; }
  [[nodiscard]] MPI_Comm Get() const noexcept { return comm_; }

protected:
  void AllReduce(Reduction op, Type type, void* value) const override
  {
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    switch (type) {
      case Type::Real: datatype = utility::GetMPIType(Real{}); break;
      case Type::LO: datatype = utility::GetMPIType(LO{}); break;
      case Type::GO: datatype = utility::GetMPIType(GO{}); break;
      default: throw pcms_error("Communicator: reduction of unsupported type");
    }
    CheckMPI(MPI_Allreduce(MPI_IN_PLACE, value, 1, datatype,
                           op == Reduction::Sum ? MPI_SUM : MPI_MAX, comm_),
             "MPI_Allreduce");
  }

  [[nodiscard]] std::shared_ptr<const Communicator> DoSplit(
    int color, int key) const override
  {
    MPI_Comm split = MPI_COMM_NULL;
    CheckMPI(MPI_Comm_split(comm_, color < 0 ? MPI_UNDEFINED : color, key,
                            &split),
             "MPI_Comm_split");
    if (split == MPI_COMM_NULL) {
      return nullptr;
    }
    return std::make_shared<MPICommunicator>(split, shared_from_this());
  }

private:
  MPI_Comm comm_;
  std::shared_ptr<const Communicator> parent_;
  int rank_ = 0;
  int size_ = 1;
};

} // namespace

std::shared_ptr<const Communicator> CreateCommunicator(MPI_Comm comm)
{
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized == 0 || finalized != 0) {
    throw pcms_error("CreateCommunicator: MPI is not initialized");
  }
  if (comm == MPI_COMM_NULL) {
    throw pcms_error("CreateCommunicator: the communicator is MPI_COMM_NULL");
  }
  MPI_Comm duplicate = MPI_COMM_NULL;
  CheckMPI(MPI_Comm_dup(comm, &duplicate), "MPI_Comm_dup");
  return std::make_shared<MPICommunicator>(duplicate, nullptr);
}

MPI_Comm GetMPIComm(const Communicator& comm) noexcept
{
  const auto* mpi = dynamic_cast<const MPICommunicator*>(&comm);
  return mpi != nullptr ? mpi->Get() : MPI_COMM_SELF;
}

} // namespace pcms

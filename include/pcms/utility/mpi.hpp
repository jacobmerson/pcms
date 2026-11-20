#ifndef PCMS_UTILITY_MPI_HPP
#define PCMS_UTILITY_MPI_HPP

#include "pcms/configuration.hpp"

#ifndef PCMS_ENABLE_MPI
#error "pcms/utility/mpi.hpp needs a PCMS built with PCMS_ENABLE_MPI"
#endif

#include "pcms/utility/communicator.hpp"
#include <memory>
#include <mpi.h>

namespace pcms
{

/// @brief Collective over `comm`: a communicator over the ranks of an MPI
///        communicator.
///
/// The MPI communicator is duplicated (MPI_Comm_dup), so PCMS collectives never
/// share a message space with the application's; the duplicate is freed with
/// the last reference to the result. If that is after MPI_Finalize the free is
/// skipped (the MPI library is gone), so a communicator held in a static does
/// not crash at exit.
/// @param comm a valid communicator (not MPI_COMM_NULL); MPI is initialized
/// @throws pcms_error if MPI is not initialized or finalized, or comm is null
[[nodiscard]] std::shared_ptr<const Communicator> CreateCommunicator(
  MPI_Comm comm);

/// @return the MPI communicator of `comm`: the duplicate made by
///         CreateCommunicator (or the one Split made), MPI_COMM_SELF for
///         Self(). Valid while `comm` is alive; do not free it.
[[nodiscard]] MPI_Comm GetMPIComm(const Communicator& comm) noexcept;

} // namespace pcms

#endif // PCMS_UTILITY_MPI_HPP

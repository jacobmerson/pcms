#include "pcms/transfer/petsc_solver.hpp"
#include "pcms/field/point_rows.hpp"
#include "pcms/transfer/petsc_utils.hpp"
#include "pcms/utility/assert.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <petscksp.h>
#include <memory>
#include <type_traits>

namespace pcms::transfer::detail
{

namespace
{

void Check(PetscErrorCode ierr)
{
  if (ierr != PETSC_SUCCESS) {
    throw pcms_error("PetscSolver: PETSc error " + std::to_string(ierr));
  }
}

// Destroys a PETSc object.
template <auto Destroy>
struct PetscDestroy
{
  template <typename Handle>
  void operator()(Handle handle) const
  {
    Destroy(&handle);
  }
};

template <typename Handle, auto Destroy>
using PetscPtr =
  std::unique_ptr<std::remove_pointer_t<Handle>, PetscDestroy<Destroy>>;

template <typename T>
using DeviceView = Kokkos::View<T*, DeviceMemorySpace>;
using UnmanagedScalars = Kokkos::View<PetscScalar*, DeviceMemorySpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

constexpr bool DeviceIsHost =
  Kokkos::SpaceAccessibility<Kokkos::HostSpace, DeviceMemorySpace>::accessible;

// the array of `vector` in DeviceMemorySpace
void RequireDeviceArray(PetscMemType memtype)
{
  if (DeviceIsHost != static_cast<bool>(PetscMemTypeHost(memtype))) {
    throw pcms_error("PetscSolver: the PETSc vector is not in the memory "
                     "space of the PCMS device");
  }
}

// The owned global ids in ascending order: row r of the matrix is the r-th.
DeviceView<GO> SortedGids(GlobalIDView<DeviceMemorySpace> gids)
{
  DeviceView<GO> sorted(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "petsc_sorted_gids"),
    gids.size());
  Kokkos::parallel_for(
    "pcms::PetscSolver::CopyGids",
    Kokkos::RangePolicy<DefaultExecutionSpace>(
      0, static_cast<LO>(sorted.extent(0))),
    KOKKOS_LAMBDA(const LO i) { sorted(i) = gids[i]; });
  Kokkos::sort(sorted);
  return sorted;
}

// the row of `gid`, or -1 if no owned holder has it (PETSc ignores the entry)
KOKKOS_INLINE_FUNCTION PetscInt RowOf(const DeviceView<GO>& sorted, GO gid)
{
  LO lo = 0;
  LO hi = static_cast<LO>(sorted.extent(0));
  while (lo < hi) {
    const LO mid = lo + (hi - lo) / 2;
    if (sorted(mid) < gid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < static_cast<LO>(sorted.extent(0)) && sorted(lo) == gid
           ? static_cast<PetscInt>(lo)
           : PetscInt{-1};
}

// the row of each owned holder
DeviceView<PetscInt> OwnedRows(GlobalIDView<DeviceMemorySpace> gids,
                               const DeviceView<GO>& sorted)
{
  DeviceView<PetscInt> rows(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "petsc_owned_rows"),
    gids.size());
  Kokkos::parallel_for(
    "pcms::PetscSolver::OwnedRows",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0,
                                               static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO o) { rows(o) = RowOf(sorted, gids[o]); });
  return rows;
}

// The COO entries of E^T W E, a point's n x n entries contiguous; rows and
// columns are the ranks of the holders' global ids among the owned ones.
struct Coo
{
  DeviceView<PetscInt> rows;
  DeviceView<PetscInt> columns;
  DeviceView<PetscScalar> values;
};

Coo AssembleCoo(const pcms::detail::PointRows& rows,
                GlobalIDView<DeviceMemorySpace> gids,
                const DeviceView<GO>& sorted,
                const Kokkos::View<const Real*, DeviceMemorySpace>& weights)
{
  const auto num_points = static_cast<LO>(rows.offsets.extent(0)) - 1;
  const auto offsets = rows.offsets;
  DeviceView<LO> start("coo_start", num_points + 1);
  Kokkos::parallel_scan(
    "pcms::PetscSolver::CooOffsets",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_points + 1),
    KOKKOS_LAMBDA(const LO p, LO& partial, const bool final) {
      if (final) {
        start(p) = partial;
      }
      if (p < num_points) {
        const LO n = offsets(p + 1) - offsets(p);
        partial += n * n;
      }
    });
  LO nnz = 0;
  Kokkos::deep_copy(nnz, Kokkos::subview(start, num_points));
  Coo coo{DeviceView<PetscInt>("coo_rows", nnz),
          DeviceView<PetscInt>("coo_columns", nnz),
          DeviceView<PetscScalar>("coo_values", nnz)};
  const auto holders = rows.holders;
  const auto entries = rows.values;
  const auto out_rows = coo.rows;
  const auto out_columns = coo.columns;
  const auto out_values = coo.values;
  Kokkos::parallel_for(
    "pcms::PetscSolver::Coo",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_points),
    KOKKOS_LAMBDA(const LO p) {
      const LO first = offsets(p);
      const LO n = offsets(p + 1) - first;
      for (LO i = 0; i < n; ++i) {
        for (LO j = 0; j < n; ++j) {
          const LO k = start(p) + i * n + j;
          out_rows(k) = RowOf(sorted, gids[holders(first + i)]);
          out_columns(k) = RowOf(sorted, gids[holders(first + j)]);
          out_values(k) = weights(p) * entries(first + i) * entries(first + j);
        }
      }
    });
  return coo;
}

// x = B^-1 b by a KSP on the assembled B, one solve per component
class PetscSolver final : public LinearSolver
{
public:
  PetscSolver(const BilinearForm& form, const std::string& options_prefix)
  {
    const auto& layout = *form.TestSpace()->GetLayout();
    if (layout.IsDistributed()) {
      throw pcms_error("PetscSolver: the assembled solve is serial; the "
                       "layout is distributed");
    }
    const auto* source = pcms::detail::AsRowsSource(form.TestEvaluator());
    if (source == nullptr) {
      throw pcms_error("PetscSolver: the assembled solve needs an evaluator "
                       "with evaluation rows");
    }
    const auto gids = layout.GetGids();
    const auto sorted = SortedGids(gids);
    size_ = static_cast<PetscInt>(sorted.extent(0));
    owned_rows_ = OwnedRows(gids, sorted);

    const auto coo = AssembleCoo(source->Rows(), source->HolderGids(), sorted,
                                 form.GetQuadrature().weights);
    const auto nnz = static_cast<PetscCount>(coo.rows.extent(0));
    // the symbolic phase of PETSc runs on the host
    auto coo_rows =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, coo.rows);
    auto coo_columns =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, coo.columns);
    Mat matrix = nullptr;
    Check(createSeqAIJMat(PETSC_COMM_SELF, size_, size_, 0, nullptr, &matrix));
    matrix_.reset(matrix);
    Check(MatSetPreallocationCOO(matrix_.get(), nnz, coo_rows.data(),
                                 coo_columns.data()));
    Check(MatSetValuesCOO(matrix_.get(), coo.values.data(), INSERT_VALUES));
    Vec vector = nullptr;
    Check(createSeqVec(PETSC_COMM_SELF, size_, &vector));
    rhs_.reset(vector);
    Check(createSeqVec(PETSC_COMM_SELF, size_, &vector));
    solution_.reset(vector);

    KSP ksp = nullptr;
    Check(KSPCreate(PETSC_COMM_SELF, &ksp));
    ksp_.reset(ksp);
    Check(KSPSetOperators(ksp_.get(), matrix_.get(), matrix_.get()));
    Check(KSPSetType(ksp_.get(), KSPPREONLY));
    PC pc = nullptr;
    Check(KSPGetPC(ksp_.get(), &pc));
    Check(PCSetType(pc, PCLU));
    if (!options_prefix.empty()) {
      Check(KSPSetOptionsPrefix(ksp_.get(), options_prefix.c_str()));
    }
    Check(KSPSetFromOptions(ksp_.get()));
    Check(KSPSetUp(ksp_.get()));
  }

  void Solve(const Field<Real>& functional,
             Field<Real>& solution) const override
  {
    const auto b = functional.GetDOFHolderData().GetValues();
    const auto rows = owned_rows_;
    Kokkos::View<Real**, DeviceMemorySpace> x(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "petsc_solution"),
      b.extent(0), b.extent(1));
    for (std::size_t c = 0; c < b.extent(1); ++c) {
      PetscScalar* rhs_array = nullptr;
      PetscMemType memtype{};
      Check(VecGetArrayWriteAndMemType(rhs_.get(), &rhs_array, &memtype));
      RequireDeviceArray(memtype);
      const UnmanagedScalars rhs(rhs_array, static_cast<std::size_t>(size_));
      Kokkos::deep_copy(rhs, PetscScalar(0));
      Kokkos::parallel_for(
        "pcms::PetscSolver::Pack",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, rows.extent(0)),
        KOKKOS_LAMBDA(const LO o) { rhs(rows(o)) = b(o, c); });
      Kokkos::fence();
      Check(VecRestoreArrayWriteAndMemType(rhs_.get(), &rhs_array));
      Check(KSPSolve(ksp_.get(), rhs_.get(), solution_.get()));
      const PetscScalar* result_array = nullptr;
      Check(
        VecGetArrayReadAndMemType(solution_.get(), &result_array, &memtype));
      RequireDeviceArray(memtype);
      const Kokkos::View<const PetscScalar*, DeviceMemorySpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        result(result_array, static_cast<std::size_t>(size_));
      Kokkos::parallel_for(
        "pcms::PetscSolver::Unpack",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, rows.extent(0)),
        KOKKOS_LAMBDA(const LO o) { x(o, c) = result(rows(o)); });
      Kokkos::fence();
      Check(VecRestoreArrayReadAndMemType(solution_.get(), &result_array));
    }
    solution.SetDOFHolderDataUnchecked(MakeConstRank2View(x));
    solution.Synchronize();
  }

private:
  PetscInt size_ = 0;
  DeviceView<PetscInt> owned_rows_;
  // destroyed in reverse order: the KSP before the matrix it uses
  PetscPtr<Mat, MatDestroy> matrix_;
  PetscPtr<Vec, VecDestroy> rhs_;
  PetscPtr<Vec, VecDestroy> solution_;
  PetscPtr<KSP, KSPDestroy> ksp_;
};

} // namespace

std::unique_ptr<LinearSolver> CreatePetscSolver(
  const BilinearForm& form, const std::string& options_prefix)
{
  return std::make_unique<PetscSolver>(form, options_prefix);
}

} // namespace pcms::transfer::detail

#include "pcms/backends/omega_h/discretization.hpp"
#include "pcms/utility/types.hpp"
#include <type_traits>
#include "pcms/utility/assert.hpp"
#include "pcms/utility/mesh_geometry.hpp"
#include <string>

namespace pcms::backend::omega_h
{
// pcms and Omega_h exchange data without conversion
static_assert(std::is_same_v<Real, Omega_h::Real>,
              "pcms and Omega_h real types must match");
static_assert(std::is_same_v<LO, Omega_h::LO>,
              "pcms and Omega_h LO types must match");
static_assert(std::is_same_v<GO, Omega_h::GO>,
              "pcms and Omega_h GO types must match");

static_assert(std::is_same_v<ClassificationId, Omega_h::ClassId>,
              "ClassificationId must match Omega_h::ClassId for direct views");
static_assert(
  std::is_same_v<ClassificationDimension, Omega_h::I8>,
  "ClassificationDimension must match Omega_h::I8 for direct views");

OmegaHDiscretization::OmegaHDiscretization(
  Omega_h::Mesh& mesh, std::string global_id_name,
  Omega_h::Read<Omega_h::I8> owned_mask)
  : mesh_(mesh),
    global_id_name_(std::move(global_id_name)),
    owned_mask_(std::move(owned_mask))
{
  if (owned_mask_.size() != 0 && owned_mask_.size() != mesh_.nverts()) {
    throw pcms_error("OmegaHDiscretization: the owned mask must have one "
                     "entry per mesh vertex");
  }
  for (int entity_dim = 0; entity_dim <= max_entity_dim_; ++entity_dim) {
    class_dims_by_dim_[entity_dim] = Omega_h::Read<Omega_h::I8>();
    class_ids_by_dim_[entity_dim] = Omega_h::Read<Omega_h::ClassId>();
  }

  for (int entity_dim = 0; entity_dim <= mesh_.dim(); ++entity_dim) {
    class_dims_by_dim_[entity_dim] = Omega_h::Read<Omega_h::I8>(
      mesh_.get_array<Omega_h::I8>(entity_dim, "class_dim"));
    class_ids_by_dim_[entity_dim] = Omega_h::Read<Omega_h::ClassId>(
      mesh_.get_array<Omega_h::ClassId>(entity_dim, "class_id"));
  }
}

namespace detail
{

std::shared_ptr<const OmegaHDiscretization> CheckDiscretizationOf(
  Omega_h::Mesh& mesh,
  std::shared_ptr<const OmegaHDiscretization> discretization,
  const char* caller)
{
  PCMS_ALWAYS_ASSERT(discretization != nullptr);
  if (&discretization->GetMesh() != &mesh) {
    throw pcms_error(std::string(caller) +
                     ": the discretization is of another mesh than the "
                     "layout's");
  }
  return discretization;
}

} // namespace detail

bool OmegaHDiscretization::SameEntities(
  const Discretization& other) const noexcept
{
  auto p = dynamic_cast<const OmegaHDiscretization*>(&other);
  return p != nullptr && &p->mesh_ == &mesh_;
}

int OmegaHDiscretization::GetDimension() const
{
  return mesh_.dim();
}

LO OmegaHDiscretization::GetNumEntities(int entity_dim) const
{
  if (entity_dim < 0 || entity_dim > mesh_.dim())
    return 0;
  return mesh_.nents(entity_dim);
}

Rank1View<const ClassificationDimension, DeviceMemorySpace>
OmegaHDiscretization::GetEntityClassificationDimensions(int entity_dim) const
{
  if (entity_dim < 0 || entity_dim > max_entity_dim_)
    return Rank1View<const ClassificationDimension, DeviceMemorySpace>(nullptr,
                                                                       0);
  const auto& dims = class_dims_by_dim_[entity_dim];
  return Rank1View<const ClassificationDimension, DeviceMemorySpace>(
    dims.data(), dims.size());
}

Rank1View<const ClassificationId, DeviceMemorySpace>
OmegaHDiscretization::GetEntityClassificationIds(int entity_dim) const
{
  if (entity_dim < 0 || entity_dim > max_entity_dim_)
    return Rank1View<const ClassificationId, DeviceMemorySpace>(nullptr, 0);
  const auto& ids = class_ids_by_dim_[entity_dim];
  return Rank1View<const ClassificationId, DeviceMemorySpace>(ids.data(),
                                                              ids.size());
}

} // namespace pcms::backend::omega_h

#pragma once

#include "pcms/transient/participant.hpp"

#include <mfem.hpp>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace testing::support
{

struct FEMSystem
{
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  std::unique_ptr<mfem::ParGridFunction> temperature;
  std::unique_ptr<mfem::ParBilinearForm> operator_form;
  std::unique_ptr<mfem::ParLinearForm> rhs;
  mfem::Array<int> essential_true_dofs;
};

FEMSystem MakeSteadyHeatSystem(mfem::ParMesh& mesh, int order, double kappa,
                               std::span<const double> dirichlet_x);
double SolveSteadyHeat(FEMSystem& system, double relative_tolerance = 1e-12,
                       int maximum_iterations = 500);
std::vector<double> ExtractLine(const mfem::ParMesh& mesh,
                                const mfem::ParGridFunction& field, double x,
                                double tolerance = 1e-12);
void SetLine(mfem::ParMesh& mesh, mfem::ParGridFunction& field, double x,
             std::span<const double> values, double tolerance = 1e-12);
void SetLine(mfem::ParMesh& mesh, mfem::ParGridFunction& field, double x,
             double value, double tolerance = 1e-12);
double MaximumExactError(const mfem::ParMesh& mesh,
                         const mfem::ParGridFunction& field);

} // namespace testing::support

namespace testing
{

class MFEMParticipant final : public pcms::transient::Participant
{
public:
  struct Configuration
  {
    std::string name;
    std::string produced_interface;
    std::string consumed_interface;
    double produced_x = 0.0;
    double consumed_x = 0.0;
    double physical_boundary_x = 0.0;
    double physical_temperature = 0.0;
    double initial_temperature = 280.0;
    double conductivity = 1.0;
  };

  MFEMParticipant(MPI_Comm communicator, std::unique_ptr<mfem::Mesh> mesh,
                  Configuration configuration);

  [[nodiscard]] std::string_view Name() const override;
  void AdvanceTo(pcms::Real target_time) override;

  [[nodiscard]] pcms::transient::Checkpoint Save() const override;
  void Restore(const pcms::transient::Checkpoint& checkpoint) override;

  [[nodiscard]] pcms::transient::InterfaceState GetInterface(
    std::string_view name) const override;
  void SetInterface(
    std::string_view name,
    const pcms::transient::InterfaceState& state) override;

  [[nodiscard]] pcms::transient::Capabilities GetCapabilities()
    const override;

  [[nodiscard]] double MaximumExactError() const;
  [[nodiscard]] std::size_t InterfaceSize() const;
  [[nodiscard]] mfem::ParMesh& Mesh() noexcept;
  [[nodiscard]] mfem::ParFiniteElementSpace& Space() noexcept;
  [[nodiscard]] mfem::ParGridFunction& Temperature() noexcept;

  [[nodiscard]] const std::optional<std::vector<pcms::Real>>&
  FirstConsumedInterface() const;
  [[nodiscard]] const std::optional<std::vector<pcms::Real>>&
  FirstProducedAfterAdvance() const;

private:
  struct SavedState
  {
    pcms::Real time = 0.0;
    std::vector<double> temperature;
    std::vector<pcms::Real> consumed;
  };

  std::unique_ptr<mfem::Mesh> serial_mesh_;
  std::unique_ptr<mfem::ParMesh> parallel_mesh_;
  Configuration configuration_;
  support::FEMSystem system_;
  pcms::Real time_ = 0.0;
  std::vector<pcms::Real> consumed_;
  std::optional<std::vector<pcms::Real>> first_consumed_;
  std::optional<std::vector<pcms::Real>> first_produced_after_advance_;
};

} // namespace testing

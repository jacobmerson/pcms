#include "testing/mfem_participant.h"

#include "pcms/transient/participant_adapter/participant_client.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/coupler/overlap_mask.h"
#include "pcms/field/function_space/mfem.h"

#include <redev.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace testing::support
{
namespace
{

int DecodeDof(int dof)
{
  return dof >= 0 ? dof : -1 - dof;
}

struct VertexOnLine
{
  double y;
  int vertex;
};

std::vector<VertexOnLine> FindLineVertices(const mfem::ParMesh& mesh, double x,
                                           double tolerance)
{
  std::vector<VertexOnLine> vertices;
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
    const double* coordinate = mesh.GetVertex(vertex);
    if (std::abs(coordinate[0] - x) <= tolerance) {
      vertices.push_back(
        VertexOnLine{mesh.SpaceDimension() > 1 ? coordinate[1] : 0.0, vertex});
    }
  }
  std::sort(vertices.begin(), vertices.end(),
            [](const VertexOnLine& lhs, const VertexOnLine& rhs) {
              return lhs.y < rhs.y;
            });
  return vertices;
}

int VertexDof(const mfem::ParFiniteElementSpace& space, int vertex)
{
  mfem::Array<int> dofs;
  space.GetVertexDofs(vertex, dofs);
  if (dofs.Size() != 1) {
    throw std::runtime_error(
      "MFEM overlap participant requires scalar order-one vertex fields");
  }
  return DecodeDof(dofs[0]);
}

} // namespace

FEMSystem MakeSteadyHeatSystem(mfem::ParMesh& mesh, int order, double kappa,
                               std::span<const double> dirichlet_x)
{
  FEMSystem system;
  system.fec =
    std::make_unique<mfem::H1_FECollection>(order, mesh.Dimension());
  system.fes =
    std::make_unique<mfem::ParFiniteElementSpace>(&mesh, system.fec.get());
  if (system.fes->GetVDim() != 1 || system.fes->GetNDofs() != mesh.GetNV()) {
    throw std::runtime_error(
      "MFEM overlap participant requires a scalar order-one H1 space");
  }

  system.temperature =
    std::make_unique<mfem::ParGridFunction>(system.fes.get());
  *system.temperature = 280.0;

  mfem::ConstantCoefficient conductivity(kappa);
  system.operator_form =
    std::make_unique<mfem::ParBilinearForm>(system.fes.get());
  system.operator_form->AddDomainIntegrator(
    new mfem::DiffusionIntegrator(conductivity));
  system.operator_form->Assemble();
  system.operator_form->Finalize();

  mfem::ConstantCoefficient zero(0.0);
  system.rhs = std::make_unique<mfem::ParLinearForm>(system.fes.get());
  system.rhs->AddDomainIntegrator(new mfem::DomainLFIntegrator(zero));
  system.rhs->Assemble();

  std::vector<int> true_dofs;
  for (double x : dirichlet_x) {
    for (const VertexOnLine& entry : FindLineVertices(mesh, x, 1e-12)) {
      const int local_dof = VertexDof(*system.fes, entry.vertex);
      const int true_dof = system.fes->GetLocalTDofNumber(local_dof);
      if (true_dof >= 0)
        true_dofs.push_back(true_dof);
    }
  }
  std::sort(true_dofs.begin(), true_dofs.end());
  true_dofs.erase(std::unique(true_dofs.begin(), true_dofs.end()),
                  true_dofs.end());
  system.essential_true_dofs.SetSize(static_cast<int>(true_dofs.size()));
  for (int i = 0; i < static_cast<int>(true_dofs.size()); ++i)
    system.essential_true_dofs[i] = true_dofs[i];

  return system;
}

double SolveSteadyHeat(FEMSystem& system, double relative_tolerance,
                       int maximum_iterations)
{
  mfem::OperatorPtr matrix;
  mfem::HypreParVector solution;
  mfem::HypreParVector rhs;
  system.operator_form->FormLinearSystem(
    system.essential_true_dofs, *system.temperature, *system.rhs, matrix,
    solution, rhs);

  auto* hypre_matrix = matrix.As<mfem::HypreParMatrix>();
  if (hypre_matrix == nullptr)
    throw std::runtime_error("MFEM did not produce a Hypre matrix");

  mfem::HypreBoomerAMG preconditioner(*hypre_matrix);
  preconditioner.SetPrintLevel(0);

  mfem::CGSolver solver(system.fes->GetComm());
  solver.SetOperator(*hypre_matrix);
  solver.SetPreconditioner(preconditioner);
  solver.SetRelTol(relative_tolerance);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(maximum_iterations);
  solver.SetPrintLevel(0);
  solver.Mult(rhs, solution);

  system.operator_form->RecoverFEMSolution(solution, *system.rhs,
                                           *system.temperature);
  return solver.GetFinalNorm();
}

std::vector<double> ExtractLine(const mfem::ParMesh& mesh,
                                const mfem::ParGridFunction& field, double x,
                                double tolerance)
{
  const auto vertices = FindLineVertices(mesh, x, tolerance);
  if (vertices.empty())
    throw std::runtime_error("requested MFEM interface line is empty");

  std::vector<double> result;
  result.reserve(vertices.size());
  const mfem::ParFiniteElementSpace& space = *field.ParFESpace();
  for (const VertexOnLine& entry : vertices)
    result.push_back(field(VertexDof(space, entry.vertex)));
  return result;
}

void SetLine(mfem::ParMesh& mesh, mfem::ParGridFunction& field, double x,
             std::span<const double> values, double tolerance)
{
  const auto vertices = FindLineVertices(mesh, x, tolerance);
  if (vertices.size() != values.size())
    throw std::runtime_error("MFEM interface trace size mismatch");

  const mfem::ParFiniteElementSpace& space = *field.ParFESpace();
  for (std::size_t i = 0; i < vertices.size(); ++i)
    field(VertexDof(space, vertices[i].vertex)) = values[i];
}

void SetLine(mfem::ParMesh& mesh, mfem::ParGridFunction& field, double x,
             double value, double tolerance)
{
  const auto vertices = FindLineVertices(mesh, x, tolerance);
  if (vertices.empty())
    throw std::runtime_error("requested MFEM boundary line is empty");

  const mfem::ParFiniteElementSpace& space = *field.ParFESpace();
  for (const VertexOnLine& entry : vertices)
    field(VertexDof(space, entry.vertex)) = value;
}

double MaximumExactError(const mfem::ParMesh& mesh,
                         const mfem::ParGridFunction& field)
{
  const mfem::ParFiniteElementSpace& space = *field.ParFESpace();
  double error = 0.0;
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
    const double expected = 270.0 + 30.0 * mesh.GetVertex(vertex)[0];
    error =
      std::max(error, std::abs(field(VertexDof(space, vertex)) - expected));
  }
  return error;
}

} // namespace testing::support

namespace testing
{

MFEMParticipant::MFEMParticipant(MPI_Comm communicator,
                                 std::unique_ptr<mfem::Mesh> mesh,
                                 Configuration configuration)
  : serial_mesh_(std::move(mesh)),
    parallel_mesh_(
      std::make_unique<mfem::ParMesh>(communicator, *serial_mesh_)),
    configuration_(std::move(configuration)),
    system_(support::MakeSteadyHeatSystem(
      *parallel_mesh_, 1, configuration_.conductivity,
      std::vector<double>{configuration_.physical_boundary_x,
                          configuration_.consumed_x}))
{
  *system_.temperature = configuration_.initial_temperature;
  support::SetLine(*parallel_mesh_, *system_.temperature,
                   configuration_.physical_boundary_x,
                   configuration_.physical_temperature);
  consumed_ = support::ExtractLine(*parallel_mesh_, *system_.temperature,
                                   configuration_.consumed_x);
}

std::string_view MFEMParticipant::Name() const
{
  return configuration_.name;
}

void MFEMParticipant::AdvanceTo(pcms::Real target_time)
{
  if (target_time < time_)
    throw std::invalid_argument("MFEMParticipant cannot advance backwards");

  support::SetLine(*parallel_mesh_, *system_.temperature,
                   configuration_.physical_boundary_x,
                   configuration_.physical_temperature);
  support::SetLine(*parallel_mesh_, *system_.temperature,
                   configuration_.consumed_x, consumed_);
  support::SolveSteadyHeat(system_);
  time_ = target_time;

  if (!first_produced_after_advance_) {
    first_produced_after_advance_ =
      support::ExtractLine(*parallel_mesh_, *system_.temperature,
                           configuration_.produced_x);
  }
}

pcms::transient::Checkpoint MFEMParticipant::Save() const
{
  SavedState saved;
  saved.time = time_;
  saved.temperature.resize(
    static_cast<std::size_t>(system_.temperature->Size()));
  for (int i = 0; i < system_.temperature->Size(); ++i)
    saved.temperature[static_cast<std::size_t>(i)] = (*system_.temperature)(i);
  saved.consumed = consumed_;
  return pcms::transient::Checkpoint{time_, std::move(saved)};
}

void MFEMParticipant::Restore(
  const pcms::transient::Checkpoint& checkpoint)
{
  const auto& saved = std::any_cast<const SavedState&>(checkpoint.state);
  if (saved.temperature.size() !=
      static_cast<std::size_t>(system_.temperature->Size())) {
    throw std::runtime_error("MFEMParticipant checkpoint size mismatch");
  }

  time_ = saved.time;
  consumed_ = saved.consumed;
  for (int i = 0; i < system_.temperature->Size(); ++i)
    (*system_.temperature)(i) =
      saved.temperature[static_cast<std::size_t>(i)];
}

pcms::transient::InterfaceState MFEMParticipant::GetInterface(
  std::string_view name) const
{
  if (name != configuration_.produced_interface)
    throw std::invalid_argument("MFEMParticipant: unknown produced interface");
  return pcms::transient::InterfaceState(support::ExtractLine(
    *parallel_mesh_, *system_.temperature, configuration_.produced_x));
}

void MFEMParticipant::SetInterface(
  std::string_view name, const pcms::transient::InterfaceState& state)
{
  if (name != configuration_.consumed_interface)
    throw std::invalid_argument("MFEMParticipant: unknown consumed interface");
  if (state.Size() != 0 && state.Size() != InterfaceSize())
    throw std::invalid_argument("MFEMParticipant: interface size mismatch");

  if (state.Size() == 0) {
    // PCMS FieldCommunicator has already deserialized the received field
    // directly into system_.temperature. Capture the artificial-boundary
    // trace for the next solve/checkpoint.
    consumed_ = support::ExtractLine(
      *parallel_mesh_, *system_.temperature, configuration_.consumed_x);
  } else {
    consumed_.assign(state.View().begin(), state.View().end());
  }
  if (!first_consumed_)
    first_consumed_ = consumed_;
}

pcms::transient::Capabilities MFEMParticipant::GetCapabilities() const
{
  return pcms::transient::Capabilities{/*can_restart=*/true,
                                       /*has_dense_output=*/false,
                                       /*reports_qoi=*/false};
}

double MFEMParticipant::MaximumExactError() const
{
  return support::MaximumExactError(*parallel_mesh_, *system_.temperature);
}

std::size_t MFEMParticipant::InterfaceSize() const
{
  return support::ExtractLine(*parallel_mesh_, *system_.temperature,
                              configuration_.consumed_x)
    .size();
}

mfem::ParMesh& MFEMParticipant::Mesh() noexcept
{
  return *parallel_mesh_;
}

mfem::ParFiniteElementSpace& MFEMParticipant::Space() noexcept
{
  return *system_.fes;
}

mfem::ParGridFunction& MFEMParticipant::Temperature() noexcept
{
  return *system_.temperature;
}

const std::optional<std::vector<pcms::Real>>&
MFEMParticipant::FirstConsumedInterface() const
{
  return first_consumed_;
}

const std::optional<std::vector<pcms::Real>>&
MFEMParticipant::FirstProducedAfterAdvance() const
{
  return first_produced_after_advance_;
}

} // namespace testing

namespace
{

void Log(const std::string& role, const std::string& message)
{
  std::cout << "[participant " << role << "] " << message << '\n';
}

std::string ValuesSummary(std::span<const double> values)
{
  if (values.empty())
    return "size=0";
  const auto [minimum, maximum] =
    std::minmax_element(values.begin(), values.end());
  double sum = 0.0;
  for (const double value : values)
    sum += value;

  std::ostringstream output;
  output << std::setprecision(12) << "size=" << values.size()
         << ", min=" << *minimum << ", max=" << *maximum
         << ", mean=" << sum / static_cast<double>(values.size())
         << ", first=" << values.front() << ", last=" << values.back();
  return output.str();
}

testing::MFEMParticipant::Configuration ConfigurationForRole(
  const std::string& role)
{
  if (role == "A") {
    return {"mfem_participant_A", "temperature_volume", "right_at_0.6",
            0.4, 0.6, 0.0, 270.0};
  }
  if (role == "B") {
    return {"mfem_participant_B", "temperature_volume", "left_at_0.4",
            0.6, 0.4, 1.0, 300.0};
  }
  throw std::invalid_argument(
    "mfem_participant role must be A or B");
}

int RunParticipant(const std::string& role, const std::string& mesh_path)
{
  const std::string channel_name = "mfem_participant_" + role;
  const auto configuration = ConfigurationForRole(role);
  Log(role, "starting; mesh=" + mesh_path);
  auto mesh = std::make_unique<mfem::Mesh>(mesh_path.c_str(), 1, 1);
  Log(role, "loaded serial MFEM mesh: dimension=" +
            std::to_string(mesh->Dimension()) + ", vertices=" +
            std::to_string(mesh->GetNV()) + ", elements=" +
            std::to_string(mesh->GetNE()));
  testing::MFEMParticipant participant(
    MPI_COMM_WORLD, std::move(mesh), configuration);
  Log(role, "configured heat problem: physical boundary x=" +
            std::to_string(configuration.physical_boundary_x) +
            " at T=" +
            std::to_string(configuration.physical_temperature) +
            ", produced interface x=" +
            std::to_string(configuration.produced_x) +
            ", consumed interface x=" +
            std::to_string(configuration.consumed_x));
  Log(role, "parallel finite-element space: local vertices=" +
            std::to_string(participant.Mesh().GetNV()) +
            ", local DOFs=" +
            std::to_string(participant.Space().GetNDofs()) +
            ", true DOFs=" +
            std::to_string(participant.Space().GlobalTrueVSize()) +
            ", interface DOFs=" +
            std::to_string(participant.InterfaceSize()));

  redev::Redev redev(MPI_COMM_WORLD, redev::ProcessType::Client);
  pcms::transient::ParticipantClient client(
    redev, participant,
    {channel_name, "temperature_volume",
     role == "A" ? "right_at_0.6" : "left_at_0.4",
     participant.InterfaceSize(),
     [&participant] { return participant.MaximumExactError(); }});
  Log(role, "created transient command channel '" + channel_name + "'");

  pcms::Coupler field_coupler(
    "mfem_overlap_fields", MPI_COMM_WORLD, false, {});
  auto* field_app = field_coupler.AddApplication(
    channel_name + "_field", "", redev::TransportType::SST);
  constexpr int overlap_attribute = 1;
  const std::string field_name = "temperature_volume";
  pcms::MFEMFieldFactory field_factory(
    participant.Mesh(), participant.Space(), participant.Temperature(),
    pcms::CoordinateSystem::Cartesian);
  auto overlap = pcms::MFEMLayout::OverlapMaskFromAttribute(
    participant.Mesh(), overlap_attribute);
  const std::size_t overlap_dofs = static_cast<std::size_t>(
    std::count(overlap.data(), overlap.data() + overlap.size(), true));
  Log(role, "MFEM field layout: owned DOFs=" +
            std::to_string(
              field_factory.GetLayout()->GetNumOwnedDofHolder()) +
            ", overlap DOFs=" + std::to_string(overlap_dofs) +
            ", overlap element attribute=" +
            std::to_string(overlap_attribute));
  field_app->SetLayoutOverlapMask(
    field_name,
    std::make_unique<pcms::OverlapMask>(
      static_cast<std::size_t>(
        field_factory.GetLayout()->GetNumOwnedDofHolder()),
      overlap));
  field_app->AddLayout(field_name, field_factory.GetLayout());
  auto field = field_app->AddField(
    field_name, field_factory.CreateField<pcms::Real>());

  int send_count = 0;
  int receive_count = 0;
  client.ConfigureFieldExchange({
    [field_app, field, &participant, &role, &send_count] {
      ++send_count;
      const auto produced = participant.GetInterface("temperature_volume");
      Log(role, "field send #" + std::to_string(send_count) +
                  ": produced trace " +
                  ValuesSummary(produced.View()));
      field_app->SendPhase([&] { field.Send(); });
      Log(role, "field send #" + std::to_string(send_count) +
                  ": complete");
    },
    [field_app, field, &participant, &role, &receive_count] {
      ++receive_count;
      Log(role, "field receive #" + std::to_string(receive_count) +
                  ": begin");
      field_app->ReceivePhase([&] { field.Receive(); });
      const auto consumed = testing::support::ExtractLine(
        participant.Mesh(), participant.Temperature(),
        role == "A" ? 0.6 : 0.4);
      Log(role, "field receive #" + std::to_string(receive_count) +
                  ": consumed trace " + ValuesSummary(consumed));
    }});
  Log(role, "field exchange configured; entering participant command loop");
  client.Run();
  Log(role, "shutdown received; field sends=" +
            std::to_string(send_count) + ", field receives=" +
            std::to_string(receive_count) + ", maximum exact error=" +
            std::to_string(participant.MaximumExactError()));
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int return_code = EXIT_SUCCESS;
  try {
    if (argc != 3)
      throw std::invalid_argument(
        "Usage: mfem_participant <A|B> <participant.mesh>");
    if (size != 1)
      throw std::invalid_argument(
        "mfem_participant currently requires one MPI rank per process");
    return_code = RunParticipant(argv[1], argv[2]);
  } catch (const std::exception& error) {
    if (rank == 0)
      std::cerr << "mfem_participant: " << error.what() << '\n';
    return_code = EXIT_FAILURE;
  }

  Kokkos::finalize();
  MPI_Finalize();
  return return_code;
}

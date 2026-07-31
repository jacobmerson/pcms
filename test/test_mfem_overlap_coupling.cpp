#include "pcms/transient/transient.hpp"
#include "pcms/transient/participant_adapter/coupler_mesh_backend.hpp"
#include "pcms/transient/participant_adapter/remote_participant.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/field/function_space/lagrange.h"
#include "pcms/transfer/interpolator.h"
#include "pcms/utility/arrays.h"

#include <Omega_h_array.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <redev.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tr = pcms::transient;

namespace
{

void Log(const std::string& message)
{
  std::cout << "[coupler] " << message << '\n';
}

std::string StateSummary(std::span<const pcms::Real> values)
{
  if (values.empty())
    return "size=0";
  const auto [minimum, maximum] =
    std::minmax_element(values.begin(), values.end());
  pcms::Real sum = 0.0;
  for (const pcms::Real value : values)
    sum += value;

  std::ostringstream output;
  output << std::setprecision(12) << "size=" << values.size()
         << ", min=" << *minimum << ", max=" << *maximum
         << ", mean=" << sum / static_cast<pcms::Real>(values.size())
         << ", first=" << values.front() << ", last=" << values.back();
  return output.str();
}

std::string FieldSummary(const pcms::Field<pcms::Real>& field)
{
  const auto values = field.GetDOFHolderDataHost();
  std::vector<pcms::Real> copied(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    copied[i] = values[i];
  return StateSummary(copied);
}

std::unique_ptr<Omega_h::Mesh> ReadCouplerMesh(
  Omega_h::Library& library, const std::string& path)
{
  Log("reading Omega_h mesh: " + path);
  auto mesh = std::make_unique<Omega_h::Mesh>(&library);
  Omega_h::binary::read(path, library.world(), mesh.get());
  Log("loaded mesh '" + path + "': dimension=" +
      std::to_string(mesh->dim()) + ", vertices=" +
      std::to_string(mesh->nverts()) + ", elements=" +
      std::to_string(mesh->nelems()));
  return mesh;
}

tr::InterfaceState ExtractVerticalTrace(
  const Omega_h::Mesh& mesh, const pcms::Field<pcms::Real>& field,
  double x)
{
  constexpr double tolerance = 1e-10;
  if (mesh.dim() < 2)
    throw std::invalid_argument(
      "MFEM overlap test requires two-dimensional coupler meshes");
  const Omega_h::HostRead<Omega_h::Real> coordinates(mesh.coords());
  const auto values = field.GetDOFHolderDataHost();
  std::vector<std::pair<double, pcms::Real>> trace;
  for (int vertex = 0; vertex < mesh.nverts(); ++vertex) {
    if (std::abs(coordinates[mesh.dim() * vertex] - x) <= tolerance) {
      trace.emplace_back(coordinates[mesh.dim() * vertex + 1],
                         values[vertex]);
    }
  }
  std::sort(trace.begin(), trace.end());
  if (trace.empty())
    throw std::runtime_error("coupler target interface is empty");

  std::vector<pcms::Real> result;
  result.reserve(trace.size());
  for (const auto& [coordinate, value] : trace) {
    static_cast<void>(coordinate);
    result.push_back(value);
  }
  return tr::InterfaceState(std::move(result));
}

void SetVerticalTrace(
  const Omega_h::Mesh& mesh, pcms::Field<pcms::Real>& field,
  double x, const tr::InterfaceState& state)
{
  constexpr double tolerance = 1e-10;
  const Omega_h::HostRead<Omega_h::Real> coordinates(mesh.coords());
  std::vector<std::pair<double, int>> trace_vertices;
  for (int vertex = 0; vertex < mesh.nverts(); ++vertex) {
    if (std::abs(coordinates[mesh.dim() * vertex] - x) <= tolerance) {
      trace_vertices.emplace_back(
        coordinates[mesh.dim() * vertex + 1], vertex);
    }
  }
  std::sort(trace_vertices.begin(), trace_vertices.end());
  if (trace_vertices.size() != state.Size())
    throw std::runtime_error("coupler interface trace size mismatch");

  const auto current = field.GetDOFHolderDataHost();
  Kokkos::View<pcms::Real*, pcms::HostMemorySpace> updated(
    "updated_coupler_trace", current.extent(0));
  for (std::size_t i = 0; i < current.extent(0); ++i)
    updated(i) = current(i);
  for (std::size_t i = 0; i < trace_vertices.size(); ++i)
    updated(trace_vertices[i].second) = state[i];
  field.SetDOFHolderDataHost(pcms::make_const_array_view(updated));
}

tr::CouplerMeshEndpoint MakeEndpoint(
  tr::Participant& participant, std::unique_ptr<Omega_h::Mesh> mesh,
  pcms::FieldHandle<pcms::Real> field, double target_x)
{
  return tr::CouplerMeshEndpoint{
    &participant,
    std::move(mesh),
    std::move(field),
    [target_x](const Omega_h::Mesh& target_mesh,
               const pcms::Field<pcms::Real>& field) {
      return ExtractVerticalTrace(target_mesh, field, target_x);
    },
    [](std::span<const pcms::Real> coordinate) {
      constexpr double tolerance = 1e-10;
      return coordinate[0] >= 0.4 - tolerance &&
             coordinate[0] <= 0.6 + tolerance;
    }};
}

bool Check(bool condition, const std::string& message)
{
  std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
  return condition;
}

} // namespace

int main(int argc, char** argv)
{
  Omega_h::Library library(&argc, &argv);
  const auto world = library.world();
  const int rank = world->rank();
  const int size = world->size();

  if (argc != 3) {
    if (rank == 0) {
      std::cerr
        << "Usage: " << argv[0]
        << " <coupler_A.osh> <coupler_B.osh>\n";
    }
    return EXIT_FAILURE;
  }

  if (size != 1) {
    if (rank == 0)
      std::cerr << "test_mfem_overlap_coupling runs on one MPI rank\n";
    return EXIT_FAILURE;
  }

  int return_code = EXIT_SUCCESS;
  try {
    Log("starting MFEM overlap-coupling server on " +
        std::to_string(size) + " MPI rank");
    Log("participant A coupler mesh: " + std::string(argv[1]));
    Log("participant B coupler mesh: " + std::string(argv[2]));

    const MPI_Comm communicator = world->get_impl();
    redev::LOs server_ranks{0};
    redev::Reals cuts{0.0, 0.0};
    redev::RCBPtn partition(2, server_ranks, cuts);
    redev::Redev redev(
      communicator, redev::Partition{std::move(partition)},
      redev::ProcessType::Server);

    tr::RemoteParticipant left(
      redev, {"mfem_participant_A", "left", "temperature_volume",
              "right_at_0.6", true});
    tr::RemoteParticipant right(
      redev, {"mfem_participant_B", "right", "temperature_volume",
              "left_at_0.4", true});
    Log("created remote participant channels; connection is deferred until "
        "the field data plane is configured");

    auto left_mesh = ReadCouplerMesh(library, argv[1]);
    auto right_mesh = ReadCouplerMesh(library, argv[2]);

    redev::LOs field_server_ranks{0};
    redev::Reals field_cuts{0.0, 0.0};
    redev::RCBPtn field_partition(
      2, field_server_ranks, field_cuts);
    pcms::Coupler field_coupler(
      "mfem_overlap_fields", communicator, true,
      redev::Partition{std::move(field_partition)});
    auto* left_field_app = field_coupler.AddApplication(
      "mfem_participant_A_field", "", redev::TransportType::SST);
    auto left_space = pcms::LagrangeFunctionSpace::FromMesh(
      *left_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    left_field_app->AddLayout(
      "temperature_volume", left_space.GetLayout());
    auto left_field = left_field_app->AddField(
      "temperature_volume", left_space.CreateField<pcms::Real>());

    auto* right_field_app = field_coupler.AddApplication(
      "mfem_participant_B_field", "", redev::TransportType::SST);
    auto right_space = pcms::LagrangeFunctionSpace::FromMesh(
      *right_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    right_field_app->AddLayout(
      "temperature_volume", right_space.GetLayout());
    auto right_field = right_field_app->AddField(
      "temperature_volume", right_space.CreateField<pcms::Real>());
    Log("registered coupler fields: left owned DOFs=" +
        std::to_string(
          left_field.GetField().GetLayout().GetNumOwnedDofHolder()) +
        ", right owned DOFs=" +
        std::to_string(
          right_field.GetField().GetLayout().GetNumOwnedDofHolder()));

    auto left_to_right_transfer = field_coupler.AddTransfer<pcms::Real>(
      left_field, right_field,
      std::make_unique<pcms::Interpolator<pcms::Real>>(
        left_space, right_space,
        pcms::OutOfBoundsPolicy{
          pcms::OutOfBoundsMode::FILL, 0.0}));
    auto right_to_left_transfer = field_coupler.AddTransfer<pcms::Real>(
      right_field, left_field,
      std::make_unique<pcms::Interpolator<pcms::Real>>(
        right_space, left_space,
        pcms::OutOfBoundsPolicy{
          pcms::OutOfBoundsMode::FILL, 0.0}));
    Log("registered bidirectional PCMS interpolation transfers with "
        "out-of-bounds fill value 0");

    Omega_h::Mesh* left_mesh_ptr = left_mesh.get();
    Omega_h::Mesh* right_mesh_ptr = right_mesh.get();
    int left_receive_count = 0;
    int left_send_count = 0;
    int right_receive_count = 0;
    int right_send_count = 0;
    left.ConfigureFieldExchange({
      [left_field_app, left_field, &left_receive_count] {
        ++left_receive_count;
        Log("A -> coupler receive #" +
            std::to_string(left_receive_count) + ": begin");
        left_field_app->ReceivePhase([&] { left_field.Receive(); });
        Log("A -> coupler receive #" +
            std::to_string(left_receive_count) + ": complete; field " +
            FieldSummary(left_field.GetField()));
      },
      [left_field_app, left_field, left_mesh_ptr, &left_send_count](
        const tr::InterfaceState& state) {
        ++left_send_count;
        Log("coupler -> A send #" + std::to_string(left_send_count) +
            ": imposing x=0.6 trace; " + StateSummary(state.View()));
        SetVerticalTrace(
          *left_mesh_ptr, left_field.GetField(), 0.6, state);
        left_field_app->SendPhase([&] { left_field.Send(); });
        Log("coupler -> A send #" + std::to_string(left_send_count) +
            ": complete");
      }});
    right.ConfigureFieldExchange({
      [right_field_app, right_field, &right_receive_count] {
        ++right_receive_count;
        Log("B -> coupler receive #" +
            std::to_string(right_receive_count) + ": begin");
        right_field_app->ReceivePhase([&] { right_field.Receive(); });
        Log("B -> coupler receive #" +
            std::to_string(right_receive_count) + ": complete; field " +
            FieldSummary(right_field.GetField()));
      },
      [right_field_app, right_field, right_mesh_ptr, &right_send_count](
        const tr::InterfaceState& state) {
        ++right_send_count;
        Log("coupler -> B send #" + std::to_string(right_send_count) +
            ": imposing x=0.4 trace; " + StateSummary(state.View()));
        SetVerticalTrace(
          *right_mesh_ptr, right_field.GetField(), 0.4, state);
        right_field_app->SendPhase([&] { right_field.Send(); });
        Log("coupler -> B send #" + std::to_string(right_send_count) +
            ": complete");
      }});

    auto backend = std::make_shared<tr::CouplerMeshBackend>(
      MakeEndpoint(left, std::move(left_mesh), left_field, 0.6),
      MakeEndpoint(right, std::move(right_mesh), right_field, 0.4),
      left_to_right_transfer, right_to_left_transfer);
    Log("coupling backend ready: meshes=" +
        std::to_string(backend->MeshCount()) + ", A overlap DOFs=" +
        std::to_string(backend->LeftOverlapDofCount()) +
        ", B overlap DOFs=" +
        std::to_string(backend->RightOverlapDofCount()));

    Log("waiting for participant handshakes");
    left.Connect();
    Log("connected participant A: interface DOFs=" +
        std::to_string(left.InterfaceSize()));
    right.Connect();
    Log("connected participant B: interface DOFs=" +
        std::to_string(right.InterfaceSize()));
    tr::Coupling left_to_right{
      &left, "temperature_volume", &right, "left_at_0.4",
      tr::IdentityTransfer(), {}};
    left_to_right.backend = backend;
    tr::Coupling right_to_left{
      &right, "temperature_volume", &left, "right_at_0.6",
      tr::IdentityTransfer(), {}};
    right_to_left.backend = backend;

    std::vector<tr::Participant*> participants{&left, &right};
    tr::CouplingSet couplings{left_to_right, right_to_left};

    tr::TransientOverlapCoupled simulation(
      participants, couplings, std::make_unique<tr::FixedTimestepper>(1.0),
      std::make_unique<tr::AcceleratedSequentialSchwarz>(
        std::make_unique<tr::AitkenRelaxation>(), 100, 1e-10),
      std::make_unique<tr::NoErrorEstimate>());
    Log("configured one window: dt=1, Schwarz max iterations=100, "
        "tolerance=1e-10, accelerator=Aitken");
    Log("capability guarantee: " + simulation.Guarantee());

    tr::WindowReport report;
    simulation.SetMonitor([&](const tr::WindowReport& current) {
      report = current;
      std::ostringstream output;
      output << std::setprecision(12)
             << "accepted window: t=[" << current.t_start << ", "
             << current.t_start + current.dt << "], attempts="
             << current.attempts << ", Schwarz iterations="
             << current.schwarz_iters << ", residual="
             << current.schwarz_residual << ", converged="
             << std::boolalpha << current.converged;
      Log(output.str());
    });
    Log("starting coupled simulation");
    simulation.Run(1.0);
    Log("coupled simulation complete");

    bool passed = true;
    passed &= Check(left.InterfaceSize() == right.InterfaceSize() &&
                      left.InterfaceSize() > 0,
                    "the geometric overlap has matching, nonempty traces");
    passed &= Check(backend->MeshCount() == 2 &&
                      backend->LeftOverlapDofCount() > 0 &&
                      backend->RightOverlapDofCount() > 0,
                    "the coupling backend owns two nonempty overlap meshes");
    passed &=
      Check(report.converged, "sequential Schwarz fixed point converged");
    passed &= Check(report.schwarz_residual <= 1e-10,
                    "interface fixed-point residual meets tolerance");
    const double left_exact_error = left.QueryScalar();
    const double right_exact_error = right.QueryScalar();
    Log("participant exact-solution errors: A=" +
        std::to_string(left_exact_error) + ", B=" +
        std::to_string(right_exact_error));
    passed &= Check(left_exact_error < 1e-8 &&
                      right_exact_error < 1e-8,
                    "coupled MFEM fields reproduce T(x) = 270 + 30x");

    Log("communication totals: A receives=" +
        std::to_string(left_receive_count) + ", A sends=" +
        std::to_string(left_send_count) + ", B receives=" +
        std::to_string(right_receive_count) + ", B sends=" +
        std::to_string(right_send_count));
    Log("Schwarz summary: iterations=" +
        std::to_string(report.schwarz_iters) + ", residual=" +
        std::to_string(report.schwarz_residual) + ", coupler sweeps=" +
        std::to_string(backend->CompletedSweeps()));

    Log("requesting participant shutdown");
    left.Shutdown();
    right.Shutdown();
    Log(std::string("test result: ") + (passed ? "PASS" : "FAIL"));
    return_code = passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "test_mfem_overlap_coupling: " << error.what() << '\n';
    return_code = EXIT_FAILURE;
  }

  return return_code;
}

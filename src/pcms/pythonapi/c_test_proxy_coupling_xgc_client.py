import c_pcms_python as pcms
import numpy as np
from mpi4py import MPI
import sys

def in_overlap(dimension, id):
    if (id >= 22 and id <= 34):
        if dimension == 2 or dimension == 1 or dimension == 0:
            return 1
    return 0

# Give pcms the MPI comm through
if __name__ == '__main__':
    MPI.Init() 
    pcms.pcms_kokkos_initialize_without_args()
    world_rank, world_size, plane_rank, plane_size, client_rank, client_size = -1, -1, -1, -1, -1, -1


    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    nplanes = 2  # You need to define the value of nplanes
    if world_size % nplanes != 0:   
        print("Number of ranks must be divisible by the number of planes")
        sys.exit("Aborting due to the invalid number of ranks")

    plane = world_rank % nplanes
    print(f"PLANE {plane}")

    plane_comm = MPI.COMM_WORLD.Split(plane, world_rank)
    plane_rank = plane_comm.rank
    plane_size = plane_comm.size
    client_comm = MPI.COMM_WORLD.Split(0 if plane_rank == 0 else MPI.UNDEFINED, world_rank)

    if client_comm != MPI.COMM_NULL:
        client_rank = client_comm.rank
        client_size = client_comm.size

    print(f"world: {world_rank} {world_size}; plane: {plane_rank} {plane_size}; client: {client_rank} {client_size}")
    client = pcms.pcms_create_client("proxy_couple", MPI._handleof(client_comm)) # passes C object communicator client_comm
    rc_file = sys.argv[1]
    rc = pcms.pcms_load_reverse_classification(rc_file, MPI._handleof(MPI.COMM_WORLD)) # passes C object communicator

    nverts = pcms.pcms_reverse_classification_count_verts(rc)
    data = np.zeros(nverts, dtype=np.int64)

    field = []
    field_adapters = []

    for i in range(nplanes):
        field_name = f"xgc_gids_plane_{i}"
        print(field_name)

    communicating_rank = (i == plane) and (plane_rank == 0)

    if plane == i:
        field_adapter = pcms.pcms_create_xgc_field_adapter(
            "adapter1", plane_comm, data, nverts, pcms.PCMS_LONG_INT, rc, in_overlap
        )
    else:
        field_adapter = pcms.pcms_create_dummy_field_adapter()
    field_adapters.append(field_adapter)
    field.append(pcms.pcms_add_field(client, field_name, field_adapter, communicating_rank))
    
    if plane_rank == 0:
        for i in range(nverts):
            data[i] = i

    pcms.pcms_begin_send_phase(client)
    pcms.pcms_send_field(field[plane])
    pcms.pcms_end_send_phase(client)
    pcms.pcms_begin_receive_phase(client)
    pcms.pcms_receive_field(field[plane])
    pcms.pcms_end_receive_phase(client)

    for i in range(nverts):
        if data[i] != i:
            print(f"ERROR: data[{i}] = {data[i]}, should be {i}")
            sys.exit()

    if plane_rank == 0:
        for i in range(nverts):
            data[i] *= 2

    pcms.pcms_begin_send_phase(client);
    pcms.pcms_send_field(field[plane]);
    pcms.pcms_end_send_phase(client);
    pcms.pcms_begin_receive_phase(client);
    pcms.pcms_receive_field(field[plane]);
    pcms.pcms_end_receive_phase(client);

    for i in range(nverts):
        if data[i] != 2 * i:
            print(f"ERROR: data[{i}] = {data[i]}, should be {2 * i}")
            sys.exit()

    for field_adapter in field_adapters:
        pcms.pcms_destroy_field_adapter(field_adapter)

    data = None
    pcms.pcms_destroy_reverse_classification(rc)
    pcms.pcms_destroy_client(client)
    if(client_comm != MPI.COMM_NULL):
        MPI.Comm.Free(client_comm)
  
    MPI.Comm.Free(plane_comm)
    pcms.pcms_kokkos_finalize()
    MPI.Finalize()



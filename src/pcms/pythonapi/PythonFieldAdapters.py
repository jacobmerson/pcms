import pcms_python as pcms
import numpy as np
from mpi4py import MPI

memory_space = pcms.PythonFieldAdapterConcept.memory_space

class PythonXGCFieldAdapter(pcms.PythonFieldAdapterConcept):

    def __init__(self, name, plane_communicator, data, reverse_classification, in_overlap):
        self.name_ = name
        self.plane_comm_ = plane_communicator
        self.plane_rank_ = None
        self.data_ = data
        self.gids_ = [0] * data.size()
        self.reverse_classification_ = None
        self.mask_ =  None
        self.in_overlap_ = in_overlap
        self.plane_root_ = 0
        self.memory_space = memory_space

        
        self.plane_rank_ = self.plane_comm_.Get_rank()
        if self.RankParticipatesCouplingCommunication():
            mask = pcms.PythonArrayMask(np.zeros_like(data, dtype=np.int8))
            assert callable(in_overlap)
            py_rcv = pcms.PyReverseClassificationVertex(reverse_classification)
            data = py_rcv.get_data()
            for (dim, id), verts in data.items(): # geom would be an iterator
                if in_overlap(dim,id): 
                    for vert in verts:
                        assert vert < len(data)
                        mask[vert] = 1

            self.reverse_classification_ = py_rcv
            self.mask_ = mask
            assert self.mask_.Size() != 0
            gids_ = np.arange(1, len(gids_) + 1, dtype=np.int64)

    def Serialize(self,buffer,permutation):
        assert self.memory_space == memory_space, "gpu space unhandled"
        if self.RankParticipatesCouplingCommunication():
            const_data = np.array(self.data_, dtype=self.data_.dtype, copy=False)
            if buffer.size() > 0:
                self.mask_.Apply(const_data, buffer, permutation)
            return self.mask_.Size()
        return 0
            
    def Deserialize(self,buffer,permutation):
        assert self.memory_space == memory_space, "gpu space unhandled"
        if self.RankParticipatesCouplingCommunication():
            self.mask_.ToFullArray(buffer, self.data_, permutation)

        #Bcast line
        pcms.Bcast(self.data_, self.plane_root_, self.plane_comm_) 

    # Representing the ScalarArrayViews as numpy arrays
    def GetGids(self):
        if self.RankParticipatesCouplingCommunication():
            gids = np.zeros(self.mask_.Size(), dtype=np.int64)
            self.mask_.Apply(self.gids_,gids)
            return gids
        return np.empty(0, dtype=np.int64) 

    def GetReversePartitionMap(self, partition):
        if self.RankParticipatesCouplingCommunication():
            reverse_partition = {}
            assert callable(self.in_overlap_)
            for (dim, id), verts in self.reverse_classification_.get_data().items():
                if self.in_overlap_(dim, id):
                    dr = pcms.RankParition(dim,id,partition)
                    it = reverse_partition.setdefault(dr, [])
                    inserted = not bool(it) #True if a new key-value pair was inserted, false otherwise
                    map_ = self.mask_.GetMap()
                    #Transformation
                    for v in verts:
                        idx = map_[v]
                        assert (idx > 0)
                        reverse_partition[dr].append(idx-1)
                    
            for rank, idxs in reverse_partition.iterms():
                reverse_partition[rank] = sorted(idxs)

            return reverse_partition
        return {}
    
    #optional
    def RankParticipatesCouplingCommunication(self):
        return self.plane_rank_ == self.plane_root_ 
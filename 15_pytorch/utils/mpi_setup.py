import os

import torch.distributed as dist

IS_MPI, ERR_MSG = False, ""
try:
    from mpi4py import MPI
    IS_MPI = True
except ImportError as ie:
    print(ie)
    ERR_MSG = ie


def mpi_init():
    if not IS_MPI:
        raise ImportError(ERR_MSG)
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()
    print("Rank: {}, Size: {}".format(comm.Get_rank(), comm.Get_size()))
    return comm, mpirank, mpisize


# mpi setup
def dist_setup(backend="nccl", is_mpi4py=False):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_PORT", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    if not is_mpi4py:
        is_mpi4py = os.getenv("OMPI_COMM_WORLD_SIZE") is None
    if backend == "mpi":
        rank, world_size = -1, -1
    elif is_mpi4py:
        _, rank, size = mpi_init()
    dist.init_process_group(backend=backend,
                            init_method=method,
                            rank=rank,
                            world_size=world_size)

    print("Rank: {}, Size: {}, Host: {} Port: {}".format(dist.get_rank(),
                                                         dist.get_world_size(),
                                                         master_addr, master_port))


def dist_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def myget_rank(comm=None):
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    if dist.is_initialized():
        rank = dist.get_rank()
    return rank


def myget_rank_size(comm=None):
    rank = myget_rank(comm)
    size = 1
    if comm is not None:
        size = comm.Get_size()
    if dist.is_initialized():
        size = dist.get_world_size()
    return rank, size


# multi process print
def print_rank(*args, comm=None):
    rank = myget_rank(comm)
    print(f"rank: {rank}", *args)


def print0(*args, comm=None):
    rank = myget_rank(comm)
    if rank == 0:
        print(*args)

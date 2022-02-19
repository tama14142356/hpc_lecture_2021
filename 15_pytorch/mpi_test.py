import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet
# import numpy as np

IS_MPI = False
ERR_MSG = ""
try:
    from mpi4py import MPI
    IS_MPI = True
except ImportError as ie:
    print(ie)
    ERR_MSG = ie

# from lib.utils import dist_setup, dist_cleanup, print_rank


def mpi_init():
    if not IS_MPI:
        raise ImportError(ERR_MSG)
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()
    print("Rank: {}, Size: {}".format(mpirank, mpisize))
    return comm, mpirank, mpisize


def dist_setup(backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_POST", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    if backend == "mpi":
        rank, world_size = -1, -1
    elif backend == "gloo":
        comm, rank, world_size = mpi_init()
        # rank = int(os.getenv("PMIX_RANK", "0"))
        # world_size = int(os.getenv("OMPI_UNIVERSE_SIZE", "1"))
    dist.init_process_group(backend=backend,
                            init_method=method,
                            rank=rank,
                            world_size=world_size)

    print("Rank: {}, Size: {}, Host: {}".format(dist.get_rank(), dist.get_world_size(),
                                                master_addr))


def dist_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def myget_rank_size(comm=None):
    rank, world_size = 0, 1
    if comm is not None:
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size


def myget_rank(comm=None):
    rank, _ = myget_rank_size(comm)
    return rank


# multi process print
def print0(*args, comm=None):
    rank = myget_rank(comm)
    if rank == 0:
        print(*args)


def print_rank(*args, comm=None):
    rank, world_size = myget_rank_size(comm)
    digit = len(str(world_size))
    str_rank = str(rank).zfill(digit)
    print(f"rank: {str_rank}", *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi_backend",
                        default="nccl",
                        choices=["nccl", "mpi", "gloo"])
    parser.add_argument("--is_mpi4py", action="store_true")
    args = parser.parse_args()

    backend = args.mpi_backend
    comm, rank, world_size = None, 0, 1

    print_rank("start check pytorch install", comm=comm)
    print_rank(f"version: {torch.__version__}", comm=comm)
    print_rank(f"cuda: {torch.cuda.is_available()}", comm=comm)
    x = torch.rand(5, 3)
    print_rank(f"rand: {x}", comm=comm)

    mpi_name = "mpi4py" if args.is_mpi4py else "dist"
    print_rank(f"start check {mpi_name} install", comm=comm)

    is_mpi4py = args.is_mpi4py
    if is_mpi4py:
        comm, rank, world_size = mpi_init()
    else:
        dist_setup(backend)
        world_size, rank = dist.get_world_size(), dist.get_rank()

    print_rank(backend, dist.is_available(), comm=comm)
    print_rank(f"setup {mpi_name} complete", comm=comm)

    s_time = time.perf_counter()
    is_cuda = torch.cuda.is_available()
    device_name = "cuda" if is_cuda else "cpu"
    if is_cuda:
        ngpus = torch.cuda.device_count()
        local_rank = rank % ngpus
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
    tmp = torch.rand(8, 1, 2, 3)
    # tmp = np.random.rand(8, 1, 2, 3)
    # os_env = os.environ

    print_rank(device_name, device, tmp.size(), comm=comm)
    # print_rank(f"{os_env}\n", comm=comm)

    x = torch.randn(1).to(device)
    # x = torch.randn(2).to(device)
    output = [torch.ones(1, ).to(device) for _ in range(world_size)]
    print_rank("x:", x, comm=comm)
    print_rank("output:", output, comm=comm)

    # print_rank("start broadcast")
    # dist.broadcast(x, src=0)
    # print_rank(x)

    print_rank("start all_gather", comm=comm)
    if is_mpi4py:
        output = comm.allgather(x)
    else:
        dist.all_gather(output, x)
    print_rank("x:", x, comm=comm)
    print_rank("output:", output, comm=comm)

    print_rank("start all_reduce", comm=comm)
    if is_mpi4py:
        output = comm.allreduce(x, op=MPI.SUM)
        y = torch.sum(output)
        nb = x.shape[0] * world_size
    else:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        y = torch.sum(x)
        nb = x.shape[0] * world_size
    mean = y / nb
    print_rank("x:", x, "y:", y, "nb:", nb, "mean:", mean, comm=comm)
    print_rank("output:", output, comm=comm)

    if not is_mpi4py:
        print_rank("start ddp", comm=comm)
        x = torch.rand(2, 3, 5, 6)
        print_rank("x:", x, comm=comm)
        model = resnet.resnet18(pretrained=False)
        print_rank("def resnet", comm=comm)
        model = model.to(device)
        print_rank("to device", comm=comm)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_rank("convert sync batch", comm=comm)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print_rank("convert ddp model", comm=comm)
        output = model(x)
        print_rank("x:", x, comm=comm)
        print_rank("output:", output, comm=comm)

    # print_rank("start gather", comm=comm)
    # dist.gather(x, gather_list=output, dst=0)
    # print_rank(x, comm=comm)

    end_time = time.perf_counter() - s_time
    print_rank("tmp exec time:", end_time, comm=comm)
    dist_cleanup()

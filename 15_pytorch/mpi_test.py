import argparse
import multiprocessing
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet
# import numpy as np

from utils.mpi_setup import dist_setup, dist_cleanup
from utils.mpi_setup import mpi_init
from utils.mpi_setup import print_rank

IS_BYOL = False
BYOL_ERR_MSG = ""
try:
    from byol_pytorch import BYOL
    IS_BYOL = True
except ImportError as ie:
    print(ie)
    BYOL_ERR_MSG = ie

IS_MPI, ERR_MSG = False, ""
try:
    from mpi4py import MPI
    IS_MPI = True
except ImportError as ie:
    print(ie)
    ERR_MSG = ie

# from lib.utils import dist_setup, dist_cleanup, print_rank

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

    cpus = multiprocessing.cpu_count()
    print_rank("cpus:", cpus, comm=comm)

    print_rank(backend, dist.is_available(), comm=comm)
    print_rank(f"setup {mpi_name} complete", comm=comm)
    print_rank("gpus:", torch.cuda.device_count(), comm=comm)

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
        x = x.to(device)
        # x = torch.rand(2, 3, 5, 6, requires_grad=True)
        print_rank("x:", x, comm=comm)
        model_resnet = resnet.resnet18(pretrained=False)
        print_rank("def resnet", comm=comm)
        model_resnet = model_resnet.to(device)
        print_rank("resnet to device", comm=comm)
        model_resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_resnet)
        print_rank("resnet convert sync batch", comm=comm)
        model_resnet = DDP(model_resnet,
                           device_ids=[local_rank],
                           output_device=local_rank)
        print_rank("convert resnet ddp model", comm=comm)
        output = model_resnet(x)
        print_rank("x:", x, comm=comm)
        print_rank("output:", output, comm=comm)
        # output.backward()
        # print_rank("resnet after backward:", torch.cuda.memory_allocated(), comm=comm)

        if IS_BYOL:
            del model_resnet
            print_rank("start byol", comm=comm)
            model_resnet = resnet.resnet50(pretrained=False)
            print_rank("def resnet50", comm=comm)
            # model_resnet = resnet.resnet18(pretrained=False)
            # print_rank("byol def resnet18", comm=comm)
            # model_resnet = model_resnet.to(device)
            # print_rank("byol to device", comm=comm)
            # model_resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_resnet)
            # print_rank("convert sync batch", comm=comm)
            # model_resnet = DDP(model_resnet,
            #                    device_ids=[local_rank],
            #                    output_device=local_rank)
            # print_rank("convert resnet ddp model", comm=comm)
            model_byol = BYOL(model_resnet,
                              256,
                              hidden_layer="avgpool",
                              projection_size=256,
                              projection_hidden_size=4096,
                              moving_average_decay=0.99,
                              use_momentum=True)
            print_rank("def byol", comm=comm)
            model_byol = model_byol.to(device)
            print_rank("byol to device", comm=comm)
            model_byol = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_byol)
            print_rank("byol convert sync batch", comm=comm)
            model_byol = DDP(model_byol,
                             device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True)
            print_rank("convert byol ddp model byol", comm=comm)
            model_byol.train()
            x = torch.randn(2, 3, 256, 256, requires_grad=True, dtype=torch.float32)
            x = x.to(device)
            output = model_byol(x)
            with torch.no_grad():
                print_rank("x:", x, comm=comm)
                print_rank("output:", output, comm=comm)
            output.backward()
            print_rank("byol after backward:", torch.cuda.memory_allocated(), comm=comm)

    # print_rank("start gather", comm=comm)
    # dist.gather(x, gather_list=output, dst=0)
    # print_rank(x, comm=comm)

    if rank == 0:
        print_rank("start memory summary", comm=comm)
        print_rank(torch.cuda.memory_allocated(), comm=comm)
        print_rank(torch.cuda.memory_reserved(), comm=comm)
        print_rank(torch.cuda.memory_summary(), comm=comm)

    end_time = time.perf_counter() - s_time
    print_rank("total exec time:", end_time, "s", comm=comm)
    dist_cleanup()

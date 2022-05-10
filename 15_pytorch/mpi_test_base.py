import os

import torch
import torch.distributed as dist


def dist_setup(backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_PORT", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    if backend == "mpi":
        rank, world_size = -1, -1
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


def myget_rank_size():
    rank, world_size = 0, 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size


def myget_rank():
    rank, _ = myget_rank_size()
    return rank


# multi process print
def print0(*args):
    rank = myget_rank()
    if rank == 0:
        print(*args)


def print_rank(*args):
    rank, world_size = myget_rank_size()
    digit = len(str(world_size))
    str_rank = str(rank).zfill(digit)
    print(f"rank: {str_rank}", *args)


if __name__ == "__main__":
    backend = "nccl"
    dist_setup(backend)
    rank, world_size = myget_rank_size()
    print("rank:", rank, "world_size:", world_size)

    ngpus = torch.cuda.device_count()
    device = rank % ngpus
    # x = torch.randn(1).to(device)
    x = torch.randn(2).to(device)
    output = [torch.ones(1, ).to(device) for _ in range(world_size)]
    print("rank {}: {}".format(rank, x))
    print("rank {}: {}".format(rank, output))
    # dist.broadcast(x, src=0)
    # dist.all_gather(output, x)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    y = torch.sum(x)
    nb = x.shape[0] * dist.get_world_size()
    mean = y / nb
    # nccl not supported gather func
    # if rank == 0:
    #     dist.gather(x, gather_list=output, dst=0)
    print("rank {}: {}".format(rank, x))
    print("rank {}: {}".format(rank, y))
    print("rank {}: {}".format(rank, mean))
    print("rank {}: {}".format(rank, output))
    dist_cleanup()

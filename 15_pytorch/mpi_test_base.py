import os

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
    world_size, rank = dist.get_world_size(), dist.get_rank()
    print("rank:", rank, "world_size:", world_size)

    dist_cleanup()

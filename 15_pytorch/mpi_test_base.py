import torch
import torch.distributed as dist

from utils.mpi_setup import dist_setup, dist_cleanup
from utils.mpi_setup import myget_rank_size

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

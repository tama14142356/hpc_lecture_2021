import os
import torch
import torch.distributed as dist


def get_device_and_comm_rank(no_cuda=False, distributed_backend='nccl'):
    # [COMM] get MPI rank
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", '1'))
    is_distributed = world_size > 1
    if is_distributed:
        # [COMM] initialize process group
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend=distributed_backend,
                                world_size=world_size,
                                rank=rank,
                                init_method=init_method)
    if not no_cuda and torch.cuda.is_available():
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print("Rank : {}, Size : {}, Host: {}".format(rank, world_size, master_ip))
    return device, rank, world_size


device, rank, world_size = get_device_and_comm_rank()

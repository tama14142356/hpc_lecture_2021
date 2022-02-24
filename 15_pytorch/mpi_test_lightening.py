import os
import pytorch_lightning.plugins.environments.lightning_environment as le
from pytorch_lightning.utilities.distributed import init_dist_connection


def dist_setup(backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_POST", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    le_environment = le.LightningEnvironment()

    init_dist_connection(cluster_environment=le_environment,
                         torch_distributed_backend=backend,
                         global_rank=rank,
                         world_size=world_size,
                         init_method=method)


if __name__ == "__main__":
    dist_setup()

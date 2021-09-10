import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def backward_hook(self, in_grad, out_grad):
    self.in_grad = in_grad
    self.out_grad = out_grad


master_addr = os.getenv("MASTER_ADDR", default="localhost")
master_port = os.getenv('MASTER_PORT', default='8888')
method = "tcp://{}:{}".format(master_addr, master_port)
rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
dist.init_process_group("nccl",
                        init_method=method,
                        rank=rank,
                        world_size=world_size)
ngpus = torch.cuda.device_count()
device = torch.device('cuda', rank % ngpus)
x = torch.tensor([[1.], [2.]], requires_grad=True)
y = torch.tensor([[1.], [1.]])
model = torch.nn.Linear(1, 1, bias=False)
model = DDP(model, device_ids=[rank % ngpus])
for param in model.parameters():
    torch.nn.init.constant_(param, 2)
model.register_full_backward_hook(backward_hook)
criterion = torch.nn.MSELoss(reduction='mean')
y_p = model(x)
loss = criterion(y_p, y)
print('loss   :', loss.data)
loss.backward()

print(len(model.out_grad))
print('dl/dy:', model.out_grad[0].data)
print('dl/dx:', model.in_grad[0].data)
for param in model.parameters():
    print('grad   :', param.grad.data)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import numpy as np
import random

# class Test(nn.Module):
#     def __init__(self, num=0):
#         super().__init__()
#         self.num = 0
#         self.rank = 0
#         if dist.is_initialized():
#             self.rank = dist.get_rank()
#
#     @staticmethod
#     def get_test(name, num):
#         rank = 0
#         if dist.is_initialized():
#             rank = dist.get_rank()
#         print(f"rank{rank}: {name} {num}")
#         return f"rank{rank}: {name} {num}"
#
#     def forward(self, x):
#         self.num += 1
#         t = torch.rand(1)
#         print(f"rank{self.rank}: {self.num} {t}")
#         test_name = self.get_test("test", self.num)
#         print(f"{test_name}")
#         return x
#
#
# class ChildTest(Test):
#     def __init__(self, num=1, is_super=False, name="child"):
#         super().__init__(num)
#         self.num, self.name = num, name
#         self.is_super = is_super
#
#     def get_test(self, name, num):
#         if self.is_super:
#             return Test.get_test(name, num) + "\n"
#         super_name = ""
#         super_name += f"rank{self.rank}: {name} child {num}"
#         print(super_name)
#         return super_name


def backward_hook(self, in_grad, out_grad):
    self.in_grad = in_grad
    self.out_grad = out_grad
    print0(f"backward: {self}")
    for data in in_grad:
        print0(f"ingrad: {data.size()}")
    for data in out_grad:
        print0(f"outgrad: {data.size()}")


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
            # print("init", flush=True)
    else:
        print(message, flush=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.test = Test(0)
        # self.test = ChildTest(1)

    def forward(self, x):
        # self.test(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(train_loader, model, criterion, optimizer, epoch, device, world_size):
    model.train()
    t = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print0('Train Epoch: {} [{:>5}/{} ({:.0%})]\tLoss: {:.6f}\t Time:{:.4f}'.
                   format(epoch,
                          batch_idx * len(data) * world_size, len(train_loader.dataset),
                          batch_idx / len(train_loader), loss.data.item(),
                          time.perf_counter() - t))
            t = time.perf_counter()
    # state_dict = model.state_dict(keep_vars=True)
    # print0(f"state dict: type: {type(state_dict)}")
    # for key in state_dict:
    #     params = state_dict[key]
    #     grad = params.grad
    #     print0(f"state dict: grad type: {type(grad)}")


def validate(val_loader, model, criterion, device):
    model.eval()
    val_loss, val_acc = 0, 0
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()
        pred = output.data.max(1)[1]
        val_acc += 100. * pred.eq(target.data).cpu().sum() / target.size(0)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    print0('\nValidation set: Average loss: {:.4f}, Accuracy: {:.1f}%\n'.format(
        val_loss, val_acc))


def main():
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
    set_seed(0)

    epochs = 5
    batch_size = 8
    learning_rate = 1.0e-02

    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    train_dataset.data = train_dataset.data[:batch_size]
    train_dataset.targets = train_dataset.targets[:batch_size]
    val_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    val_dataset.data = val_dataset.data[:batch_size]
    val_dataset.targets = val_dataset.targets[:batch_size]
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    model = CNN().to(device)
    model = DDP(model, device_ids=[rank % ngpus])
    model_names = model.module._modules
    # model_names = model._modules
    print0(model_names)
    for model_name in model_names:
        module = model_names[model_name]
        # wandb_key = f"{model_name}({module.__class__.__name__})"
        # print(wandb_key)
        print0(module)
        module.register_full_backward_hook(backward_hook)
    # model.module.register_full_backward_hook(backward_hook)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train(train_loader, model, criterion, optimizer, epoch, device, world_size)
        validate(val_loader, model, criterion, device)
        for model_name in model_names:
            module = model_names[model_name]
            # wandb_key = f"{model_name}({module.__class__.__name__})"
            # print(wandb_key)
            # module.register_full_backward_hook(backward_hook)
            print0(module)
            print0(model_name)
            print0(len(module.out_grad))
            print0(len(module.in_grad))
            print0(f"epoch: {epoch} dl/dy: {module.out_grad[0].data.size()}")
            print0(f"epoch: {epoch} dl/dx: {module.in_grad[0].data.size()}")
            for data in module.in_grad:
                print0(data.size())
            for data in module.out_grad:
                print0(data.size())
            # print0(module.out_grad.size())
            # print0(module.in_grad.size())
        # print0(model.out_grad[0].data)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()

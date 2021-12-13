import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import numpy as np
import random

from utils import dist_setup, dist_cleanup, print0
from utils import myget_rank_size


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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
        if batch_idx % 200 == 0:
            print0('Train Epoch: {} [{:>5}/{} ({:.0%})]\tLoss: {:.6f}\t Time:{:.4f}'.
                   format(epoch,
                          batch_idx * len(data) * world_size, len(train_loader.dataset),
                          batch_idx / len(train_loader), loss.data.item(),
                          time.perf_counter() - t))
            t = time.perf_counter()


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
    backend = "nccl"
    # backend = "mpi"
    dist_setup(backend)
    rank, world_size = myget_rank_size()
    if backend == "nccl":
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda', rank % ngpus)
    else:
        # ngpus = world_size
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

    set_seed(0)

    epochs = 5
    batch_size = 8
    learning_rate = 1.0e-02

    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    # train_dataset.data = train_dataset.data[:batch_size]
    # train_dataset.targets = train_dataset.targets[:batch_size]
    val_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    # val_dataset.data = val_dataset.data[:batch_size]
    # val_dataset.targets = val_dataset.targets[:batch_size]
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    model = CNN().to(device)
    if backend == "nccl":
        model = DDP(model, device_ids=[rank % ngpus])
    else:
        model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train(train_loader, model, criterion, optimizer, epoch, device, world_size)
        validate(val_loader, model, criterion, device)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()

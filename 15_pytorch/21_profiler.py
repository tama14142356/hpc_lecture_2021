import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.profiler
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
import time


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


def train(train_loader, model, criterion, optimizer, epoch, device, logs):
    model.train()
    t = time.perf_counter()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=8),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logs),
            record_shapes=True,
            profile_memory=False) as prof:
        for batch_idx, (data, target) in enumerate(train_loader):
            print0("batch idx: {}".format(batch_idx))
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prof.step()
            print0('Train Epoch: {} [{:>5}/{} ({:.0%})]\tLoss: {:.6f}\t Time:{:.4f}'.
                   format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          batch_idx / len(train_loader), loss.data.item(),
                          time.perf_counter() - t))
            t = time.perf_counter()
            if batch_idx > 10:
                break


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


def main(args):
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

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    val_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    # shuffle=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sampler)
    # shuffle=False)
    model = CNN().to(device)
    model = DDP(model, device_ids=[rank % ngpus])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train(train_loader, model, criterion, optimizer, epoch, device, args.logs)
        validate(val_loader, model, criterion, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST profile Example')
    parser.add_argument('--logs',
                        type=str,
                        default="./logs",
                        help='profile log directory (default: ./logs)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1.0e-02,
                        metavar='LR',
                        help='learning rate (default: 1.0e-02)')
    args = parser.parse_args()
    main(args)

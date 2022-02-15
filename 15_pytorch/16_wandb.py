import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import wandb
import os
import glob
import webdataset as wds

from utils import dist_setup, dist_cleanup, myget_rank_size
from utils import print_rank


def print0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
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


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix="", postfix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.postfix = postfix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += self.postfix
        print0('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, criterion, optimizer, epoch, device, length):
    batch_time = AverageMeter('Time', ':.4f')
    train_loss = AverageMeter('Loss', ':.6f')
    train_acc = AverageMeter('Accuracy', ':.6f')
    progress = ProgressMeter(length, [train_loss, train_acc, batch_time],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    t = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        print_rank(batch_idx, target.size(), data.size())
        output = model(data)
        print_rank(batch_idx, output.size(), data.size())
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        train_acc.update(acc, data.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            cur_time = time.perf_counter() - t
            batch_time.update(cur_time)
            t = time.perf_counter()
            progress.display(batch_idx)
            # rank = 0
            # if dist.is_initialized():
            #     rank = dist.get_rank()
            # if rank == 0:
            #     wandb.log({"sec/batch": cur_time})
    return train_loss.avg, train_acc.avg, batch_time.avg


def validate(val_loader, model, criterion, device, length):
    val_loss = AverageMeter('Loss', ':.6f')
    val_acc = AverageMeter('Accuracy', ':.1f')
    progress = ProgressMeter(length, [val_loss, val_acc],
                             prefix='\nValidation: ',
                             postfix='\n')
    model.eval()
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        val_acc.update(acc, data.size(0))
    progress.display(length)
    return val_loss.avg, val_acc.avg


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--bs',
                        '--batch_size',
                        type=int,
                        default=32,
                        metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        '--learning_rate',
                        type=float,
                        default=1.0e-02,
                        metavar='LR',
                        help='learning rate (default: 1.0e-02)')
    parser.add_argument("--mpi_backend",
                        type=str,
                        default="nccl",
                        choices=["nccl", "mpi", "gloo"])
    parser.add_argument("--use_wds", action="store_true")
    args = parser.parse_args()

    dist_setup(backend=args.mpi_backend)
    rank, world_size = myget_rank_size()
    # is_cuda_avail = torch.cuda.is_available()
    if args.mpi_backend == "nccl":
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda', rank % ngpus)
    else:
        device = torch.device("cpu")

    if rank == 0:
        wandb.init(project="ssl_test_result", entity="tomo", name="wandb_mnist_tmp")
        wandb.config.update(args)

    if args.use_wds:
        # normalize = transforms.Normalize((0.1307, ), (0.3081, ))

        preproc = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])

        train_img_num = 60000
        train_length = train_img_num // args.bs
        train_tars = sorted(
            glob.glob(os.path.join("./data/webdatasets", "train", "*.tar")))
        train_dataset = wds.WebDataset(train_tars).decode("pill").to_tuple("jpg", "cls")
        train_dataset = train_dataset.map_tuple(preproc, lambda x: torch.tensor(x))

        val_img_num = 10000
        val_length = val_img_num // args.bs
        val_tars = sorted(glob.glob(os.path.join("./data/webdatasets", "val", "*.tar")))
        val_dataset = wds.WebDataset(val_tars).decode("pill").to_tuple("jpg", "cls")
        val_dataset = val_dataset.map_tuple(preproc, lambda x: torch.tensor(x))

        local_batch_size = args.bs // world_size
        train_loader = wds.WebLoader(train_dataset,
                                     num_workers=0,
                                     batch_size=local_batch_size)
        val_loader = wds.WebLoader(val_dataset,
                                   num_workers=0,
                                   batch_size=local_batch_size)
    else:
        train_dataset = datasets.MNIST('./data',
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())
        val_dataset = datasets.MNIST('./data',
                                     train=False,
                                     transform=transforms.ToTensor())
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.bs,
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.bs,
                                                 shuffle=False)
        train_length, val_length = len(train_loader), len(val_loader)

    model = CNN().to(device)
    if rank == 0:
        wandb.config.update({"model": model.__class__.__name__, "dataset": "MNIST"})
    if args.mpi_backend == "nccl":
        model = DDP(model, device_ids=[rank % ngpus])
    else:
        model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # data_plot = defaultdict(list)
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc, batch_time = train(train_loader, model, criterion,
                                                  optimizer, epoch, device,
                                                  train_length)
        val_loss, val_acc = validate(val_loader, model, criterion, device, val_length)
        if rank == 0:
            # data_plot["train_loss"].append([epoch, train_loss])
            # data_plot["train_acc"].append([epoch, train_acc])
            # data_plot["val_loss"].append([epoch, val_loss])
            # data_plot["val_acc"].append([epoch, val_acc])
            # data_plot["batch_time"].append([epoch, val_acc])
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'batch_time': batch_time
            })
    # if rank == 0:
    #     for key in data_plot:
    #         tmp_data = data_plot[key]
    #         table = wandb.Table(data=tmp_data, columns=["epoch", "value"])
    #         wandb.log(
    #             {f"{key}": wandb.plot.line(table, "epoch", "value", title=f"{key}")})

    dist_cleanup()


if __name__ == '__main__':
    main()

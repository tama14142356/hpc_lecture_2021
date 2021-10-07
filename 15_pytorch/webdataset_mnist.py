import argparse
# from itertools import islice
# from random import shuffle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import webdataset as wds


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = x.view(-1, D_in)
        h = self.fc1(x)
        h_r = F.relu(h)
        y_p = self.fc2(h_r)
        return F.log_softmax(y_p, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="mnist.tar", help="tar file path")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--debug_load_time", action="store_true")
    args = parser.parse_args()

    url = args.url

    epochs = 10
    batch_size = 32
    D_in = 784
    H = 100
    D_out = 10
    learning_rate = 1.0e-02

    torch.manual_seed(0)
    # read input data and labels

    # preproc = transforms.Compose([transforms.ToTensor])
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize((0.1307, ), (0.3081, ))

    preproc = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    print("start load..")
    if args.normal:
        train_dataset = datasets.MNIST(url,
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    else:
        train_dataset = (wds.WebDataset(url).decode("pill").to_tuple(
            "ppm", "cls").map_tuple(preproc, lambda x: torch.tensor(x)))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   num_workers=4,
                                                   batch_size=batch_size)

    # define model
    model = TwoLayerNet(D_in, H, D_out)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    lossv, accv, length = [], [], len(train_loader) if args.normal else 0

    if not args.normal:
        print("start calc length")
        t = time.perf_counter()
        for x, y in train_loader:
            length += 1
        print("calc time: {:.4f} length: {}".format(time.perf_counter() - t, length))

    print("start train")
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        t = time.perf_counter()
        tmp_t = t
        # Loop over each batch from the training set
        for batch_idx, (x, y) in enumerate(train_loader):
            if args.debug_load_time:
                print("batch_idx: {} load time: {:.4f}".format(
                    batch_idx,
                    time.perf_counter() - tmp_t))
                tmp_t = time.perf_counter()
            # print(batch_idx, x.size(), y.size(), y)
            # forward pass: compute predicted y
            y_p = model(x)

            # compute loss
            loss = criterion(y_p, y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{:>5}/{} ({:.0%})]\tLoss: {:.6f}\t Time:{:.4f}'.
                      format(epoch, batch_idx * len(x), length, batch_idx / length,
                             loss.data.item(),
                             time.perf_counter() - t))
                t = time.perf_counter()

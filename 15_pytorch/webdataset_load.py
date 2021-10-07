import argparse
from itertools import islice

import torch
# from torch.utils.data import IterableDataset
# import torchvision
from torchvision import transforms
from torchvision import datasets
import webdataset as wds
# from webdataset.iterators import shuffle

# from lib.datasets import BDD


def identity(x):
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="mnist.tar", help="tar file path")
    args = parser.parse_args()

    url = args.url

    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    print("0 sample")
    for img, target in train_dataset:
        print(img.shape, target)
        break
    print()
    raw_dataset = wds.WebDataset(url)
    print("1 sample")
    for sample in raw_dataset:
        print(type(sample))
        for key, value in sample.items():
            print(key, repr(value)[:50])
            print(type(value))
        break
    print()

    # dataset = raw_dataset.decode("pil")
    # dataset = raw_dataset.decode("rgb")
    # dataset = raw_dataset.decode("torchrgb")
    dataset = raw_dataset.decode("pill")
    print("2 sample")
    for sample in dataset:
        print(type(sample))
        for key, value in sample.items():
            print(key, repr(value)[:50])
            print(type(value))
        break
    print()

    dataset_tmp = dataset.to_tuple("ppm", "cls")
    print("3\' sample")
    for image, data in islice(dataset_tmp, 0, 5):
        print(image.size, type(image), type(data), data)
        print(image)
    print()
    dataset_tmp = dataset.shuffle(4).to_tuple("ppm", "cls")
    print("3\'\' sample")
    for image, data in islice(dataset_tmp, 0, 5):
        print(image.size, type(image), type(data), data)
        print(image)
    print()

    dataset = dataset.to_tuple("ppm", "cls")
    # dataset = dataset.to_tuple("jpg", "cls")
    print("3 sample")
    for sample in dataset:
        print(type(sample))
        for val in sample:
            print(val, type(val))
            if type(val) != int:
                print(val.size)
        break
    print()

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize((0.1307, ), (0.3081, ))

    # preproc = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    preproc = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # preproc = transforms.Compose([transforms.ToTensor()])

    # dataset = dataset.map_tuple(preproc, lambda x: x)
    dataset = dataset.map_tuple(preproc, lambda x: torch.tensor(x))
    # dataset = rgb_dataset.map_tuple(preproc, lambda x: x)
    # dataset = torch_dataset.map_tuple(preproc, lambda x: x)

    print("4 sample")
    for sample in dataset:
        print(type(sample))
        for val in sample:
            print(type(val), val.size())
        break
    print()

    # for image, data in islice(dataset, 0, 3):
    #     print(image.shape, image.dtype, type(data))

    batch_size = 20
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=4,
                                             batch_size=batch_size)
    # dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size),
    #                                          num_workers=4,
    #                                          batch_size=None)
    # print(len(dataloader))
    print("5 sample")
    for i, (images, targets) in enumerate(dataloader):
        # images, targets = next(iter(dataloader))
        print(images.shape, targets)
        if i > 2:
            break
    print()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=4,
                                             batch_size=batch_size,
                                             shuffle=True)
    print("6 sample")
    for i, (images, targets) in enumerate(dataloader):
        # images, targets = next(iter(dataloader))
        print(images.shape, targets)
        if i > 2:
            break
    print()

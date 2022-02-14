import argparse
import sys

import webdataset as wds
import torchvision

# from lib.datasets import BDD


def create_pickle_webdataset(dataset, name, is_test=False):
    sink = wds.TarWriter("{}.tar".format(name))
    for index, data in enumerate(dataset):
        if is_test:
            input, output = data
        else:
            input, output = data, 1
        if index % 1000 == 0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": input,
            "output.pyd": output,
        })
    sink.close()


def create_img_webdataset(dataset, subset, name, is_test=False):
    # length = len(dataset)
    num_tars = 4
    tar_files = [f"{name}-{subset}-{i:03d}.tar" for i in range(num_tars)]
    sinks = [wds.TarWriter(tar_file) for tar_file in tar_files]
    for index, data in enumerate(dataset):
        if is_test:
            input, output = data
        else:
            input, output = data, 1
        if index % 1000 == 0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        sinks[index % num_tars].write({
            "__key__": "%06d" % index,
            "jpg": input,
            "cls": output,
        })
    for sink in sinks:
        sink.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        type=str,
                        choices=["img", "pickle", "video"],
                        default="img")
    parser.add_argument("--path", type=str, default="./data")
    parser.add_argument("--name", type=str, default="mnist")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--datasetname", type=str, default="mnist")
    args = parser.parse_args()

    is_test = False
    is_train = args.subset == "train"
    if args.datasetname == "mnist":
        dataset = torchvision.datasets.MNIST(root=args.path,
                                             train=is_train,
                                             download=True)
        is_test = True
    # elif args.datasetname == "bdd":
    #     dataset = BDD(args.path)

    if args.type == "pickle":
        create_pickle_webdataset(dataset, args.name, is_test)
    elif args.type == "img":
        create_img_webdataset(dataset, args.subset, args.name, is_test)

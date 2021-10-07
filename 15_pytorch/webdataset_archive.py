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


def create_img_webdataset(dataset, name, is_test=False):
    sink = wds.TarWriter("{}.tar".format(name))
    for index, data in enumerate(dataset):
        if is_test:
            input, output = data
        else:
            input, output = data, 1
        if index % 1000 == 0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        sink.write({
            "__key__": "%06d" % index,
            "jpg": input,
            "cls": output,
        })
    sink.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        type=str,
                        choices=["img", "pickle", "video"],
                        default="img")
    parser.add_argument("--path", type=str, default="./data")
    parser.add_argument("--name", type=str, default="mnist")
    parser.add_argument("--datasetname", type=str, default="mnist")
    args = parser.parse_args()

    is_test = False
    if args.datasetname == "mnist":
        dataset = torchvision.datasets.MNIST(root=args.path, download=True)
        is_test = True
    # elif args.datasetname == "bdd":
    #     dataset = BDD(args.path)

    if args.type == "pickle":
        create_pickle_webdataset(dataset, args.name, is_test)
    elif args.type == "img":
        create_img_webdataset(dataset, args.name, is_test)

import argparse


def config():
    args = argparse.ArgumentParser(description="SDP_CEVI_TEJAS")

    args.add_argument(
        "--bs",
        type=int,
        default=1024,
        help="Pretraining Batch Size default = 256",
    )

    args.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learing Rate default = 1e-3",
    )

    args.add_argument(
        "--logint",
        type=int,
        default=10,
        help="Print log interval default = 10",
    )

    args.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="No of cluster centriods for kmeans default = 10",
    )

    args.add_argument(
        "--recentre",
        type=int,
        default=20,
        help="recentre updation instervel default = 2",
    )

    args.add_argument(
        "--preepochs",
        type=int,
        default=500,
        help="Pretrain epochs default = 500",
    )

    args.add_argument(
        "--A",
        type=float,
        default=0.5,
        help="alpha in reconstruction loss default = 0.5"
    )

    args.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch manuel seed default = 0 ",
    )

    args.add_argument(
        "--save",
        type=str,
        default="/home/cvgws-06/Desktop/Tejas/IDEC-pytorch-20210809T053841Z-001/IDEC-pytorch/",
        help="pretrain model dir ",

    )

    args.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Train dataset default = fmnist ",


    )

    args.add_argument(
        "--trail",
        type=str,
        default="initial",
        help="Trail ID",

    )

    return args.parse_args()

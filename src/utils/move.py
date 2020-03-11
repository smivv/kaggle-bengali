import os
import json
import shutil

import argparse
from argparse import ArgumentParser
from pathlib import Path


def build_args(parser: ArgumentParser):
    """
    Builds the command line parameters
    """
    parser.add_argument("logdir", type=Path, help="Path to model logdir")

    return parser


def parse_args():
    """
    Parses the command line arguments for the main method
    """
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def round(m):
    return int(round(m, 2) * 100)


def main(args, _):
    """
    Main method for `move`
    """
    logdir: Path = args.logdir

    expname = logdir.stem

    with open(str(logdir / "checkpoints" / "_metrics.json")) as f:
        data = json.load(f)

        recall = data["best"]["_total_recall"]
        epoch = None

        for k, v in data.items():
            if k == "best":
                continue
            if k.startswith("stage") and v["_total_recall"] == recall:
                epoch = int(k.split(".")[1])
                break
            if k.startswith("epoch") and v[1] == recall:
                epoch = int(k.split("_")[1])
                break

        shutil.copyfile(str(logdir / "checkpoints" / "best.pth"),
                        str(logdir / "trace" / f"{expname}_{epoch}_{recall}.pth"))

        shutil.copyfile(str(logdir / "configs" / "config.yml"),
                        str(logdir / "trace" / f"{expname}_{epoch}_{recall}.yml"))

        os.rename(str(logdir / "trace" / "traced-best-forward.pth"),
                  str(logdir / "trace" / f"{expname}_{epoch}_{recall}.jit"))


if __name__ == "__main__":
    main(parse_args(), None)

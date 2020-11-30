from typing import Optional, List, Dict, Any

import faiss
import pickle
import torch.distributed  # noqa: WPS301

import numpy as np
import pandas as pd

from time import time
from scipy.stats import mode
from pathlib import Path

from catalyst.dl import IRunner, CallbackOrder, Callback

from src.utils.knn import build_index, knn
from src.datasets.cico import INPUT_FILENAME_KEY, INPUT_TARGET_KEY,  \
    OUTPUT_EMBEDDINGS_KEY


class DoECallback(Callback):

    def __init__(
            self,
            target_key: str = INPUT_TARGET_KEY,
            filename_key: str = INPUT_FILENAME_KEY,
            embeddings_key: str = OUTPUT_EMBEDDINGS_KEY,
            class_names: List[str] = None,
            train_loaders: List[str] = None,
            test_loaders: List[str] = None,
            doe_path: str = None
    ):
        super().__init__(CallbackOrder.Metric)

        self.target_key = target_key
        self.filename_key = filename_key
        self.embeddings_key = embeddings_key

        self.class_names = {i: name for i, name in enumerate(class_names)}
        self.class_ids = {name: i for i, name in enumerate(class_names)}

        self.train_loaders = train_loaders
        self.test_loaders = test_loaders

        self.train_targets: List = []
        self.train_embeddings: List = []

        self.test_targets: List = []
        self.test_filenames: List = []
        self.test_embeddings: List = []

        self.dfs: Dict[str, pd.DataFrame] = pd.read_excel(doe_path, sheet_name=None)

    def _init(self):
        self.train_targets: List = []
        self.train_embeddings: List = []

        self.test_targets: List = []
        self.test_filenames: List = []
        self.test_embeddings: List = []

    def _detach(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def on_epoch_start(self, runner: "IRunner"):
        self._init()

    def on_loader_start(self, runner: "IRunner"):
        if runner.loader_name in self.test_loaders:
            self.test_targets: List = []
            self.test_filenames: List = []
            self.test_embeddings: List = []

    def on_batch_end(self, runner: "IRunner"):

        embs: np.ndarray = self._detach(runner.output[self.embeddings_key])
        targets: np.ndarray = self._detach(runner.input[self.target_key])
        filenames: List = runner.input[self.filename_key]

        if runner.loader_name in self.train_loaders:
            self.train_targets.extend(targets)
            self.train_embeddings.extend(embs)
        elif runner.loader_name in self.test_loaders:
            self.test_targets.extend(targets)
            self.test_filenames.extend(filenames)
            self.test_embeddings.extend(embs)

    def on_loader_end(self, runner: "IRunner"):

        if runner.loader_name not in self.test_loaders:
            return

        doe_name = runner.loader_name.replace("extra_", "")

        metrics = {
            "All": 0,
            "Recognized": 0,
            "NotRecognized": 0,
            "Top1": 0,
            "Top2": 0,
            "Top3": 0,
        }

        if "DoEv1" in runner.loader_name:

            self.train_embeddings = np.array(self.train_embeddings)
            self.test_embeddings = np.array(self.test_embeddings)
            self.train_targets = np.array(self.train_targets)
            self.test_targets = np.array(self.test_targets)

            index = build_index(
                embeddings=self.test_embeddings,
                labels=self.train_targets,
            )

            length = len(self.test_embeddings)

            for i, (embedding, target) in enumerate(zip(
                    self.test_embeddings, self.test_targets)):

                start = time()

                results: List[Dict] = knn(
                    embedding=embedding,
                    index=index,
                    top_k=3,
                    k=1,
                    labels2captions=self.class_names
                )

                targets = [r["target"] for r in results]

                try:
                    position = targets.index(target) + 1

                    metrics["Recognized"] += 1
                    metrics[f"Top{position}"] += 1
                except ValueError:
                    metrics["NotRecognized"] += 1

                metrics["All"] += 1

                end = time()

                print(f"Processed {i + 1}/{length} for {(end - start):.1f}s")

        elif "DoEv2" in runner.loader_name:

            self.test_filenames = \
                {f: i for i, f in enumerate(self.test_filenames)}

            self.train_embeddings = np.array(self.train_embeddings)
            self.test_embeddings = np.array(self.test_embeddings)
            self.train_targets = np.array(self.train_targets)

            index1 = build_index(
                embeddings=self.train_embeddings,
                labels=self.train_targets,
            )

            length = len(self.dfs)

            for i, (caption, df) in enumerate(self.dfs.items()):

                start = time()

                if caption not in self.class_ids:
                    t = len(self.class_names)
                    print(f"Add Classname {t}:{caption}")
                    self.class_names[t] = caption
                    self.class_ids[caption] = t

                target = self.class_ids[caption]

                def iname(index):
                    return f"{caption}({index})"

                for _, row in df.iterrows():
                    train_indcs = np.array([
                        self.test_filenames[iname(s)]
                        for s in str(row["Train Images"]).split(",")
                    ])
                    test_idx = self.test_filenames[iname(row["Test Images"])]

                    index2 = build_index(
                        embeddings=self.test_embeddings[train_indcs],
                        labels=np.array([target] * len(train_indcs)),
                    )

                    results: List[Dict] = sorted(knn(
                        index=index1,
                        embedding=self.test_embeddings[test_idx],
                        top_k=3,
                        k=1,
                        labels2captions=self.class_names
                    ) + knn(
                        index=index2,
                        embedding=self.test_embeddings[test_idx],
                        top_k=1,
                        k=1,
                        labels2captions=self.class_names
                    ), key=lambda x: x["distance"])[:3]

                    targets = [r["target"] for r in results]

                    try:
                        position = targets.index(target) + 1

                        metrics["Recognized"] += 1
                        metrics[f"Top{position}"] += 1
                    except ValueError:
                        metrics["NotRecognized"] += 1

                    metrics["All"] += 1

                end = time()

                print(f"Processed {i + 1}/{length} for {(end - start):.1f}s")

        result = {}
        for k, v in metrics.items():
            if k == "All":
                continue
            result[f"{doe_name}_{k}"] = metrics[k] / metrics['All']

        # runner.loader_metrics.update(**metrics)
        runner.epoch_metrics.update(**result)

from typing import Optional, List, Dict, Any

import faiss
import pickle
import torch.distributed  # noqa: WPS301

import numpy as np
import pandas as pd

from scipy.stats import mode
from pathlib import Path

from catalyst.dl import IRunner, CallbackOrder, Callback

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

    def _knn(
            self,
            all_embeddings: np.ndarray,
            all_targets: np.ndarray,
            embedding: np.ndarray,
            index: faiss.Index = None,
            top_k: int = 3,
            k: int = 5,
    ) -> List[Dict[str, Any]]:
        """

        Args:
            embedding (np.ndarray): Embeddings to look kNN for
            top_k (int): Top K results to return
            k (int): Number of closest neighbors (k > 0)
            index (bool): Whether to use index or not

        Returns (Dict[str, Any]): Results

        """
        assert 0 < k < len(self.class_names), \
            "`k` should be positive number"

        if index is not None:
            # Search for closest embeddings in terms of inner product distance
            dists, nn_indcs = index.search(
                embedding[np.newaxis, ...], k=len(all_embeddings))
            dists = np.arccos(dists)

            dists = np.squeeze(dists)
            nn_indcs = np.squeeze(nn_indcs)
        else:
            # Search for closest embeddings in terms of inner product distance
            dists = np.dot(all_embeddings, embedding.T)
            dists = np.arccos(dists)

            # Sort resulting distances
            nn_indcs = dists.argsort(axis=0)
            dists = dists[nn_indcs]

        embeddings = all_embeddings[nn_indcs]
        nn_targets = all_targets[nn_indcs]

        results = []
        true_target = None
        for _ in range(top_k):

            if true_target is not None:
                not_equal_indcs = np.where(nn_targets != true_target)[0]
                nn_targets = nn_targets[not_equal_indcs]
                dists = dists[not_equal_indcs]
                embeddings = embeddings[not_equal_indcs]

            # Tale first k neighbor classes
            k_nn_targets = nn_targets[:k]

            # Find most frequent from them
            true_target = mode(k_nn_targets, axis=0)[0][0]

            closest_index = nn_targets.tolist().index(true_target)
            closest_distance = dists[closest_index]

            result = {
                "target": true_target,
                "caption": self.class_names[true_target],
                "distance": closest_distance,
                # "embeddings": embeddings[closest_index],
            }

            results.append(result)

        return results

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

        if "DoEv1" not in runner.loader_name or \
                "DoEv2" not in runner.loader_name:
            return

        doe_name = runner.loader_name.replace("infer_", "")

        metrics = {
            "All": 0,
            "Recognized": 0,
            "NotRecognized": 0,
            "Top1": 0,
            "Top2": 0,
            "Top3": 0,
        }

        if "DoEv1" in runner.loader_name:

            train_embeddings = np.array(self.train_embeddings)
            test_embeddings = np.array(self.test_embeddings)
            train_targets = np.array(self.train_targets)
            test_targets = np.array(self.test_targets)

            for embedding, target in zip(test_embeddings, test_targets):
                results: List[Dict] = self._knn(
                    all_embeddings=train_embeddings,
                    all_targets=train_targets,
                    embedding=embedding,
                    top_k=3,
                    k=1
                )

                captions = [r["caption"] for r in results]

                try:
                    position = captions.index(target) + 1

                    metrics["Recognized"] += 1
                    metrics[f"Top{position}"] += 1
                except ValueError:
                    metrics["NotRecognized"] += 1

                metrics["All"] += 1

        elif "DoEv2" in runner.loader_name:

            test_filenames = {f: i for i, f in enumerate(self.test_filenames)}

            for caption, df in self.dfs.items():

                if caption not in self.class_names:
                    self.class_names[len(self.class_names)] = caption

                def iname(index):
                    return f"{caption}({index}).jpg"

                for i, row in df.iterrows():
                    train_indcs = [
                        test_filenames[iname(s)]
                        for s in str(row["Train Images"]).split(",")
                    ]
                    test_idx = test_filenames[iname(row["Test Images"])]

                    results: List[Dict] = self._knn(
                        all_embeddings=np.array(
                            self.train_embeddings +
                            [self.test_embeddings[idx] for idx in train_indcs]
                        ),
                        all_targets=np.array(
                            self.train_targets + [caption]
                        ),
                        embedding=self.test_embeddings[test_idx],
                        top_k=3,
                        k=1
                    )

                    captions = [r["caption"] for r in results]

                    try:
                        position = captions.index(caption) + 1

                        metrics["Recognized"] += 1
                        metrics[f"Top{position}"] += 1
                    except ValueError:
                        metrics["NotRecognized"] += 1

                    metrics["All"] += 1

        for k, v in metrics.items():
            if k == "All":
                continue
            metrics[f"{doe_name}_{k}"] = metrics[k] / metrics['All']
            del metrics[k]

        del metrics["All"]

        runner.loader_metrics.update(**metrics)
        # runner.epoch_metrics.update(**metrics)
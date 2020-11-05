from typing import Optional, List, Dict

import math
import pickle
import torch.distributed  # noqa: WPS301

import numpy as np
import pandas as pd

from scipy.stats import mode
from pathlib import Path

from catalyst.dl import IRunner, CallbackOrder, Callback
from catalyst.utils.torch import get_activation_fn


FILENAMES_KEY = "filenames"
EMBEDDINGS_KEY = "embeddings"
PREDICTS_KEY = "predicts"
TARGETS_KEY = "targets"
LABELS_KEY = "labels"

KEYS = [FILENAMES_KEY, EMBEDDINGS_KEY, PREDICTS_KEY, TARGETS_KEY, LABELS_KEY]

TOP1_KEY = "top1"
TOP3_KEY = "top3"

TOP1_KNN_KEY = "top1_knn"
TOP3_KNN_KEY = "top3_knn"

XLS_KEYS = [FILENAMES_KEY, LABELS_KEY, TOP1_KEY, TOP3_KEY, TOP1_KNN_KEY, TOP3_KNN_KEY]


class EmbeddingsLoggerCallback(Callback):

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            label_key: str = "labels",
            filenames_key: str = "filenames",
            embeddings_key: str = "embeddings",
            class_names: List[str] = None,
            database_path: Path = None,
            save_dir: str = None,
            activation: Optional[str] = None,
    ):
        super().__init__(CallbackOrder.Metric)

        self.filenames_key = filenames_key
        self.embeddings_key = embeddings_key

        self.labels_key = label_key
        self.input_key = input_key
        self.output_key = output_key

        self.save_dir = save_dir

        self.class_names = np.asarray(class_names)

        self.activation_fn = get_activation_fn(activation)

        self.n_folds = 1

        self.database_path = Path(database_path)
        if self.database_path is not None:
            self.train_emb = self._read_pickle(self.database_path)

    def _init(self):
        self.data = {}

    def _detach(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _read_pickle(self, pickle_dir: Path, name='all'):
        pickle_path = pickle_dir / (name + '.pickle')
        return pickle.loads(open(pickle_path, "rb").read())

    def _save_pickle(self, embeddings, pickle_dir: Path, name: str):
        with open((pickle_dir / (name + ".pickle")).as_posix(), "wb") as file:
            pickle.dump(obj=embeddings, file=file)

    def _save_excel(self, embeddings, csv_dir: Path, name: str):
        df = pd.DataFrame.from_dict(embeddings)
        df.to_excel((csv_dir / (name + ".xlsx")).as_posix())

    def _knn(self, train_emb, test_emb=None, k=5, topk=3,
             metric="angular", method="most frequent"):

        leave_one_out = test_emb is None

        if leave_one_out:
            test_emb = train_emb

        test_length = len(test_emb[self.labels_key])
        x_train, y_train = np.asarray(train_emb[self.embeddings_key]), \
                           np.asarray(train_emb[self.labels_key])
        x_test, y_test = np.asarray(test_emb[self.embeddings_key]), \
                         np.asarray(test_emb[self.labels_key])

        y_pred = None
        result = False
        while not result:
            try:
                y_pred = []
                end_idx, batch_size = 0, math.ceil(test_length / self.n_folds)
                for s, start_idx in enumerate(
                        range(0, test_length, batch_size)):

                    print(f"Nearest Neighbors evaluation for {s + 1}th of "
                          f"{self.n_folds} folds started..")

                    end_idx = min(start_idx + batch_size, test_length)

                    x = x_test[start_idx: end_idx]

                    if metric.startswith("angular"):
                        dot = np.dot(x, x_train.T)
                        if metric.endswith("norm"):
                            n1 = np.linalg.norm(x, axis=1)[:, np.newaxis]
                            n2 = np.linalg.norm(x_train.T, axis=0)[np.newaxis,
                                 :]
                            product = np.arccos(dot / n1 / n2)
                        else:
                            product = np.arccos(dot)
                    elif metric == "euclidean":
                        product = np.linalg.norm(x[:, np.newaxis] - x_train,
                                                 axis=2)
                    else:
                        raise Exception("Unknown metric")

                    knn_indcs = product.argsort(axis=1)
                    knn_classes = y_train[knn_indcs]

                    if leave_one_out:
                        # first elements are always the same
                        knn_classes = knn_classes[:, 1:]

                    length = len(knn_classes)
                    results = [[] for _ in range(length)]

                    if method == "first":
                        for i in range(length):
                            for class_ in knn_classes[i]:
                                if class_ not in results[i]:
                                    results[i].append(class_)
                                if len(results[i]) == topk:
                                    break
                    elif method == "most frequent":
                        for i in range(length):
                            rm_class = None
                            classes = knn_classes[i].copy()
                            for j in range(topk):
                                if rm_class is not None:
                                    classes = classes[classes != rm_class]
                                k_classes = classes[:k]
                                result, _ = mode(k_classes, axis=0)
                                rm_class = result[0]
                                results[i].append(rm_class)
                    else:
                        raise Exception("Unknown metric")

                    y_pred.extend(results)

                result = True
            except MemoryError:
                result = False
                self.n_folds *= 2
                print(f"Memory error with {int(self.n_folds / 2)} fold, "
                      f"will try with {self.n_folds}..")

        return np.asarray(y_pred), y_test

    def on_stage_start(self, runner: "IRunner"):
        self._init()

    def on_loader_start(self, runner: "IRunner"):
        self.data[runner.loader_name] = {
            FILENAMES_KEY: [],
            EMBEDDINGS_KEY: [],
            PREDICTS_KEY: [],
            TARGETS_KEY: [],
            LABELS_KEY: [],
            TOP1_KEY: [],
            TOP3_KEY: [],
            TOP1_KNN_KEY: [],
            TOP3_KNN_KEY: [],
        }

    def on_batch_end(self, runner: "IRunner"):

        filenames = runner.input[self.filenames_key]
        labels = runner.input[self.labels_key]
        embeddings = self._detach(runner.output[self.embeddings_key])

        targets: np.ndarray = self._detach(runner.input[self.input_key])

        predicts = self.activation_fn(runner.output[self.output_key])
        predicts = self._detach(predicts)

        assert not np.isnan(np.sum(targets)) and \
               not np.isnan(np.sum(predicts))

        knn_top3, _ = self._knn(
            train_emb=self.train_emb,
            test_emb={
                EMBEDDINGS_KEY: embeddings,
                LABELS_KEY: targets,
            },
            metric="angular",
            method="most frequent",
            topk=3,
            k=5,
        )
        knn_top1 = knn_top3[:, 0]

        indcs = np.argsort(-predicts, axis=1)

        top1 = self.class_names[indcs[:, 0]]
        top3 = self.class_names[indcs[:, :3]]

        data = self.data[runner.loader_name]

        data[FILENAMES_KEY].extend(filenames)
        data[EMBEDDINGS_KEY].extend(embeddings)
        data[PREDICTS_KEY].extend(predicts)
        data[TARGETS_KEY].extend(targets)
        data[LABELS_KEY].extend(labels)

        data[TOP1_KEY].extend(top1)
        data[TOP3_KEY].extend(top3)

        data[TOP1_KNN_KEY].extend(knn_top1)
        data[TOP3_KNN_KEY].extend(knn_top3)

    def on_stage_end(self, runner: "IRunner"):
        save_to: Path = Path(runner.logdir)
        save_to /= "embeddings"

        if self.save_dir is not None:
            save_to /= self.save_dir

        save_to.mkdir(parents=True, exist_ok=True)

        data = {k: [] for k in XLS_KEYS}
        for loader_name, loader_values in self.data.items():
            self._save_pickle(loader_values, save_to, loader_name)

            for key in XLS_KEYS:
                data[key] += loader_values[key]

        self._save_excel(data, save_to, "data")

        # labels = self.data[runner.loader_name][LABELS_KEY]
        # result = {k: 0 for k in range(15)}
        # for l in labels:
        #     result[l] += 1
        # print(result)

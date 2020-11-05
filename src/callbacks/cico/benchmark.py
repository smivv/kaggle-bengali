from typing import Optional, List, Dict, Union

import math
import pickle
import torch.distributed  # noqa: WPS301

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy.stats import mode
from pathlib import Path

from catalyst.dl import IRunner, State, CallbackOrder, Callback
from catalyst.utils.torch import get_activation_fn

from src.utils.plots import plot_confusion_matrix
from src.utils.utils import build_index

FILENAMES_KEY = "filenames"
EMBEDDINGS_KEY = "embeddings"
PREDICTS_KEY = "predicts"
TARGETS_KEY = "targets"
LABELS_KEY = "labels"

TOP1_KEY = "top1"
TOP3_KEY = "top3"

TOP1_EQ_KEY = "top1_eq"
TOP3_EQ_KEY = "top3_eq"

TOP1_KNN_KEY = "top1_knn"
TOP3_KNN_KEY = "top3_knn"

TOP1_KNN_EQ_KEY = "top1_knn_eq"
TOP3_KNN_EQ_KEY = "top3_knn_eq"

KEYS = [FILENAMES_KEY, EMBEDDINGS_KEY, PREDICTS_KEY, TARGETS_KEY, LABELS_KEY]

PICKLE_KEYS = [
    FILENAMES_KEY, EMBEDDINGS_KEY, TARGETS_KEY, LABELS_KEY
]

XLS_KEYS = [
    FILENAMES_KEY, LABELS_KEY,
    TOP1_KEY, TOP1_EQ_KEY, TOP3_KEY, TOP3_EQ_KEY,
    TOP1_KNN_KEY, TOP1_KNN_EQ_KEY, TOP3_KNN_KEY, TOP3_KNN_EQ_KEY
]

ALL_KEYS = [
    FILENAMES_KEY, EMBEDDINGS_KEY, PREDICTS_KEY, TARGETS_KEY, LABELS_KEY,
    TOP1_KEY, TOP1_EQ_KEY, TOP3_KEY, TOP3_EQ_KEY,
    TOP1_KNN_KEY, TOP1_KNN_EQ_KEY, TOP3_KNN_KEY, TOP3_KNN_EQ_KEY
]


class BenchmarkingCallback(Callback):

    def __init__(
            self,
            input_key: str = "target",
            output_key: str = "logits",
            label_key: str = "label",
            filenames_key: str = "filename",
            embeddings_key: str = "embeddings",
            test_cases: Dict[str, Union[str, List[str]]] = None,
            class_names: List[str] = None,
            save_dir: str = None,
            activation: Optional[str] = None,
            benchmarking_plan_path: str = None,
    ):
        super().__init__(CallbackOrder.Metric)

        self.index = build_index(benchmarking_plan_path)

        self.input_key = input_key
        self.output_key = output_key
        self.labels_key = label_key
        self.filenames_key = filenames_key
        self.embeddings_key = embeddings_key

        self.test_cases = test_cases

        self.save_dir = save_dir

        self.class_names = np.asarray(class_names)

        self.activation_fn = get_activation_fn(activation)

        self.n_folds = 8

        self.best_score = -1
        
        self.torch_metrics = {
            "angular": self._knn_angular_torch,
            "angular_normed": self._knn_angular_normed_torch,
            "euclidean": self._knn_euclidean_torch,
        }
        
        self.numpy_metrics = {
            "angular": self._knn_angular_numpy,
            "angular_normed": self._knn_angular_normed_numpy,
            "euclidean": self._knn_euclidean_numpy,
        }

    def _init(self):
        self.data = {}

    def _detach(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _read_pickle(self, pickle_dir: Path, name='all'):
        pickle_path = pickle_dir / (name + '.pickle')
        return pickle.loads(open(pickle_path, "rb").read())

    def _save_pickle(self, data, pickle_dir: Path, name: str):
        data_ = {}
        for key in PICKLE_KEYS:
            data_ = data[key]
        with open((pickle_dir / (name + ".pickle")).as_posix(), "wb") as file:
            pickle.dump(obj=data_, file=file)

    def _save_excel(self, data, csv_dir: Path, name: str):
        data_ = {}
        for key in XLS_KEYS:
            data_ = data[key]
        df = pd.DataFrame.from_dict(data_)
        df.to_excel((csv_dir / (name + ".xlsx")).as_posix())

    def _knn_angular_torch(self, x, y):
        dot = torch.mm(x, y.transpose(0, 1))
        return self._detach((-dot).argsort(dim=1))

    def _knn_angular_normed_torch(self, x, y):
        dot = torch.mm(x, y.transpose(0, 1))
        n1 = torch.unsqueeze(torch.norm(x, dim=1), dim=1)
        n2 = torch.unsqueeze(torch.norm(torch.transpose(y, 0, 1), dim=0), dim=0)
        product = torch.acos(dot / n1 / n2)
        return self._detach(product.argsort(dim=1))

    def _knn_euclidean_torch(self, x, y):
        product = torch.norm(torch.unsqueeze(x, dim=1) - y, dim=2)
        return self._detach(product.argsort(dim=1))

    def _knn_angular_numpy(self, x, y):
        return (-np.dot(x, y.T)).argsort(axis=1)

    def _knn_angular_normed_numpy(self, x, y):
        dot = np.dot(x, y.T)
        n1 = np.linalg.norm(x, axis=1)[:, np.newaxis]
        n2 = np.linalg.norm(y.T, axis=0)[np.newaxis, :]
        product = np.arccos(dot / n1 / n2)
        return product.argsort(dim=1)

    def _knn_euclidean_numpy(self, x, y):
        product = np.linalg.norm(x[:, np.newaxis] - y, axis=2)
        return product.argsort(axis=1)
    
    def _knn(self, train_emb, test_emb=None, k=5, topk=3,
             metric="angular", method="most frequent", use_torch=False):

        leave_one_out = test_emb is None

        if leave_one_out:
            test_emb = train_emb

        test_length = len(test_emb[TARGETS_KEY])
        x_train, y_train = np.asarray(train_emb[EMBEDDINGS_KEY]), np.asarray(
            train_emb[TARGETS_KEY])
        x_test, y_test = np.asarray(test_emb[EMBEDDINGS_KEY]), np.asarray(
            test_emb[TARGETS_KEY])

        if use_torch:
            x_train = torch.FloatTensor(x_train)
            y_train = torch.FloatTensor(y_train)
            x_test = torch.FloatTensor(x_test)
            y_test = torch.FloatTensor(y_test)

            metric_fn = self.torch_metrics[metric]
        else:
            metric_fn = self.numpy_metrics[metric]

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

                    knn_indcs = metric_fn(x, x_train)
                    knn_classes = y_train[knn_indcs]

                    if leave_one_out:
                        # first elements are always the same
                        knn_classes = knn_classes[:, 1:]

                    # reduce size of array to speed up process
                    knn_classes = knn_classes[:, :1000]

                    uniques, indices = \
                        np.unique(knn_classes, return_index=True, axis=1)

                    indices = np.argsort(indices, axis=1)
                    uniques = uniques[indices]

                    if method == "first":
                        results = uniques[:, topk].tolist()
                    elif method == "most frequent":
                        results = []
                        rm_class = None
                        for j in range(topk):
                            if rm_class is not None:
                                uniques = uniques[uniques != rm_class]
                            k_uniques = uniques[:, :k]
                            result, _ = mode(k_uniques, axis=1)
                            rm_class = result[:, 0]
                            results.append(rm_class)
                        results = np.hstack(results)
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

    def _knn_torch(self, train_emb, test_emb=None, k=5, topk=3,
                   metric="angular", method="most frequent"):

        leave_one_out = test_emb is None

        if leave_one_out:
            test_emb = train_emb

        test_length = len(test_emb[TARGETS_KEY])
        x_train, y_train = np.asarray(train_emb[EMBEDDINGS_KEY]), np.asarray(
            train_emb[TARGETS_KEY])
        x_test, y_test = np.asarray(test_emb[EMBEDDINGS_KEY]), np.asarray(
            test_emb[TARGETS_KEY])

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
                        # dot = np.dot(x, x_train.T)
                        dot = torch.mm(x, x_train.transpose(0, 1))
                        if metric.endswith("norm"):
                            raise NotImplemented("")
                            # n1 = np.linalg.norm(x, axis=1)[:, np.newaxis]
                            # n2 = np.linalg.norm(x_train.T, axis=0)[np.newaxis, :]
                            # product = np.arccos(dot / n1 / n2)
                        else:
                            # product = np.arccos(dot)
                            product = torch.acos(dot)
                    elif metric == "euclidean":
                        product = np.linalg.norm(x[:, np.newaxis] - x_train,
                                                 axis=2)
                    else:
                        raise Exception("Unknown metric")

                    # knn_indcs = product.argsort(axis=1)
                    knn_indcs = self._detach(product.argsort(dim=1))
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

    def _benchmark(self, filenames, true, predicted):
        result = {
            k: {'true': [], 'pred': []}
            for k in ['size', 'dishware', 'trial', 'scale', 'recipe kind']
        }

        for filename, true, pred in zip(filenames, true, predicted):

            # filename = Path(filename).name

            if "Camera" not in filename:
                continue

            if '5000k' in filename:
                cat, subcat, var, _, _, camera, _ = filename.split('_')
            else:
                cat, subcat, var, _, camera, _ = filename.split('_')

            test_case = self.index[f"{cat}_{subcat}_{var}_{camera}"]

            for k, v in result.items():
                if k in test_case:
                    result[k]['true'].append(true)
                    result[k]['pred'].append(pred)
        return result

    def _save_benchmark(self, result, save_to: Path, prefix: str = "top1"):
        out = {}
        for k, v in result.items():
            plot_confusion_matrix(
                y_true=v['true'],
                y_pred=v['pred'],
                classes=self.class_names,
                save_to=(save_to / f'_{prefix}_{k}_cm_norm.png').as_posix(),
                normalize=True
            )

            out[k] = accuracy_score(v['true'], v['pred'])

            open((save_to / f"_{prefix}_{k}_{out[k]}").as_posix(), 'w').close()
        return out

    def on_stage_start(self, runner: "IRunner"):
        self._init()

    def on_loader_start(self, runner: "IRunner"):
        self.data[runner.loader_name] = {k: [] for k in ALL_KEYS}

    def on_batch_end(self, runner: "IRunner"):

        filenames = runner.input[self.filenames_key]
        labels = runner.input[self.labels_key]
        embeddings = self._detach(runner.output[self.embeddings_key])

        targets: np.ndarray = self._detach(runner.input[self.input_key])

        predicts = self.activation_fn(runner.output[self.output_key])
        predicts = self._detach(predicts)

        assert not np.isnan(np.sum(targets)) and \
               not np.isnan(np.sum(predicts))

        indcs = np.argsort(-predicts, axis=1)

        top1_indcs = indcs[:, 0]
        top3_indcs = indcs[:, :3]

        top1_labels = self.class_names[top1_indcs]
        top3_labels = self.class_names[top3_indcs]

        data = self.data[runner.loader_name]

        data[FILENAMES_KEY].extend(filenames)
        data[EMBEDDINGS_KEY].extend(embeddings)
        data[TARGETS_KEY].extend(targets)
        data[LABELS_KEY].extend(labels)

        data[TOP1_KEY].extend(top1_labels)
        data[TOP3_KEY].extend(top3_labels)

        top1_eq = np.equal(top1_indcs, targets)
        top3_eq = np.max(np.equal(top3_indcs, targets[..., np.newaxis]), axis=1)

        data[TOP1_EQ_KEY].extend(top1_eq)
        data[TOP3_EQ_KEY].extend(top3_eq)

    def on_epoch_end(self, runner: "IRunner"):

        save_to: Path = Path(runner.logdir)
        save_to /= "benchmarking"

        if self.save_dir is not None:
            save_to /= self.save_dir

        save_to.mkdir(parents=True, exist_ok=True)

        is_better = False
        for train_loader_name, test_loader_names in self.test_cases.items():

            if isinstance(test_loader_names, str):
                train_emb = self.data[test_loader_names]
            else:
                train_emb = {key: [] for key in KEYS}
                for key in KEYS:
                    for loader_name in test_loader_names:
                        train_emb[key] = train_emb[key] + \
                                         self.data[loader_name][key]

            data = self.data[train_loader_name]

            top3_indcs, _ = self._knn(
                train_emb=train_emb,
                test_emb=data,
                metric="angular",
                method="most frequent",
                topk=3,
                k=5,
            )
            top1_indcs = top3_indcs[:, 0]

            data[TOP1_KNN_KEY].extend(top1_indcs)
            data[TOP3_KNN_KEY].extend(top3_indcs)

            filenames = data[FILENAMES_KEY]
            targets = data[TARGETS_KEY]

            top1_eq = np.equal(top1_indcs, targets)
            top3_eq = np.max(np.equal(top3_indcs, targets[..., np.newaxis]), axis=1)

            data[TOP1_KNN_EQ_KEY].extend(top1_eq)
            data[TOP3_KNN_EQ_KEY].extend(top3_eq)

            if train_loader_name == "valid":
                top3_indcs = np.where(top3_eq, np.array(targets), top3_indcs[:, 0])
                top1_indcs, top3_indcs = top1_indcs.tolist(), top3_indcs.tolist()

                top1_score = accuracy_score(targets, top1_indcs)
                top3_score = accuracy_score(targets, top3_indcs)

                runner.epoch_metrics["_bench_top1_acc"] = top1_score
                runner.epoch_metrics["_bench_top3_acc"] = top3_score

                is_better = top1_score > self.best_score

                if is_better:
                    self.best_score = top1_score

                    for prefix, values in {"top1": top1_indcs, "top3": top3_indcs}:
                        plot_confusion_matrix(
                            y_true=targets,
                            y_pred=values,
                            classes=self.class_names,
                            normalize=True,
                            save_to=(save_to / f"_bench_{prefix}_main_cm").as_posix()
                        )

                        benchmark = self._benchmark(filenames, targets, values)
                        result = self._save_benchmark(benchmark, save_to, prefix)
                        for k, v in result:
                            runner.epoch_metrics[f"_bench_{prefix}_{k}_acc"] = v

            if is_better:
                data = {k: [] for k in XLS_KEYS}
                for loader_name, loader_values in self.data.items():
                    if loader_name in ["train", "valid"]:
                        self._save_pickle(loader_values, save_to, loader_name)

                    for key in XLS_KEYS:
                        data[key] += loader_values[key]

                self._save_excel(data, save_to, "data")

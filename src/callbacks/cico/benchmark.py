from typing import Union, Optional, List, Dict

import faiss
import pickle
import torch.distributed  # noqa: WPS301

import numpy as np
import pandas as pd

from time import time
from pathlib import Path

from catalyst.dl import IRunner, CallbackOrder, Callback
from catalyst.utils.torch import get_activation_fn

from sklearn.metrics import accuracy_score

from src.utils.knn import build_benchmark_index, build_index, knn
from src.datasets.cico import \
    INPUT_FILENAME_KEY, INPUT_TARGET_KEY, INPUT_LABEL_KEY, \
    OUTPUT_TARGET_KEY, OUTPUT_EMBEDDINGS_KEY

ALL_KEYS = [
    INPUT_FILENAME_KEY, INPUT_TARGET_KEY, INPUT_LABEL_KEY,
    OUTPUT_TARGET_KEY, OUTPUT_EMBEDDINGS_KEY,
]

XLS_KEYS = [
    INPUT_FILENAME_KEY, INPUT_TARGET_KEY, INPUT_LABEL_KEY,
]

PKL_KEYS = [
    INPUT_FILENAME_KEY, INPUT_TARGET_KEY, INPUT_LABEL_KEY,
    OUTPUT_EMBEDDINGS_KEY
]


class BenchmarkingCallback(Callback):

    def __init__(
            self,
            target_key: str,
            label_key: str,
            filename_key: str,
            embedding_key: str,
            logit_key: str = None,

            class_names: List[str] = None,
            activation: Optional[str] = None,
            
            enable_benchmark: bool = False,
            benchmark_train_loader: str = None,
            benchmark_test_loader: str = None,
            benchmark_xlsx: Union[str, Path] = None,
            
            enable_doev1: bool = False,
            doev1_train_loaders: Union[str, List[str]] = None,
            doev1_test_loaders: Union[str, List[str]] = None,

            enable_doev2: bool = False,
            doev2_train_loaders: Union[str, List[str]] = None,
            doev2_test_loaders: Union[str, List[str]] = None,
            doev2_xlsx: str = None,

            save_dir: str = None,
            save_loaders: Union[str, List[str]] = None,
    ):
        super().__init__(CallbackOrder.Metric)

        self.filename_key = filename_key
        self.embedding_key = embedding_key

        self.labels_key = label_key
        self.target_key = target_key
        self.logit_key = logit_key

        self.save_dir = save_dir

        self.class_names = {i: name for i, name in enumerate(class_names)}
        self.class_ids = {name: i for i, name in enumerate(class_names)}

        self.activation_fn = get_activation_fn(activation)
        
        self.enable_benchmark = enable_benchmark
        if enable_benchmark:
            self.benchmark_train_loader = benchmark_train_loader
            self.benchmark_test_loader = benchmark_test_loader
            self.benchmark_index: Dict = build_benchmark_index(benchmark_xlsx)
        
        self.enable_doev1 = enable_doev1
        if enable_doev1:
            self.doev1_train_loaders = doev1_train_loaders
            self.doev1_test_loaders = doev1_test_loaders

        self.enable_doev2 = enable_doev2
        if enable_doev2:
            self.doev2_train_loaders = doev2_train_loaders
            self.doev2_test_loaders = doev2_test_loaders
            self.doev2_dfs: Dict[str, pd.DataFrame] = \
                pd.read_excel(doev2_xlsx, sheet_name=None)

        self.save_loaders = save_loaders

    def _init(self):
        self.data = {}

    def _detach(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _read_pickle(self, pickle_dir: Path, name="all"):
        pickle_path = pickle_dir / (name + ".pkl")
        return pickle.loads(open(pickle_path, "rb").read())

    def _save_pickle(self, embeddings, pickle_dir: Path, name: str):
        with open((pickle_dir / (name + ".pkl")).as_posix(), "wb") as file:
            pickle.dump(obj=embeddings, file=file)

    def _save_excel(self, embeddings, csv_dir: Path, name: str):
        df = pd.DataFrame.from_dict(embeddings)
        df.to_excel(str(csv_dir / (name + ".xlsx")))
    
    def _gather_data(self, loaders: Union[str, List[str]]):
        if isinstance(loaders, str):
            return {k: np.array(v) for k, v in self.data[loaders].items()}
        else:
            data = {k: [] for k in ALL_KEYS}
            for k in ALL_KEYS:
                for loader in loaders:
                    data[k] += self.data[loader][k]
                data[k] = np.array(data[k])
            return data

    def doev1(self, runner: "IRunner"):

        metrics = {
            "target": [],
            "top1": [],
            "top3": [],
        }

        train_data = self._gather_data(self.doev1_train_loaders)
        test_data = self._gather_data(self.doev1_test_loaders)

        index = build_index(
            embeddings=train_data[OUTPUT_EMBEDDINGS_KEY],
            labels=train_data[INPUT_TARGET_KEY],
        )

        for i, (embedding, target) in enumerate(zip(
                test_data[OUTPUT_EMBEDDINGS_KEY],
                test_data[INPUT_TARGET_KEY])):

            results: List[Dict] = knn(
                embedding=embedding,
                index=index,
                top_k=3,
                k=1,
                labels2captions=self.class_names
            )

            targets = [r["label"] for r in results]

            top1 = targets[0]
            try:
                top3 = targets[targets.index(target)]
            except ValueError:
                top3 = top1

            metrics["target"].append(target)
            metrics["top1"].append(top1)
            metrics["top3"].append(top3)

        runner.epoch_metrics.update(**{
            "doev1_top1_acc": accuracy_score(metrics["top1"], metrics["target"]),
            "doev1_top3_acc": accuracy_score(metrics["top3"], metrics["target"]),
        })

    def doev2(self, runner: "IRunner"):

        metrics = {
            "target": [],
            "top1": [],
            "top3": [],
        }

        train_data = self._gather_data(self.doev2_train_loaders)
        test_data = self._gather_data(self.doev2_test_loaders)

        test_filenames = \
            {f: i for i, f in enumerate(test_data[INPUT_FILENAME_KEY])}
        test_embeddings = test_data[OUTPUT_EMBEDDINGS_KEY]

        index1 = build_index(
            embeddings=train_data[OUTPUT_EMBEDDINGS_KEY],
            labels=train_data[INPUT_TARGET_KEY],
        )

        for i, (caption, df) in enumerate(self.doev2_dfs.items()):

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
                    test_filenames[iname(s)]
                    for s in str(row["Train Images"]).split(",")
                ])
                test_idx = test_filenames[iname(row["Test Images"])]

                index2 = build_index(
                    embeddings=test_embeddings[train_indcs],
                    labels=np.array([target] * len(train_indcs)),
                )

                results: List[Dict] = sorted(knn(
                    index=index1,
                    embedding=test_embeddings[test_idx],
                    top_k=3,
                    k=1,
                    labels2captions=self.class_names
                ) + knn(
                    index=index2,
                    embedding=test_embeddings[test_idx],
                    top_k=1,
                    k=1,
                    labels2captions=self.class_names
                ), key=lambda x: x["distance"])[:3]

                targets = [r["label"] for r in results]

                top1 = targets[0]
                try:
                    top3 = targets[targets.index(target)]
                except ValueError:
                    top3 = top1

                metrics["target"].append(target)
                metrics["top1"].append(top1)
                metrics["top3"].append(top3)

        runner.epoch_metrics.update(**{
            "doev2_top1_acc": accuracy_score(metrics["top1"], metrics["target"]),
            "doev2_top3_acc": accuracy_score(metrics["top3"], metrics["target"]),
        })
        
    def benchmark(self, runner: "IRunner"):

        metrics = {
            k: {"target": [], "top1": [], "top3": []}
            for k in ["size", "dishware", "trial", "scale", "recipe kind"]
        }

        train_data = self._gather_data(self.benchmark_train_loader)
        test_data = self._gather_data(self.benchmark_test_loader)

        index = build_index(
            embeddings=train_data[OUTPUT_EMBEDDINGS_KEY],
            labels=train_data[INPUT_TARGET_KEY],
        )

        for i, (filename, embedding, target) in enumerate(zip(
                test_data[INPUT_FILENAME_KEY],
                test_data[OUTPUT_EMBEDDINGS_KEY],
                test_data[INPUT_TARGET_KEY])):

            if "Camera" not in filename:
                continue

            if "5000k" in filename:
                cat, subcat, var, _, _, camera, _ = filename.split("_")
            else:
                cat, subcat, var, _, camera, _ = filename.split("_")

            test_case = self.benchmark_index[f"{cat}_{subcat}_{var}_{camera}"]

            results: List[Dict] = knn(
                embedding=embedding,
                index=index,
                top_k=3,
                k=1,
                labels2captions=self.class_names
            )

            targets = [r["label"] for r in results]

            top1 = targets[0]
            try:
                top3 = targets[targets.index(target)]
            except ValueError:
                top3 = top1

            for k, v in metrics.items():
                if k in test_case:
                    metrics[k]["target"].append(target)
                    metrics[k]["top1"].append(top1)
                    metrics[k]["top3"].append(top3)

        all_top1, all_top3 = [], []
        for k, v in metrics.items():
            top1 = accuracy_score(metrics[k]["target"], metrics[k]["top1"])
            top3 = accuracy_score(metrics[k]["target"], metrics[k]["top3"])

            all_top1.append(top1)
            all_top3.append(top3)

            runner.epoch_metrics.update(**{
                f"{k}_top1_acc": top1,
                f"{k}_top3_acc": top3,
            })

        runner.epoch_metrics.update(**{
            "valid__avg_top1_acc": sum(all_top1) / len(all_top1),
            "valid__avg_top3_acc": sum(all_top3) / len(all_top3),
        })

    def on_stage_start(self, runner: "IRunner"):
        self._init()

    def on_loader_start(self, runner: "IRunner"):
        self.data[runner.loader_name] = {k: [] for k in ALL_KEYS}

    def on_batch_end(self, runner: "IRunner"):

        embeddings = self._detach(runner.output[self.embedding_key])
        filenames = runner.input[self.filename_key]
        labels = runner.input[self.labels_key]

        targets = self._detach(runner.input[self.target_key])

        data = self.data[runner.loader_name]

        data[INPUT_FILENAME_KEY].extend(filenames)
        data[OUTPUT_EMBEDDINGS_KEY].extend(embeddings)

        data[INPUT_TARGET_KEY].extend(targets)
        data[INPUT_LABEL_KEY].extend(labels)

        if self.logit_key is not None:
            predicts = self.activation_fn(runner.output[self.logit_key])
            predicts = self._detach(predicts)

            data[OUTPUT_TARGET_KEY].extend(predicts)

    def on_epoch_end(self, runner: "IRunner") -> None:
        if self.enable_doev1:
            self.doev1(runner=runner)
        if self.enable_doev2:
            self.doev2(runner=runner)
        if self.enable_benchmark:
            self.benchmark(runner=runner)

    def on_stage_end(self, runner: "IRunner") -> None:

        if self.save_dir is not None:
            save_to: Path = Path(runner.logdir) / "embeddings" / self.save_dir

            save_to.mkdir(parents=True, exist_ok=True)

            for loader_name, loader_data in self.data.items():
                pkl_data = {}
                for key in PKL_KEYS:
                    pkl_data[key] = loader_data[key]
                self._save_pickle(pkl_data, save_to, loader_name)

            data = self._gather_data(self.save_loaders)

            pkl_data = {}
            for key in data.keys():
                if key in PKL_KEYS:
                    pkl_data[key] = data[key]

            self._save_pickle(pkl_data, save_to, "all")
            # self._save_excel(xls_data, save_to, runner.stage_name)

            index = build_index(
                embeddings=data[OUTPUT_EMBEDDINGS_KEY],
                labels=data[INPUT_TARGET_KEY],
            )

            faiss.write_index(index, str(save_to / "database.index"))

import math
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .saving import read_pickle

from scipy import stats

from sklearn.neighbors import NearestNeighbors


def eval_metrics(y_true, y_pred):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1-score': f1_score(y_true, y_pred, average='weighted')}


def print_metrics(metrics):
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"--------------------- K Nearest Neighbors accuracy metrics ----------------------------")
    print(f"Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, "
          f"Recall: {metrics['recall']}, F1-score: {metrics['f1-score']}")
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------------------------------")


def nearest_neighbors_numpy(train_emb, test_emb=None, n_folds=1, metric='angular', method='most frequent', k=5):
    leave_one_out = test_emb is None

    if leave_one_out:
        test_emb = train_emb

    test_length = len(test_emb['targets'])
    x_train, y_train = np.asarray(train_emb['embeddings']), np.asarray(train_emb['targets'])
    x_test, y_test = np.asarray(test_emb['embeddings']), test_emb['targets']

    y_pred = None
    metrics = None
    while metrics is None:
        try:
            y_pred = []
            end_idx, batch_size = 0, math.ceil(test_length / n_folds)
            for s, start_idx in enumerate(range(0, test_length, batch_size)):

                print(f"Nearest Neighbors evaluation for {s + 1}th of {n_folds} folds started..")

                end_idx = min(start_idx + batch_size, test_length)

                x = x_test[start_idx: end_idx]

                if metric.startswith('angular'):
                    dot = np.dot(x, x_train.T)
                    if metric.endswith('norm'):
                        n1 = np.linalg.norm(x, axis=1)[:, np.newaxis]
                        n2 = np.linalg.norm(x_train.T, axis=0)[np.newaxis, :]
                        product = np.arccos(dot / n1 / n2)
                    else:
                        product = np.arccos(dot)
                elif metric == 'euclidean':
                    product = np.linalg.norm(x[:, np.newaxis] - x_train, axis=2)
                else:
                    raise Exception("Unknown metric")

                knn_indcs = product.argsort(axis=1)
                # knn_vals = np.take_along_axis(product, knn_indcs, axis=1)
                # knn_classes = np.take_along_axis(np.tile(y_train, (knn_indcs.shape[0], 1)), knn_indcs, axis=1)
                knn_classes = y_train[knn_indcs]

                if leave_one_out:
                    # first elements are always the same
                    # knn_indcs = knn_indcs[:, 1:]
                    # knn_vals = knn_vals[:, 1:]
                    knn_classes = knn_classes[:, 1:]

                if method == 'first':
                    y_pred.extend(knn_classes[:, 0].tolist())
                elif method == 'most frequent':
                    # knn_indcs = knn_indcs[:, 0:k]
                    # knn_vals = knn_vals[:, 0:k]
                    knn_classes = knn_classes[:, 0:k]

                    knn_classes, _ = stats.mode(knn_classes, axis=1)

                    y_pred.extend(knn_classes[:, 0].tolist())
                else:
                    raise Exception("Unknown metric")

            # y_pred = np.asarray(y_pred)
            # metrics = eval_metrics(y_test, y_pred)
            metrics = 1
        except MemoryError:
            metrics = None
            n_folds *= 2
            print(f"Memory error with {int(n_folds / 2)} fold, will try with {n_folds}..")

    return metrics, y_test, y_pred, n_folds


def nearest_neighbors_numpy2(train_emb, test_emb=None, n_folds=1, metric='euclidean', k=5):
    leave_one_out = test_emb is None

    if leave_one_out:
        test_emb = train_emb

    test_length = len(test_emb['captions'])
    x_train, y_train = train_emb['values'], train_emb['captions']
    x_test, y_test = test_emb['values'], test_emb['captions']

    metrics = None
    while metrics is None:
        try:
            y_pred = []

            classifier = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='brute')
            classifier.fit(x_train, y_train)

            end_idx, batch_size = 0, math.ceil(test_length / n_folds)
            for s, start_idx in enumerate(range(0, test_length, batch_size)):

                print(f"Nearest Neighbors evaluation for {s + 1}th of {n_folds} folds started..")

                end_idx = min(start_idx + batch_size, test_length)

                x = x_test[start_idx: end_idx]

                knn_ids = classifier.kneighbors(x, return_distance=False)

                if leave_one_out:
                    knn_ids = knn_ids[:, 1:]
                knn_ids = knn_ids[:, 0:k]

                knn_classes = y_train[knn_ids]
                knn_classes, _ = stats.mode(knn_classes, axis=1)

                y_pred.extend(knn_classes[:, 0].tolist())

            metrics = eval_metrics(y_test, y_pred)
        except MemoryError:
            metrics = None
            n_folds *= 2
            print(f"Memory error with {int(n_folds / 2)} fold, will try with {n_folds}..")

    return metrics, y_test, y_pred, n_folds

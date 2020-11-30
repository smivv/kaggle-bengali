from typing import List, Dict, Any, Union, Optional

import faiss

import numpy as np
import pandas as pd

from pathlib import Path
from ast import literal_eval
from scipy.stats import mode


def build_benchmark_index(path: Union[str, Path]):
    df = pd.read_excel(str(path), sheet_name='Benchmarking')

    df = df[df['Utilized'] == 1]
    df = df.loc[df['Test case'].notnull()]
    df['Test Set'] = df['Test Set'].apply(lambda x: literal_eval(x))

    index = {}
    for i, row in df.iterrows():
        cat, subcat, var = row['Category'], row['Subcategory'], row['Variant']
        for cam in row['Test Set']:
            index[f"{cat}_{subcat}_{var}_{cam}"] = row['Test case']

    return index


def build_index(
        embeddings: np.ndarray = None,
        labels: np.ndarray = None,
) -> faiss.Index:
    """
    Loads database from pickle.

    Args:
        embeddings (Optional[np.ndarray]): Embeddings.
        labels (Optional[np.ndarray]): Labels.

    Returns (faiss.Index): Index.

    """
    index = faiss.index_factory(
        embeddings.shape[1],
        "IDMap,Flat",
        faiss.METRIC_INNER_PRODUCT
    )

    index.add_with_ids(embeddings, labels)

    return index


def knn(
        index: faiss.Index,
        embedding: np.ndarray,
        labels2captions: Union[np.ndarray, Dict],
        top_k: int = 3,
        k: int = 1,
):
    """
        Performs kNN on factory index + index with `name`.

        Args:
            index (faiss.Index): Index object
            embedding (np.ndarray): Embeddings query.
            top_k (int): Top K results to return.
            k (int): K parameter in kNN.
            labels2captions (Dict[int, str]):

        Returns List[Dict]: Closest neighbors.

        """

    results = []

    # Search for closest embeddings in terms of inner product distance
    nn_distances, nn_labels = index.search(
        embedding[np.newaxis, ...], k=index.ntotal)
    nn_distances = np.clip(nn_distances, 0.0, 1.0)
    nn_distances = np.arccos(nn_distances)

    nn_distances = np.squeeze(nn_distances)
    nn_labels = np.squeeze(nn_labels)

    true_label = None

    top_k = min(top_k, len(np.unique(nn_labels)))
    for _ in range(top_k):

        if true_label is not None:
            not_equal_indcs = np.where(nn_labels != true_label)[0]
            nn_labels = nn_labels[not_equal_indcs]
            nn_distances = nn_distances[not_equal_indcs]

        # Tale first k neighbor classes
        if nn_labels.ndim == 0:
            true_label = int(nn_labels)
            true_caption = labels2captions[true_label]
            closest_distance = float(nn_distances)
        else:
            knn_labels = nn_labels[:k]

            # Find most frequent from them
            true_label = mode(knn_labels, axis=0)[0][0]
            true_caption = labels2captions[true_label]

            closest_index = nn_labels.tolist().index(true_label)
            closest_distance = nn_distances[closest_index]

        result = {
            "label": true_label,
            "caption": true_caption,
            "distance": closest_distance,
        }

        results.append(result)

    return results

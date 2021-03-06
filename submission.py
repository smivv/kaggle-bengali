import time
import torch
import torch.nn.functional as F

import pandas as pd

from sklearn.metrics import recall_score

import safitty
from catalyst.utils import load_ordered_yaml

from src.models.bengali.multiheadnet import MultiHeadNet
from src.transforms.bengali import get_transforms


def softmax(outputs, k=None):
    probs, labels = {}, {}

    for i, v in outputs.items():
        values = F.softmax(v, dim=1).data.cpu()

        if k is not None:
            values, indices = values.topk(k=k, dim=1)
            probs[i], labels[i] = values, indices
        else:
            probs[i] = values

    if k is not None:
        return probs, labels
    else:
        return probs


def topk(outputs, k=1):
    probs, labels = {}, {}

    for i, v in outputs.items():
        values, indices = v.topk(k=k, dim=1)
        probs[i] = values.numpy().squeeze()
        labels[i] = indices.numpy().squeeze()

    return probs, labels


def predict(models: list, image: torch.Tensor):
    outputs = None
    with torch.no_grad():
        for model in models:
            g, v, c = model["model"](image)

            values = softmax({})

            if outputs is None:
                outputs = {}
            else:
                outputs[GRAPHEME_INPUT_KEY] += values[GRAPHEME_INPUT_KEY]
                outputs[VOWEL_INPUT_KEY] += values[VOWEL_INPUT_KEY]
                outputs[CONSONANT_INPUT_KEY] += values[CONSONANT_INPUT_KEY]

    for k in INPUT_KEYS:
        outputs[k] /= len(models)

    values, labels = topk(outputs)

    return values, labels


JIT = True
TARGET_TO_USE = [0, 1, 2]

DATASET_PATH = "/workspace/Datasets/BENGALI/"
# DATASET_PATH = "/kaggle/input/bengaliai-cv19/"

num_classes = get_num_classes(DATASET_PATH)

transforms = get_transforms(
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
)
transforms["train"] = transforms["test"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODELS = [
    {
        "path": "/home/smirnvla/PycharmProjects/catalyst-classification/logs/se_resnext50_32x4d_5e-3_radaml_onplateu_strat_aug_neck_2lheads/trace/se_resnext50_32x4d_9806",
    },
]

for M in MODELS:
    if JIT:
        model = torch.jit.load(M["path"] + ".jit", map_location=device)
    else:
        with open(M["path"] + ".yml", "r") as f:
            config = load_ordered_yaml(f)

            model = MultiHeadNet.get_from_params(
                backbone_params=safitty.get(config, 'model_params', 'backbone_params'),
                neck_params=safitty.get(config, 'model_params', 'neck_params'),
                heads_params=safitty.get(config, 'model_params', 'heads_params')
            )

            checkpoint = torch.load(M["path"] + ".pth", map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    M["model"] = model


row_id, timings = [], []

gt, target = [], []

BATCH_SIZE = 32
TEST_ONLY = False

with torch.no_grad():
    for file_to_load in range(4):

        start = time.time()

        loader = get_loaders(dataset_path=DATASET_PATH,
                             transforms=transforms,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             target_to_use=TARGET_TO_USE,
                             test_only=TEST_ONLY,
                             load_from="parquet",
                             use_original=True,
                             files_to_load=[file_to_load])[
            "test" if TEST_ONLY else "train"]

        end = time.time()

        print(
            f"--- {end - start} seconds for loading {len(loader)} batches ---")

        for batch_idx, batch in enumerate(loader):

            start = time.time()

            image = batch["image"]
            image_ids = batch["image_id"]

            values, labels = predict(MODELS, image.to(device))

            for i, image_id in enumerate(image_ids):
                row_id += [
                    f"{image_id}_{GRAPHEME_INPUT_KEY}",
                    f"{image_id}_{VOWEL_INPUT_KEY}",
                    f"{image_id}_{CONSONANT_INPUT_KEY}",
                ]
                target += [
                    labels[GRAPHEME_INPUT_KEY][i],
                    labels[VOWEL_INPUT_KEY][i],
                    labels[CONSONANT_INPUT_KEY][i]
                ]
                gt += [
                    batch[GRAPHEME_INPUT_KEY][i],
                    batch[VOWEL_INPUT_KEY][i],
                    batch[CONSONANT_INPUT_KEY][i]
                ]

            end = time.time()

            timings.append(end - start)

            del image, labels, start, end

            print("--- %s recall ---" % (
                recall_score(gt, target, average='macro')))

            print("--- %s seconds for batch in average ---" % (
                    sum(timings) / len(timings)))

        del loader

sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
sub_df.to_csv('submission.csv', index=False)

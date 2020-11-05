import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from src.utils.saving import save_pickle, read_pickle, map_embeddings
from src.utils.knn import nearest_neighbors_numpy
from src.utils.plots import plot_confusion_matrix
from src.utils.utils import build_index


join = os.path.join
exists = os.path.exists
basename = os.path.basename

DATASET_DIR = '/workspace/Datasets/CICO1.5/'
# EMB_DIR = '/home/smirnvla/PycharmProjects/tensorflow-models/research/slim/tmp/train/20191010_cico15_nasnet_mobile_1e+1_256_sgd_triplet/benchmarks/all'
EMB_DIR = '/home/smirnvla/PycharmProjects/catalyst-classification/logs/cico15_256_arcface_64_5e-1_radam_plateu_1e-0_l2neck_stratified_v2test/embeddings/cico15'
BENCH_PATH = join(DATASET_DIR, 'CICO 1.5 benchmarking_plan_v12_FINAL_Working Document v2.xlsx')

INDEX = build_index(BENCH_PATH)

CLASSES = np.asarray([
    'Bread_and_Rolls__Baguette',
    'Bread_and_Rolls__Rolls_frozen',
    'Cake_and_Pastry__Muffins',
    'No_Food__',
    'Pizza__',
    'Pizza_and_Quiche__Onion_Tart',
    'Poultry__Chicken,_whole',
    'Poultry__Chicken_legs',
    'Poultry__Chicken_nuggets',
    'Poultry__Chicken_wings',
    'Side_dishes__Croquettes',
    'Side_dishes__French_fries_frozen',
    'Side_dishes__Wedges',
    'Unknown__Cookies',
    'Vegetables__Mediterranean_Vegetables'
  ])

train_emb = read_pickle(EMB_DIR, 'infer_train')
test_emb = read_pickle(EMB_DIR, 'infer_valid')
all_emb = {k: train_emb[k] + test_emb[k] for k in train_emb.keys()}

n_folds = 64
key = "knn_predicts"

if key not in test_emb:
    _, test_y_true, test_y_pred, _ = nearest_neighbors_numpy(train_emb, test_emb, n_folds)
    test_emb[key] = test_y_pred
    save_pickle(test_emb, EMB_DIR, "infer_valid")
else:
    test_y_true, test_y_pred = test_emb["targets"], test_emb[key]

# plot_confusion_matrix(test_y_true, test_y_pred, classes=CLASSES, save_to=os.path.join(DATASET_DIR, '_main_cm.png'))
plot_confusion_matrix(test_y_true, test_y_pred, classes=CLASSES, normalize=True, save_to=os.path.join(DATASET_DIR, '_main_cm_norm.png'))

RESULT = {k: {'true': [], 'pred': []} for k in ['size', 'dishware', 'trial', 'scale', 'recipe kind']}

for i, (path, true, pred) in enumerate(zip(test_emb['filenames'], test_y_true, test_y_pred)):

    filename = basename(path).split('.jpg')[0]

    if "Camera" not in filename:
        continue

    if '5000k' in filename:
        category, subcategory, variant, _, _, camera, _ = filename.split('_')
    else:
        category, subcategory, variant, _, camera, _ = filename.split('_')

    test_case = INDEX[f"{category}_{subcategory}_{variant}_{camera}"]

    for k, v in RESULT.items():
        if k in test_case:
            RESULT[k]['true'].append(true)
            RESULT[k]['pred'].append(pred)

for k, v in RESULT.items():
    # plot_confusion_matrix(v['true'], v['pred'], classes=CLASSES, save_to=join(DATASET_DIR, f'_{k}_cm.png'))
    plot_confusion_matrix(v['true'], v['pred'], classes=CLASSES, normalize=True, save_to=join(DATASET_DIR, f'_{k}_cm_norm.png'))
    open(join(DATASET_DIR, f"_{k}_{accuracy_score(v['true'], v['pred'])}"), 'w').close()

"""--------------------------------------------------------------------------"""

# if key not in all_emb:
#     _, all_y_true, all_y_pred, _ = nearest_neighbors_numpy(all_emb, n_folds)
#     all_emb[key] = all_y_pred
#     save_pickle(all_emb, EMB_DIR, "infer_all")
# else:
#     all_y_true, all_y_pred = all_emb["targets"], all_emb[key]
#
# for k, v in {
#     'train': train_emb,
#     'test': test_emb,
#     'all': all_emb
# }.items():
#     _, y_true, y_pred, n_folds = nearest_neighbors_numpy(v, n_folds=n_folds)
#
#     plot_confusion_matrix(y_true, y_pred, save_to=os.path.join(DATASET_DIR, f'_{k}_cm.png'))
#     plot_confusion_matrix(y_true, y_pred, normalize=True, save_to=os.path.join(DATASET_DIR, f'_{k}_cm_norm.png'))
#     open(join(DATASET_DIR, f"_{k}_{accuracy_score(y_true, y_pred)}"), 'w').close()

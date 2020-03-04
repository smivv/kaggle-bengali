import os
import json
import shutil

path = "/home/smirnvla/PycharmProjects/catalyst-classification/logs"
experiment = "se_resnext50_32x4d_5e-3_radaml_onplateu_strat_aug_neck_2lheads"
name = "se_resnext50_32x4d"

with open(os.path.join(path, experiment, "checkpoints", "_metrics.json")) as f:
    recall = int(json.load(f)["best"]["_total_recall"] * 100)

    ckpt_from = os.path.join(path, experiment, "checkpoints", "best.pth")
    ckpt_to = os.path.join(path, experiment, "trace", f"{name}_{recall}.pth")
    shutil.copyfile(ckpt_from, ckpt_to)

    config_from = os.path.join(path, experiment, "configs", "resnext50_ce_1e-3_1st_radam.yml")
    config_to = os.path.join(path, experiment, "trace", f"{name}_{recall}.yml")
    shutil.copyfile(config_from, config_to)

    jit_from = os.path.join(path, experiment, "trace", "traced-best-forward.pth")
    jit_to = os.path.join(path, experiment, "trace", f"{name}_{recall}.jit")
    os.rename(jit_from, jit_to)

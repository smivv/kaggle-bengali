# flake8: noqa

from .experiment import Experiment
from .runner import ModelRunner as Runner

from catalyst.dl import registry
from catalyst.contrib.dl.callbacks import MetricAggregatorCallback

from src.losses.arcface import ArcFaceProduct, ArcFaceLoss
from src.losses.dense import DenseCrossEntropy
from src.losses.center import CenterLoss
from src.losses.focal import FocalLossMultiClassFixed
from src.callbacks.tensorboard import VisualizationCallback
from src.callbacks.recall import RecallCallback
from src.callbacks.cutmixup import CutMixUpCallback
from src.schedulers.cosine import CosineAnnealingWarmUpRestarts
from src.models.multiheadnet import MultiHeadNet

registry.Callback(MetricAggregatorCallback)
registry.Callback(RecallCallback)
registry.Callback(CutMixUpCallback)
registry.Callback(VisualizationCallback)
registry.Criterion(FocalLossMultiClassFixed)
registry.Criterion(DenseCrossEntropy)
registry.Criterion(ArcFaceLoss)
registry.Criterion(CenterLoss)
registry.Module(ArcFaceProduct)
registry.Scheduler(CosineAnnealingWarmUpRestarts)

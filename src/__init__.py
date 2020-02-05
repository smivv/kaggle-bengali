# flake8: noqa

from .experiment import Experiment
from .runner import ModelRunner as Runner

from catalyst.dl import registry
from catalyst.contrib.dl.callbacks import MetricAggregatorCallback

from src.losses.arcface import ArcFaceProduct, ArcFaceLoss
from src.callbacks.tensorboard import VisualizationCallback
from src.models.efficientnet import EfficientNetMultiHeadNet

registry.Callback(MetricAggregatorCallback)
registry.Callback(VisualizationCallback)
registry.Module(ArcFaceProduct)
registry.Criterion(ArcFaceLoss)

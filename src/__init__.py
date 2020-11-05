# flake8: noqa

from .experiment import Experiment

from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from src.callbacks.tensorboard import VisualizationCallback
from src.callbacks.visualizer import VisualizationCallback2
from src.callbacks.cico.benchmark import BenchmarkingCallback
from src.callbacks.cico.embeddings import EmbeddingsLoggerCallback
from src.models.cico.generic import GenericModel
from src.schedulers.cosine import CosineAnnealingWarmUpRestarts
from src.losses.cico.arcface import ArcFaceLinear, ArcFaceLoss, L2Norm

registry.Module(L2Norm)
registry.Model(GenericModel)
registry.Criterion(ArcFaceLoss)
registry.Module(ArcFaceLinear)
registry.Callback(VisualizationCallback)
registry.Callback(VisualizationCallback2)
registry.Callback(BenchmarkingCallback)
registry.Callback(EmbeddingsLoggerCallback)
registry.Scheduler(CosineAnnealingWarmUpRestarts)

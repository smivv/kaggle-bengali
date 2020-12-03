# flake8: noqa

from .experiment import Experiment

from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner

from src.callbacks.tensorboard import VisualizationCallback, ProjectorCallback
from src.callbacks.cico.doe import DoECallback
from src.callbacks.cico.benchmark import BenchmarkingCallback
from src.models.cico.generic import GenericModel
from src.schedulers.cosine import CosineAnnealingWarmUpRestarts
from src.losses.cico.arcface import ArcFaceLinear, ArcFaceLoss, L2Norm
from src.losses.cico.triplet import TripletSemiHardLoss

registry.Model(GenericModel)

registry.Module(L2Norm)
registry.Module(ArcFaceLinear)

registry.Criterion(ArcFaceLoss)
registry.Criterion(TripletSemiHardLoss)

registry.Callback(VisualizationCallback)
registry.Callback(ProjectorCallback)

registry.Callback(DoECallback)
registry.Callback(BenchmarkingCallback)

registry.Scheduler(CosineAnnealingWarmUpRestarts)

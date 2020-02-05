from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils
from catalyst.dl import registry
from catalyst.contrib.models import SequentialNet

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters


@registry.Model
class EfficientNetMultiHeadNet(nn.Module):
    def __init__(
            self,
            backbone: EfficientNet,
            neck: nn.ModuleList,
            heads: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck or (lambda *args: args)
        self.heads = heads

    def forward(self, x: torch.Tensor):
        bs = x.size(0)

        x = self.backbone.extract_features(x)

        # Pooling and final linear layer
        x = self.backbone._avg_pooling(x)
        features = x.view(bs, -1)

        embeddings = self.neck(features)

        result = {
            "features": features,
            "embeddings": embeddings
        }

        for key, head in self.heads.items():
            result[key] = head(embeddings)

        return result

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        model = cls.from_name(model_name, override_params=override_params)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels,
                                      kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_from_params(
            cls,
            backbone_params: Dict = None,
            neck_params: Dict = None,
            heads_params: Dict = None,
    ) -> "EfficientNetMultiHeadNet":

        backbone_params_ = deepcopy(backbone_params)
        neck_params_ = deepcopy(neck_params)
        heads_params_ = deepcopy(heads_params)

        if "requires_grad" in backbone_params_:
            requires_grad = backbone_params_.pop("requires_grad")
        else:
            requires_grad = False

        if "pretrained" in backbone_params_:
            pretrained = backbone_params_.pop("pretrained")
        else:
            pretrained = True

        if "in_channels" in backbone_params_:
            in_channels = backbone_params_.pop("in_channels")
        else:
            in_channels = 3

        if pretrained is not None:
            backbone = EfficientNet.from_pretrained(**backbone_params_)
        else:
            backbone = EfficientNet.from_name(**backbone_params_)

        backbone.set_swish(memory_efficient=True)

        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=backbone._global_params.image_size)
            out_channels = round_filters(32, backbone._global_params)
            backbone._conv_stem = Conv2d(in_channels, out_channels,
                                         kernel_size=3, stride=2, bias=False)

        for param in backbone.parameters():
            param.requires_grad = requires_grad

        enc_size = backbone._conv_head.out_channels

        neck_params_["hiddens"].insert(0, enc_size)
        emb_size = neck_params_["hiddens"][-1]

        if neck_params_ is not None:
            neck = SequentialNet(**neck_params_)
        else:
            neck = None

        if heads_params_ is not None:
            head_kwargs_ = {}
            for head, params in heads_params_.items():
                if isinstance(heads_params_, int):
                    head_kwargs_[head] = nn.Linear(emb_size, params, bias=True)
                elif isinstance(heads_params_, dict):
                    params["hiddens"].insert(0, emb_size)
                    head_kwargs_[head] = SequentialNet(**params)
            heads = nn.ModuleDict(head_kwargs_)
        else:
            heads = None

        return cls(
            backbone=backbone,
            neck=neck,
            heads=heads
        )

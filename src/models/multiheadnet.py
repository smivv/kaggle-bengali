from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn

import pretrainedmodels

from catalyst import utils
from catalyst.dl import registry
from catalyst.contrib.models import SequentialNet

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters

from ..datasets.bengali import INPUT_KEYS, NUM_CLASSES, \
    GRAPHEME_OUTPUT_KEY, VOWEL_OUTPUT_KEY, CONSONANT_OUTPUT_KEY


@registry.Model
class MultiHeadNet(nn.Module):
    def __init__(
            self,
            backbone: EfficientNet,
            neck: nn.ModuleList = None,
            heads: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads

    def forward(self, x: torch.Tensor):
        bs = x.size(0)

        # x = self.backbone.conv0(x)

        if hasattr(self.backbone, "extract_features"):
            x = self.backbone.extract_features(x)
        elif hasattr(self.backbone, "features"):
            x = self.backbone.features(x)
        else:
            raise NotImplementedError("Method not found")

        # Pooling and final linear layer
        x = self.backbone._adapt_avg_pooling(x)
        features = x.view(bs, -1)
        # features = self.backbone._dropout(x.view(bs, -1))

        embeddings = self.neck(features) if self.neck is not None else features

        result = {
            "features": features,
            "embeddings": embeddings
        }

        for key, head in self.heads.items():
            result[key] = head(embeddings)

        return result[GRAPHEME_OUTPUT_KEY], \
               result[VOWEL_OUTPUT_KEY], \
               result[CONSONANT_OUTPUT_KEY]

    @classmethod
    def get_from_params(
            cls,
            backbone_params: Dict = None,
            neck_params: Dict = None,
            heads_params: Dict = None,
    ) -> "MultiHeadNet":

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

        if backbone_params_["model_name"] in pretrainedmodels.__dict__:
            model_name = backbone_params_.pop("model_name")

            backbone = pretrainedmodels.__dict__[model_name](
                num_classes=1000,
                pretrained="imagenet" if pretrained else None
            )

            if in_channels != 3:
                if model_name == "pnasnet5large":
                    old_conv = backbone.conv_0.conv
                    backbone.conv_0.conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        bias=False
                    )
                    backbone.conv_0.conv.weight = nn.Parameter(
                        data=old_conv.weight.data[:, 0, :, :].unsqueeze(1),
                        requires_grad=requires_grad
                    )
                elif model_name in ["se_resnext50_32x4d",
                                    "se_resnext101_32x4d"]:
                    old_conv = backbone.layer0.conv1
                    backbone.layer0.conv1 = nn.Conv2d(
                        in_channels=1,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        padding=(3, 3),
                        stride=old_conv.stride,
                        bias=False
                    )

                    backbone.layer0.conv1.weight = nn.Parameter(
                        data=old_conv.weight.data[:, 0, :, :].unsqueeze(1),
                        requires_grad=requires_grad
                    )

                    # backbone.avg_pool = backbone.last_linear = None

                elif model_name == "densenet161":
                    old_conv = backbone.conv0
                    backbone.conv0 = nn.Conv2d(
                        in_channels=1,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        bias=False
                    )
                    backbone.conv0.weight = nn.Parameter(
                        data=old_conv.weight.data[:, 0, :, :].unsqueeze(1),
                        requires_grad=requires_grad
                    )
                else:
                    raise NotImplementedError("This model not yet implemented")
                # backbone.conv0 = nn.Conv2d(
                #     in_channels=1,
                #     out_channels=3,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     bias=True,
                # )
            enc_size = backbone.last_linear.in_features

        elif backbone_params_["model_name"].startswith("efficientnet"):
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
                                             kernel_size=3,
                                             stride=2, bias=False)

            enc_size = backbone._conv_head.out_channels
        else:
            raise NotImplementedError("This model not yet implemented")

        backbone._adapt_avg_pooling = nn.AdaptiveAvgPool2d(1)
        backbone._dropout = nn.Dropout(p=0.2)

        for param in backbone.parameters():
            param.requires_grad = requires_grad

        neck = None
        if neck_params_:
            neck_params_["hiddens"].insert(0, enc_size)
            emb_size = neck_params_["hiddens"][-1]

            if neck_params_ is not None:
                neck = SequentialNet(**neck_params_)
            neck.requires_grad = requires_grad
        else:
            emb_size = enc_size

        # heads = {}
        # for key, out_size in zip(INPUT_KEYS, NUM_CLASSES):
        #     heads[key] = nn.Linear(in_features=emb_size,
        #                            out_features=out_size, bias=True)
        if heads_params_ is not None:
            head_kwargs_ = {}
            for head, params in heads_params_.items():
                if isinstance(heads_params_, int):
                    head_kwargs_[head] = nn.Linear(emb_size, params, bias=True)
                elif isinstance(heads_params_, dict):
                    params["hiddens"].insert(0, emb_size)
                    head_kwargs_[head] = SequentialNet(**params)
                head_kwargs_[head].requires_grad = requires_grad
            heads = nn.ModuleDict(head_kwargs_)
        else:
            heads = None

        model = cls(
            backbone=backbone,
            neck=neck,
            heads=heads
        )

        utils.set_requires_grad(model, requires_grad)

        print(model)
        return model

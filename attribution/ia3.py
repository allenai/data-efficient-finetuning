"""
Adapted from t-few repo:
https://github.com/r-three/t-few/blob/master/src/models/lora.py
"""
import re
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from attribution.model import BasicSeq2Seq

logger = logging.getLogger(__name__)


class IA3Linear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.ia3_vector = nn.Parameter(
            torch.ones(1, linear_layer.in_features)
        )

    def forward(self, input):
        return F.linear((input * self.ia3_vector.flatten()), self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None,
        )


def modify_with_ia3(transformer: PreTrainedModel, lora_modules: str, lora_layers: str):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"IA3 can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        IA3Linear(layer),
                    )
    return transformer


@Model.register("ia3_seq2seq")
class IA3BasicSeq2Seq(BasicSeq2Seq):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        # regex from https://github.com/r-three/t-few/blob/master/configs/ia3.json
        self.transformer = modify_with_ia3(
            self.transformer,
            ".*SelfAttention|.*EncDecAttention|.*DenseReluDense",
            "k|v|wi_1.*"
        )
        # only train ia3 parameters
        for name, param in self.transformer.named_parameters():
            if "ia3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

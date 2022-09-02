from typing import Any, Dict, List
from overrides import overrides
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import Average

from model import BasicSeq2Seq
from ia3 import modify_with_lora

logger = logging.getLogger(__name__)


@Model.register("ia3_seq2seq")
class IA3BasicSeq2Seq(BasicSeq2Seq):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        # regex from https://github.com/r-three/t-few/blob/master/configs/ia3.json
        self.transformer = modify_with_lora(
            self.transformer,
            ".*SelfAttention|.*EncDecAttention|.*DenseReluDense",
            "k|v|wi_1.*"
        )
        # only train lora parameters
        for name, param in self.transformer.named_parameters():
            if "ia3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

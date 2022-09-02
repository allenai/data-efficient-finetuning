import logging

from allennlp.models import Model

from attribution.model import BasicSeq2Seq
from attribution.ia3 import modify_with_lora

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

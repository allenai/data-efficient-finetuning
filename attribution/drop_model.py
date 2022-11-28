from typing import Dict, List
from overrides import overrides
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)


# just for drop eval
@Model.register("drop_model")
class DropModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str = "google/t5-small-lm-adapt",
        max_length: int = 128,
        fake_training: bool = False,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_length
        self._accuracy = Average()
        self._fake_training = fake_training
        self._use_drop_metrics = True
        if self._fake_training:
            logger.info(
                "Faking training. This will only dump the pretrained transformer into a model archive."
            )

    def forward(
        self,
        prompt_and_input: TextFieldTensors,
        query_id: List[str] = None,
        target: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(prompt_and_input)
        attention_mask = util.get_text_field_mask(prompt_and_input)

        target_ids = util.get_token_ids_from_text_field_tensors(target)
        answer_option_ids = target_ids
        answer_option_ids[answer_option_ids == 0] = -100

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=answer_option_ids,
            use_cache=False,
            return_dict=True,
        )
        loss = output["loss"]
        if self._fake_training:
            loss = loss * 0.0
        output_dict = {"loss": loss}
        if not self.training:
            outputs = self.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=self.max_len,
            )
            if not self._use_drop_metrics:
                self._accuracy(target_ids == outputs)
            else:
                output_dict["answer"] = [
                    self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs
                ]
                output_dict["query_id"] = query_id
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}

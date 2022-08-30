from typing import Any, Dict, List
from overrides import overrides
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)


@Model.register("seq2seq")
class BasicSeq2Seq(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str = 'google/t5-xl-lm-adapt',
        compute_test_metrics: bool = True,
        relevant_label_index: int=None,
        gradient_checkpointing: bool=False,
        fake_training: bool = False,
        checkpoint_for_initialization: str = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if checkpoint_for_initialization:
            logger.info(f"Loading weights from checkpoint: {checkpoint_for_initialization}")
            self.load_state_dict(torch.load(checkpoint_for_initialization))
        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._compute_test_metrics = compute_test_metrics
        self._accuracy = Average()
        # We use this to compute precision and recall. If not set, precision and recall will be 0.
        self._relevant_label_index = relevant_label_index
        self._precision = Average()
        self._recall = Average()
        self._fake_training = fake_training
        if self._fake_training:
            logger.info("Faking training. This will only dump the pretrained transformer into a model archive.")

    def forward(
        self,
        prompt_and_input: TextFieldTensors,
        answer_options: TextFieldTensors = None,
        correct_answer_index: torch.Tensor = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(prompt_and_input)
        attention_mask = util.get_text_field_mask(prompt_and_input)

        # (batch_size, num_options, answer_length)
        answer_option_ids = util.get_token_ids_from_text_field_tensors(answer_options)
        answer_option_ids[answer_option_ids == 0] = -100
        # (batch_size, answer_length)
        correct_answer_ids = answer_option_ids[
                torch.arange(answer_option_ids.shape[0]),
                correct_answer_index.squeeze()
        ]

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=correct_answer_ids,
            use_cache=False,
            return_dict=True,
        )
        loss = output['loss']
        if self._fake_training:
            loss = loss * 0.0

        output_dict = {'loss': loss, 'response': []}
        if not self.training and self._compute_test_metrics:
            batch_size, num_options, _ = answer_option_ids.shape
            for i in range(batch_size):
                instance_input_ids = input_ids[i:i+1]
                instance_attention_mask = attention_mask[i:i+1]
                instance_metadata = metadata[i]
                correct_option_id = correct_answer_index[i].detach().cpu()[0]
                min_loss = None
                best_option_id = None
                for j in range(num_options):
                    # (1, answer_length)
                    option_ids = answer_option_ids[i:i+1, j:j+1].squeeze(1)
                    option_output = self.transformer(
                        input_ids=instance_input_ids,
                        attention_mask=instance_attention_mask,
                        labels=option_ids,
                        use_cache=False,
                        return_dict=True
                    )
                    option_loss = option_output['loss'].detach().cpu()
                    if min_loss is None or min_loss > option_loss:
                        min_loss = option_loss
                        best_option_id = j
                self._accuracy(correct_option_id == best_option_id)
                if best_option_id == self._relevant_label_index:
                    self._precision(correct_option_id == best_option_id)

                if correct_option_id == self._relevant_label_index:
                    self._recall(correct_option_id == best_option_id)

                output_dict['response'].append(best_option_id)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "precision": self._precision.get_metric(reset),
            "recall": self._recall.get_metric(reset),
        }

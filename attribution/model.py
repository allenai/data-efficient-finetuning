from typing import Any, Dict, List
from overrides import overrides
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import Average, FBetaMeasure

logger = logging.getLogger(__name__)


@Model.register("seq2seq")
class BasicSeq2Seq(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str = 'google/t5-xl-lm-adapt',
        relevant_label_index: int=0,
        gradient_checkpointing: bool=False,
        fake_training: bool = False,
        checkpoint_for_initialization: str = None,
        weights_file : str = None,
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
        self._accuracy = Average()
        # used for LexGLUE tasks
        self._micro = FBetaMeasure(average="micro")
        self._macro = FBetaMeasure(average="macro")
        # We use this to compute precision and recall. If not set, precision and recall will be 0.
        self._relevant_label_index = relevant_label_index
        self._precision = Average()
        self._recall = Average()
        self._fake_training = fake_training
        self.loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")  # match hf t5
        if self._fake_training:
            logger.info("Faking training. This will only dump the pretrained transformer into a model archive.")
        if weights_file is not None:
            with open(weights_file, 'rb') as f:
                self.load_state_dict(torch.load(f))


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
        if not self.training:
            batch_size, num_options, _ = answer_option_ids.shape
            for i in range(batch_size):
                # setup - we pass through all options as a batch for minor speedup
                instance_input_ids = input_ids[i:i+1]
                instance_input_ids = instance_input_ids.repeat(num_options, 1)
                instance_attention_mask = attention_mask[i:i+1]
                instance_attention_mask = instance_attention_mask.repeat(num_options, 1)
                correct_option_id = correct_answer_index[i].detach().cpu()[0]
                option_ids = answer_option_ids[i:i+1].squeeze(0)
                # pass through
                option_output = self.transformer(
                    input_ids=instance_input_ids,
                    attention_mask=instance_attention_mask,
                    labels=option_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = option_output["logits"].detach()
                losses = self.loss_fct(logits.permute([0, 2, 1]), option_ids)
                losses = losses.sum(dim=-1) #/ (losses != 0).sum(dim=-1)
                min_loss = None
                best_option_id = 0
                for j, option_loss in enumerate(losses):
                    if min_loss is None or min_loss > option_loss:
                        min_loss = option_loss
                        best_option_id = j
                self._accuracy(correct_option_id == best_option_id)
                # None since we need a batch_size dim.
                option_losses = -losses[None, ].detach().cpu()
                self._micro(option_losses, torch.tensor([correct_option_id]))
                self._macro(option_losses, torch.tensor([correct_option_id]))
                if best_option_id == self._relevant_label_index:
                    self._precision(correct_option_id == best_option_id)

                if correct_option_id == self._relevant_label_index:
                    self._recall(correct_option_id == best_option_id)

                output_dict['response'].append(best_option_id)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = {
            "accuracy": self._accuracy.get_metric(reset),
            "precision": self._precision.get_metric(reset),
            "recall": self._recall.get_metric(reset),
        }
        # Without this check, when called before any evaluation, causes error
        if self._macro._true_positive_sum is not None:
            metrics_dict.update(
                {
                    "macro_f1": self._macro.get_metric(reset)["fscore"],
                    "micro_f1": self._micro.get_metric(reset)["fscore"],
                }
            )
        return metrics_dict

# a regular model, but we load the underlying model from a .th file.
# useful for training on top of other trained models.
@Model.register("load_seq2seq")
class LoadBasicSeq2Seq(BasicSeq2Seq):
    def __init__(
        self,
        load_from_file:str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if load_from_file is not None:
            with open(load_from_file, 'rb') as f:
                self.load_state_dict(torch.load(f))
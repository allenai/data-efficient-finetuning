import json
import logging
import random
from collections import defaultdict
from typing import List, Iterable, Optional, Tuple, Dict
import torch

from overrides import overrides
import datasets

from allennlp.data.fields import (
    MetadataField,
    TextField,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from transformers import AutoTokenizer

from natural_instructions.ni_collator import DataCollatorForNI


logger = logging.getLogger(__name__)

# full reader for ni instructions, similar to drop reader
@DatasetReader.register("ni_reader")
class NaturalInstructionsReader(DatasetReader):
    def __init__(
        self,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 1024,
        split_name: str = "train",
        return_original_instance: bool = False,
        max_num_instances_per_task: int = 100,
        max_num_instances_per_eval_task: int = 100,
        num_pos_examples: int = 0,  # set to '2' for the 'tk-instruct' model
        add_task_definition: bool = True,  # set to true for tk-instruct
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.return_original_instance = return_original_instance
        self._dataset = datasets.load_dataset(
            "natural_instructions/ni_dataset.py",
            max_num_instances_per_task=max_num_instances_per_task,
            max_num_instances_per_eval_task=max_num_instances_per_eval_task,
            split=split_name,
        )
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)
        self._collator = DataCollatorForNI(
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            num_pos_examples=num_pos_examples,
            add_task_definition=add_task_definition,
            text_only=True,
            max_source_length=max_query_length
        )

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}
        self._max_query_length = max_query_length
        self._stats = defaultdict(int)


    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        for sample in self._dataset:
            converted_sample = self._collator([sample])
            yield self.text_to_instance(
                converted_sample['inputs'][0],
                sample['id'],
                converted_sample['labels'][0])

    def text_to_instance(
        self,
        input_text: str,
        query_id: str,
        target: str,
    ) -> Instance:
        fields = {}

        tokenized_input = self._tokenizer.tokenize(input_text)
        if len(tokenized_input) > self._max_query_length:
            self._stats["Truncated inputs"] += 1
            tokenized_input = tokenized_input[: self._max_query_length]

        input_field = TextField(tokenized_input)
        fields["prompt_and_input"] = input_field
        if self.return_original_instance:
            fields["pretokenized_input"] = input_text

        tokenized_target = self._tokenizer.tokenize(target)
        if len(tokenized_target) > self._max_query_length:
            self._stats["Truncated targets"] += 1
            tokenized_target = tokenized_target[: self._max_query_length]

        target_field = TextField(tokenized_target)
        fields["target"] = target_field
        if self.return_original_instance:
            fields["pretokenized_target"] = target

        query_id_field = MetadataField(query_id)
        fields["instance_id"] = query_id_field
        if self.return_original_instance:
            fields["preindexed_id"] = query_id

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["prompt_and_input"].token_indexers = self._token_indexers
        instance.fields["target"].token_indexers = self._token_indexers

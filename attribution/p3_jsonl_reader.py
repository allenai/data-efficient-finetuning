import json
import pickle
import logging
from collections import defaultdict
from typing import Any, Dict, List, Iterable, Text

from overrides import overrides

import torch

from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("p3_jsonl")
class P3ClusterReader(DatasetReader):
    def __init__(
        self,
        model_name: str = "google/t5-xl-lm-adapt",
        max_query_length: int = 512,
        max_answer_length: int = 256,
        return_original_input: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.return_original_input = return_original_input
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name)
        }
        self._max_query_length = max_query_length
        self._max_answer_length = max_answer_length
        self._stats = None

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        for instance in self.shard_iterable(self.__read(file_path)):
            yield instance 

    def __read(self, file_path: str) -> Iterable[Instance]:
        self._stats = defaultdict(int)
        logger.info(f"Reading data from {file_path}")
        for line in open(file_path):
            instance_data = json.loads(line)
            if "target" not in instance_data or "input" not in instance_data:
                self._stats["Instances without inputs or targets (skipped)"] += 1
                continue
            if not isinstance(instance_data["target"], str):
                self._stats["Instances whose targets are not strings (skipped)"] += 1
                continue
            target = instance_data["target"].strip()
            if "answer_choices" not in instance_data:
                self._stats["Instances without answer options (kept)"] += 1
                answer_options = [target]
            else:
                answer_options = [c.strip() for c in instance_data["answer_choices"]]
                if target not in answer_options:
                    answer_options.append(target)
                    self._stats["Instances with targets not in answer choices (kept)"] += 1
            yield self.text_to_instance(
                instance_data["input"],
                target,
                answer_options
            )

        logger.info("Dataset stats:")
        for key, value in self._stats.items():
            logger.info(f"\t{key}: {value}")

    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        input_text: str,
        target: str,
        options: List[str],
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        tokenized_input = self._tokenizer.tokenize(input_text)
        if len(tokenized_input) > self._max_query_length:
            self._stats["Truncated inputs"] += 1
            tokenized_input = tokenized_input[:self._max_query_length]

        input_field = TextField(tokenized_input)
        fields["prompt_and_input"] = input_field
        if self.return_original_input:
            fields['pretokenized_input'] = input_text

        answer_option_fields = []
        for option in options:
            tokenized_option = self._tokenizer.tokenize(option)
            if len(tokenized_option) > self._max_answer_length:
                self._stats["Truncated options"] += 1
                tokenized_option = tokenized_option[:self._max_answer_length]
            answer_option_fields.append(TextField(tokenized_option))
        options_list_field = ListField(answer_option_fields)
        fields["answer_options"] = options_list_field

        answer_index = None
        for i, option in enumerate(options):
            if target in option:
                answer_index = i
                break
        fields["correct_answer_index"] = IndexField(answer_index, options_list_field)
        if metadata is not None:
            fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["prompt_and_input"].token_indexers = self._token_indexers
        for field in instance.fields["answer_options"].field_list:
            field.token_indexers = self._token_indexers

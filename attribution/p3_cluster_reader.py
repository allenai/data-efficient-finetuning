import json
import pickle
import logging
from collections import defaultdict
from typing import Any, Dict, List, Iterable

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


@DatasetReader.register("p3_cluster")
class P3ClusterReader(DatasetReader):
    def __init__(
        self,
        p3_data_path: str,
        split_name: str,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._p3_data = json.load(open(p3_data_path))
        self._split_name = split_name
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name)
        }
        self._max_query_length = max_query_length
        self._stats = None

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        self._stats = defaultdict(int)
        logger.info("Reading the cluster file")
        cluster_data = pickle.load(open(file_path, "rb"))
        for dataset_name, cluster_info in cluster_data.items():
            for instance_id in cluster_info[self._split_name]:
                if dataset_name not in self._p3_data:
                    self._stats["Instances skipped due to missing dataset partitions"] += 1
                    continue
                if str(instance_id) not in self._p3_data[dataset_name][self._split_name]:
                    self._stats["Instances skipped due to missing instance ids"] += 1
                    continue
                instance_info = self._p3_data[dataset_name][self._split_name][str(instance_id)]
                if len(instance_info["options"]) <= 1:
                    self._stats["Instances without multiple options"] += 1
                elif len(instance_info["options"]) > 10:
                    self._stats["Instances with too many options"] += 1
                elif instance_info["target"] not in instance_info["options"]:
                    self._stats["Target not in options"] += 1
                elif not instance_info["is_correct"]:
                    self._stats["Instance has incorrect answer"] += 1
                else:
                    yield self.text_to_instance(
                            instance_info["input"],
                            instance_info["target"],
                            instance_info["options"],
                            {"dataset_name": instance_info["dataset"], "index": instance_info["index"]}
                    )

        print("Dataset stats:")
        for key, value in self._stats.items():
            print(f"\t{key}: {value}")

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

        answer_option_fields = [
                TextField(self._tokenizer.tokenize(option)) for option in options
        ]
        options_list_field = ListField(answer_option_fields)
        fields["answer_options"] = options_list_field

        answer_index = None
        for i, option in enumerate(options):
            if option == target:
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

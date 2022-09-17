import logging
from collections import defaultdict
from typing import Iterable, Optional, Tuple

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

from attribution.huggingface_readers import HuggingfaceReaderRankClassification


logger = logging.getLogger(__name__)


# full reader for drop to pass through query id
@DatasetReader.register("drop_reader")
class DROPReader(DatasetReader):
    def __init__(
        self,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 512,
        split_name: str = "train",
        val_size: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        dataset_name, subset_name = self.get_dataset_name()
        self._dataset_name = dataset_name
        self._subset_name = subset_name
        original_val_set = datasets.load_dataset(
            dataset_name, subset_name, split="validation"
        )
        small_val_size = val_size  # I consider under 2000 examples as small
        val_split_size = val_size  # when splitting out val, get 1000 examples
        seed = 42
        if split_name == "train":
            if len(original_val_set) >= small_val_size:
                self._dataset = datasets.load_dataset(
                    dataset_name, subset_name, split="train"
                )
            else:
                # for small val sets, split out val from train and use old val as test
                # this is because some casehold splits are specially designed, so I want
                # to keep these as-is (rather than just split the val set in half)
                self._dataset = datasets.load_dataset(
                    dataset_name, subset_name, split="train"
                ).train_test_split(test_size=val_split_size, seed=seed)["train"]
        if split_name == "validation":
            # for large val sets, just split out from val
            if len(original_val_set) >= small_val_size:
                self._dataset = original_val_set.train_test_split(
                    train_size=val_split_size, seed=seed
                )["train"]
            else:
                # for small val sets, split out val from train and use old val as test
                self._dataset = datasets.load_dataset(
                    dataset_name, subset_name, split="train"
                ).train_test_split(test_size=val_split_size, seed=seed)["test"]
        elif split_name == "test":
            # for large val sets, test is the small split from val
            if len(original_val_set) >= small_val_size:
                self._dataset = original_val_set.train_test_split(
                    train_size=val_split_size, seed=seed
                )["test"]
            else:
                # for small val sets, split our new val from train (val becomes test)
                self._dataset = datasets.load_dataset(
                    dataset_name, subset_name, split="validation"
                )
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}
        self._max_query_length = max_query_length
        self._stats = defaultdict(int)

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "drop", None

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using GPT-3 DROP prompt.
        input = (
            f"Passage: {instance['passage']}\nQuestion: {instance['question']}\nAnswer:"
        )
        answer = instance["answers_spans"]["spans"][0]
        return [[input, instance["query_id"], answer]]

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        for sample in self._dataset:
            converted_samples = self.hf_to_instance(sample)
            for inputs, qid, targets in converted_samples:
                yield self.text_to_instance(inputs, qid, targets)

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
        # fields["pretokenized_input"] = input_text

        tokenized_target = self._tokenizer.tokenize(target)
        if len(tokenized_target) > self._max_query_length:
            self._stats["Truncated targets"] += 1
            tokenized_target = tokenized_target[: self._max_query_length]

        target_field = TextField(tokenized_target)
        fields["target"] = target_field

        query_id_field = MetadataField(query_id)
        fields["query_id"] = query_id_field

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["prompt_and_input"].token_indexers = self._token_indexers
        instance.fields["target"].token_indexers = self._token_indexers


# drop reader that aligns with the other formats for multitask
# NOTE: don't use this for eval!
@DatasetReader.register("multi_task_drop_reader")
class DropMReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "drop", None

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        input = (
            f"Passage: {instance['passage']}\nQuestion: {instance['question']}\nAnswer:"
        )
        answer = [instance["answers_spans"]["spans"][0]]
        return [[input, answer, 0]]

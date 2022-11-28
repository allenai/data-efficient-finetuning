from gzip import READ
import json
import logging
import random
from collections import defaultdict
from typing import List, Iterable, Optional, Tuple, Dict

from overrides import overrides
import datasets

from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from attribution.huggingface_readers import *


logger = logging.getLogger(__name__)

READER_MAPPING = {
    "rte": RTEReader,
    "anli_r1": ANLIR1Reader,
    "anli_r2": ANLIR2Reader,
    "anli_r3": ANLIR3Reader,
    "cb": CBReader,
    "hellaswag": HellaSwagReader,
    "story_cloze": StoryClozeReader,
    "winogrande": WinoGrandeReader,
    "wsc": WSCReader,
    "copa": COPAReader,
    "wic": WiCReader
}

@DatasetReader.register("icl_reader")
class ICLReader(DatasetReader):
    def __init__(
        self,
        reader_class_name: str = 'rte',
        model_name='google/t5-base-lm-adapt',
        retrieve_file='dummy',
        split_name='train',
        max_query_length=1024,
        use_val_split=False,
        val_size=1000,
        **kwargs,
    ):
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        reader_class = READER_MAPPING[reader_class_name]
        if 'split_name' in kwargs:
            kwargs.pop('split_name')
        self.instance_reader = reader_class(model_name=model_name, max_query_length=max_query_length, split_name='validation', use_val_split=False, return_original_instance=True, **kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(model_name)
        self._train_reader = reader_class(model_name=model_name, split_name='train', use_val_split=False, return_original_instance=True, **kwargs)
        self.retrieve_file = retrieve_file
        self.retrieve_iterator = self._train_reader.read(self.retrieve_file)
        self.random = random.Random(42)
        self.max_query_length = max_query_length
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        for instance in self.instance_reader.read(file_path):
            instance = instance.fields
            text_input = instance['pretokenized_input'] + '\nAnswer:'
            self._stats['counter'] += 1
            while True:
                try:
                    sample_instance = next(self.retrieve_iterator)
                except StopIteration:
                    self.retrieve_iterator = self._train_reader.read(self.retrieve_file)
                    sample_instance = next(self.retrieve_iterator)
                sample_instance = sample_instance.fields
                icl_sample = sample_instance['pretokenized_input'] + '\n Answer:' + sample_instance['answer_options_pretokenized'][sample_instance['correct_answer_index_value']]
                if len(self._tokenizer.tokenize(icl_sample + '\n' + text_input)) < self.max_query_length:
                    text_input = icl_sample + '\n' + text_input
                    self._stats['num_examples'] += 1
                else:
                    break
            self._stats['avg_examples_per_instance'] = self._stats['num_examples'] / self._stats['counter']
            # write our augmented input back
            fields = {}
            fields['prompt_and_input'] = TextField(self._tokenizer.tokenize(text_input))
            fields['answer_options'] = instance['answer_options']
            fields['correct_answer_index'] = instance['correct_answer_index']
            yield Instance(fields)

        logger.info("Dataset stats:")
        for key, value in self._stats.items():
            logger.info("%s: %d", key, value)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["prompt_and_input"].token_indexers = self._token_indexers
        for field in instance.fields["answer_options"].field_list:
            field.token_indexers = self._token_indexers

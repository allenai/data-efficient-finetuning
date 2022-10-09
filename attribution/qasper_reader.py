import json
import pickle
import logging
from collections import defaultdict
from typing import Any, Dict, List, Iterable
import random

from overrides import overrides

import torch

from allennlp.common.util import JsonDict
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
random.seed(23019)

@DatasetReader.register("qasper_evidence_prompt")
class QasperEvidencePromptReader(DatasetReader):
    def __init__(
        self,
        model_name: str = "bigscience/T0_3B",
        max_query_length: int = 512,
        answer_options: List[str] = ["Yes", "No"],
        negative_sample_ratio: float = 1.0,
        return_original_query: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._return_original_query = return_original_query
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name)
        }
        self._max_query_length = max_query_length
        self._answer_options = answer_options
        self._negative_sample_ratio = negative_sample_ratio
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading the dataset")
        with open(file_path, "r") as datafile:
            data = json.load(datafile)
            for article_id, article in self.shard_iterable(data.items()):
                if not article["full_text"]:
                    continue
                article["article_id"] = article_id
                yield from self._article_to_instances(article)

        logger.info("Dataset stats:")
        for key, value in self._stats.items():
            logger.info("%s: %d", key, value)

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_article(article)
        self._stats["number of documents"] += 1
        for question_answer in article["qas"]:
            question = question_answer['question']
            self._stats["number of questions"] += 1
            self._stats["number of answers"] += len(question_answer["answers"])
            if len(question_answer["answers"]) > 1:
                self._stats["questions with multiple answers"] += 1

            all_evidence = set()
            for answer_annotation in question_answer["answers"]:
                evidence = self._extract_evidence(
                    answer_annotation["answer"]
                )
                for span in evidence:
                    all_evidence.add(span)

            evidence_mask = self._get_evidence_mask(list(all_evidence), paragraphs)
            for paragraph_index, (paragraph, is_evidence) in enumerate(zip(paragraphs, evidence_mask)):
                input_ = f"Question: {question} Paragraph: {paragraph} Is the answer to the question in the paragraph? Answer Yes or No."
                target = "Yes" if is_evidence else "No"
                if target == "Yes":
                    self._stats["number of positive targets"] += 1
                elif random.random() <= self._negative_sample_ratio:
                    self._stats["number of negative targets"] += 1
                else:
                    continue
                metadata = {
                    "question_id": question_answer["question_id"],
                    "paper_id": article.get("article_id"),
                    "question": question,
                    "paragraph": paragraph,
                    "paragraph_index": paragraph_index,
                    "query": input_,
                    "target": target,
                    "answer_options": self._answer_options
                }
                yield self.text_to_instance(
                    input_,
                    target,
                    self._answer_options,
                    metadata
                )
                self._stats["number of instances"] += 1

    def _get_paragraphs_from_article(self, article: JsonDict) -> List[str]:
        full_text = article["full_text"]
        paragraphs = []
        for section_info in full_text:
            # TODO (pradeep): It is possible there are other discrepancies between plain text, LaTeX and HTML.
            # Do a thorough investigation and add tests.
            if section_info["section_name"] is not None:
                paragraphs.append(section_info["section_name"])
            for paragraph in section_info["paragraphs"]:
                paragraph_text = paragraph.replace("\n", " ").strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
        return paragraphs

    def _extract_evidence(
        self, answer: List[JsonDict]
    ) -> List[str]:
        evidence_spans = [x.replace("\n", " ").strip() for x in answer["evidence"]]
        evidence_spans = [x for x in evidence_spans if x != ""]
        if not evidence_spans:
            self._stats["answers with no evidence"] += 1
        # TODO (pradeep): Deal with figures and tables.
        if any(["FLOAT SELECTED" in span for span in evidence_spans]):
            # Ignoring question if any of the selected evidence is a table or a figure.
            self._stats["answers with table or figure as evidence"] += 1
        if len(evidence_spans) > 1:
            self._stats["multiple_evidence_spans_count"] += 1

        return evidence_spans

    @staticmethod
    def _get_evidence_mask(evidence: List[str], paragraphs: List[str]) -> List[bool]:
        evidence_mask = []
        for paragraph in paragraphs:
            for evidence_str in evidence:
                if evidence_str in paragraph:
                    evidence_mask.append(True)
                    break
            else:
                evidence_mask.append(False)
        return evidence_mask

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
        if self._return_original_query:
            fields['pretokenized_input'] = input_text

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

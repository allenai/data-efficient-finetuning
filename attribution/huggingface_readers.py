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
    IndexField,
    ListField,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)


# A reader for rank classification tasks.
class HuggingfaceReaderRankClassification(DatasetReader):
    def __init__(
        self,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 512,
        split_name: str = "train",
        val_size: int = 1000,
        use_val_split: bool = True,
        split_mapping: Dict[str, str] = {"train": "train", "validation": "validation"},
        return_original_instance: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        dataset_name, subset_name = self.get_dataset_name()
        data_dir = self.get_dataset_dir()  # for story cloze
        self._dataset_name = dataset_name
        self._subset_name = subset_name
        self.split_name = split_name
        self.return_original_instance = return_original_instance
        original_val_set = datasets.load_dataset(
            dataset_name,
            subset_name,
            split=split_mapping["validation"],
            data_dir=data_dir
        )
        small_val_size = val_size  # I consider under 1000 examples as small
        val_split_size = val_size  # when splitting out val, get 1000 examples
        if use_val_split:
            if split_name == "train":
                if len(original_val_set) >= small_val_size:
                    self._dataset = datasets.load_dataset(
                        dataset_name, subset_name, split=split_mapping["train"], data_dir=data_dir
                    )
                else:
                    # for small val sets, split out val from train and use old val as test
                    # this is because some casehold splits are specially designed, so I want
                    # to keep these as-is (rather than just split the val set in half)
                    self._dataset = datasets.load_dataset(
                        dataset_name, subset_name, split=split_mapping["train"], data_dir=data_dir
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
                        dataset_name, subset_name, split=split_mapping["train"], data_dir=data_dir
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
                        dataset_name, subset_name, split=split_mapping["validation"], data_dir=data_dir
                    )
        else:
            self._dataset = datasets.load_dataset(
                dataset_name, subset_name, split=split_mapping[split_name], data_dir=data_dir
            )
        if split_name == "train":
            self._dataset = self._dataset.shuffle(seed)
        self._transformer_model_name = model_name
        self._tokenizer = PretrainedTransformerTokenizer(model_name)

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}
        self._max_query_length = max_query_length
        self._stats = defaultdict(int)

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        raise NotImplementedError("Implement a dataset-specific version")

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        raise NotImplementedError("Specify ds name for hf")
    
    # we usually dont need this, but story cloze requires it.
    def get_dataset_dir(self) -> Optional[str]:
        return None

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        for sample in self._dataset:
            converted_sample = self.hf_to_instance(sample)
            for inputs, options, idx in converted_sample:
                yield self.text_to_instance(inputs, options, idx)

    def text_to_instance(
        self, input_text: str, answer_options: List[str], correct_answer_idx: int
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

        answer_option_fields = [
            TextField(self._tokenizer.tokenize(option)) for option in answer_options
        ]
        options_list_field = ListField(answer_option_fields)
        fields["answer_options"] = options_list_field
        if self.return_original_instance:
            fields["answer_options_pretokenized"] = answer_options

        fields["correct_answer_index"] = IndexField(
            correct_answer_idx, options_list_field
        )
        if self.return_original_instance:
            fields["correct_answer_index_value"] = correct_answer_idx

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["prompt_and_input"].token_indexers = self._token_indexers
        for field in instance.fields["answer_options"].field_list:
            field.token_indexers = self._token_indexers


@DatasetReader.register("casehold_reader")
class CaseHOLDReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "lex_glue", "case_hold"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # following setup similar to unifiedqa
        input = f"What is the correct holding statement for the following text?\nText: {instance['context']} \n(A): {instance['endings'][0]}\n(B): {instance['endings'][1]}\n(C): {instance['endings'][2]}\n(D): {instance['endings'][3]}\n(E): {instance['endings'][4]}"
        return [[input, instance["endings"], instance["label"]]]


@DatasetReader.register("unfair_tos_reader")
class UnfairTOSReader(HuggingfaceReaderRankClassification):
    labels_to_terms = {
        0: "Limitation of liability",
        1: "Unilateral termination",
        2: "Unilateral change",
        3: "Content removal",
        4: "Contract by using",
        5: "Choice of law",
        6: "Jurisdiction",
        7: "Arbitration",
    }

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "lex_glue", "unfair_tos"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # we convert this into 8 instances asking if label applies
        samples = []
        answer_options = ["yes", "no"]
        for label, term in self.labels_to_terms.items():
            input = f'Is there an instance of {term} in the following text?. Answer yes or no.\nText: {instance["text"]}'
            output = "yes" if label in instance["labels"] else "no"
            samples.append((input, answer_options, answer_options.index(output)))
        return samples


@DatasetReader.register("eurlex_reader")
class EurlexReader(HuggingfaceReaderRankClassification):
    def __init__(
        self,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 512,
        split_name: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(model_name, max_query_length, split_name, **kwargs)
        self._labels = self._dataset.features["labels"].feature.names
        self._concept_dict = json.load(open("data/eurovoc_descriptors.json", "r"))
        self._r = random.Random(42)
        self._split = split_name

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "lex_glue", "eurlex"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # convert into 127 instances asking yes/no.
        samples = []
        answer_options = ["yes", "no"]
        # pick random labels + the true labels, adding up to 10 - train only
        if self._split == "train":
            label_sel = [l for l in instance["labels"]]
            options = [l for l in range(len(self._labels)) if l not in label_sel]
            while len(label_sel) < 5:
                label_sel.append(self._r.choice(options))
            self._r.shuffle(label_sel)
            label_sel = [
                self._dataset.features["labels"].feature.int2str(l) for l in label_sel
            ]
        else:
            # validation, only check the labels we know should be on the thing.
            label_sel = [
                self._dataset.features["labels"].feature.int2str(l)
                for l in instance["labels"]
            ]
        for label in label_sel:
            concept_name = self._concept_dict[label]["en"]
            input = f"Does the following text involve {concept_name}? Answer yes or no.\nText: {instance['text']}."
            output = (
                "yes"
                if self._dataset.features["labels"].feature.str2int(label)
                in instance["labels"]
                else "no"
            )
            samples.append((input, answer_options, answer_options.index(output)))
        return samples


@DatasetReader.register("ledgar_reader")
class LedgarReader(HuggingfaceReaderRankClassification):
    def __init__(
        self,
        model_name: str = "google/t5-small-lm-adapt",
        max_query_length: int = 512,
        split_name: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(model_name, max_query_length, split_name, **kwargs)
        self.labels = self._dataset.features["label"]

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "lex_glue", "ledgar"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # rank classification seems fine here
        input = f"What is the main topic of the following contract provision?\nContract: {instance['text']}"
        return [[input, self.labels.names, instance["label"]]]


@DatasetReader.register("sciq_reader")
class SciQReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "sciq", None

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # sciq in training, so using 'multiple choice' prompt.
        answers = [
            instance["distractor1"],
            instance["distractor2"],
            instance["distractor3"],
            instance["correct_answer"],
        ]
        # shuffle answers so model cant learn!
        random.Random(42).shuffle(answers)
        correct_answer_idx = answers.index(instance["correct_answer"])
        input = f"Answer the following question given this paragraph:\n{instance['support']}\nQ: {instance['question']}\nChoices:\n-{answers[0]}\n-{answers[1]}\n-{answers[2]}\n-{answers[3]}"
        return [[input, answers, correct_answer_idx]]


@DatasetReader.register("rte_reader")
class RTEReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "super_glue", "rte"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using highest-avg prompt for rte from my other project.
        input = f"{instance['premise']}\n Question: Does this imply that \"{instance['hypothesis']}\"? Yes or no?"
        answers = ["yes", "no"]
        # 0 = entail, 1 = not entail
        correct_answer = 0 if instance["label"] == 0 else 1
        return [[input, answers, correct_answer]]


# CB
@DatasetReader.register("cb_reader")
class CBReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "super_glue", "cb"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using gpt-3 style prompt
        input = f"{instance['premise']}\nQuestion: {instance['hypothesis']} True, False, or Neither?"
        # 0 = entail, 1 = contradict, 2 = neutral
        answers = ["true", "false", "neither"]
        correct_answer = int(instance["label"])
        return [[input, answers, correct_answer]]


# HellaSwag
@DatasetReader.register("hellaswag_reader")
class HellaSwagReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "hellaswag", "None"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        input = f"Complete the description with an appropriate ending:\nFirst, {instance['ctx_a']} Then, {instance['ctx_b']} ...\n(a) {instance['endings'][0]}\n(b) {instance['endings'][1]}\n(c) {instance['endings'][2]}\n(d) {instance['endings'][3]}"
        answers = instance['endings']
        correct_answer = int(instance['label'])
        return [[input, answers, correct_answer]]


# StoryCloze
## NB: requires downloading separately as it requires agreeing to a thing
# following ia3, we use the val as train and test as val.
@DatasetReader.register("story_cloze_reader")
class StoryClozeReader(HuggingfaceReaderRankClassification):
    def __init__(
        self,
        split_mapping={"train": "validation", "validation": "test"},
        **kwargs,
    ) -> None:
        super().__init__(split_mapping=split_mapping, **kwargs)
        
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "story_cloze", "2016"

    # we usually dont need this, but story cloze requires it.
    # TODO: replace with the location of downloaded story cloze data.
    def get_dataset_dir(self) -> Optional[str]:
        return ''

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        input = f"{instance['input_sentence_1']} {instance['input_sentence_2']} {instance['input_sentence_3']} {instance['input_sentence_4']} What is a possible continuation for the story given the following options ?\n- {instance['sentence_quiz1']}\n- {instance['sentence_quiz2']}"
        answers = [instance['sentence_quiz1'], instance['sentence_quiz2']]
        # answers given are 1-indexed
        correct_answer = instance['answer_right_ending'] - 1
        return [[input, answers, correct_answer]]

# WinoGrande
@DatasetReader.register("winogrande_reader")
class WinoGrandeReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "winogrande", "winogrande_xl"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using underscore refer to
        input = f"{instance['sentence']}\nWhat does the _ in the above sentence refer to? {instance['option1']} or {instance['option2']}?"
        # 0 = false, 1 = true
        answers = ['1', '2']
        assert instance['answer'] in answers
        correct_answer = answers.index(instance['answer'])
        return [[input, answers, correct_answer]]

# WSC
@DatasetReader.register("wsc_reader")
class WSCReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "super_glue", "wsc"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using gpt3 style
        input = f"Passage: {instance['text']} \nQuestion: In the passage above, does the pronoun \"{instance['span2_text']}\" refer to {instance['span1_text']}?\nAnswer:"
        # 0 = false, 1 = true
        answers = ['False', 'True']
        correct_answer = int(instance["label"])
        return [[input, answers, correct_answer]]

# COPA
@DatasetReader.register("copa_reader")
class COPAReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "super_glue", "copa"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using using 'plausible alternatives' prompt
        input = f"{instance['premise']}  As a consequence... \nHelp me pick the more plausible option:\n- {instance['choice1']}\n- {instance['choice2']}"
        # 0 = choice1, 1 = choice2
        answers = [instance['choice1'], instance['choice2']]
        correct_answer = int(instance["label"])
        return [[input, answers, correct_answer]]

# WiC
@DatasetReader.register("wic_reader")
class WiCReader(HuggingfaceReaderRankClassification):
    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "super_glue", "wic"

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using gpt-3 style prompt
        input = f"{instance['sentence1']}\n{instance['sentence2']}\nQuestion: Is the word '{instance['word']}' used in the same sense in the two sentences above? Yes, No?"
        # 0 = false, 1 = true
        answers = ["no", "yes"]
        correct_answer = int(instance["label"])
        return [[input, answers, correct_answer]]

## anli is handled specially as its splits have weird names.

# ANLI R1
@DatasetReader.register("anli_r1_reader")
class ANLIR1Reader(HuggingfaceReaderRankClassification):
    def __init__(
        self,
        split_mapping={"train": "train_r1", "validation": "dev_r1"},
        **kwargs,
    ) -> None:
        super().__init__(split_mapping=split_mapping, **kwargs)

    def get_dataset_name(self) -> Tuple[str, Optional[str]]:
        return "anli", None

    def hf_to_instance(self, instance) -> Tuple[str, str]:
        # using gpt-3 style prompt
        input = f"{instance['premise']}\nQuestion: {instance['hypothesis']} True, False, or Neither?"
        # 0 = entail, 1 = neutral, 2 = contradiction
        answers = ["true", "neither", 'false']
        correct_answer = instance['label']
        return [[input, answers, correct_answer]]


# the other anlis are identical, just diff splits :)
# ANLI R2
@DatasetReader.register("anli_r2_reader")
class ANLIR2Reader(ANLIR1Reader):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            split_mapping={"train": "train_r2", "validation": "dev_r2"},
            **kwargs,
        )

# ANLI R3
@DatasetReader.register("anli_r3_reader")
class ANLIR3Reader(ANLIR1Reader):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            split_mapping={"train": "train_r3", "validation": "dev_r3"},
            **kwargs,
        )


# to generate files containing training data
# easy to repurpose to generate whatever you want.
if __name__ == '__main__':
    import json
    data_classes = [StoryClozeReader, RTEReader, CBReader, HellaSwagReader, COPAReader, WinoGrandeReader, WSCReader, WiCReader, ANLIR1Reader, ANLIR2Reader, ANLIR3Reader]
    data_names = ['story_cloze', 'rte', 'cb', 'hellaswag', 'copa', 'winogrande', 'wsc', 'wic', 'anli_r1', 'anli_r2', 'anli_r3']
    for cls, name in zip(data_classes, data_names):
        print(name)
        reader = cls(
            model_name="google/t5-large-lm-adapt",
            max_query_length=512,
            split_name='train',
            val_size=1e100,
            use_val_split=False
        )
        lines = []
        for sample in reader.read('dummy'):
            lines.append(json.dumps({
                "prompt_and_input": sample['prompt_and_input_pretokenized'],
                "answer_options": sample['answer_options_pretokenized'],
                'correct_answer_index': sample['correct_answer_index_value']
            }))
        with open(f'retrieve_data/{name}_val_data.jsonl', 'w') as f:
            f.write('\n'.join(lines))
    
import csv
import json
import logging
import os
from abc import ABC

import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from utils.file_utils import read_lines, read_jsonl_lines

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MultiFormatDataProcessor(DataProcessor, ABC):

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                records.append(obj)
        return records


class McExample(object):
    def get_option_segments(self):
        raise NotImplementedError


class MultipleChoiceFeatures(object):
    def __init__(self,
                 example_id,
                 option_features,
                 label=None):
        self.example_id = example_id
        self.option_features = self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in option_features
        ]
        if label is not None:
            self.label = int(label) - 1
        else:
            self.label = None


class AnliExample(object):
    def __init__(self,
                 example_id,
                 beginning: str,
                 middle_options: list,
                 ending: str,
                 label=None):
        self.example_id = example_id
        self.beginning = beginning
        self.ending = ending
        self.middle_options = middle_options
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        lines = [
            "example_id:\t{}".format(self.example_id),
            "beginning:\t{}".format(self.beginning)
        ]
        for idx, option in enumerate(self.middle_options):
            lines.append("option{}:\t{}".format(idx, option))

        lines.append("ending:\t{}".format(self.ending))

        if self.label is not None:
            lines.append("label:\t{}".format(self.label))
        return ", ".join(lines)

    def to_json(self):
        return {
            "story_id": self.example_id,
            "obs1": self.beginning,
            "obs2": self.ending,
            "hyp1": self.middle_options[0],
            "hyp2": self.middle_options[1],
            "label": self.label
        }

    def to_middles_only_format(self):
        return [
            {
                "segment1": self.middle_options[0]
            },
            {
                "segment1": self.middle_options[1]
            }
        ]

    def to_middles_sequence_format(self):
        return [
            {
                "segment1": self.middle_options[0],
                "segment2": self.middle_options[1]
            }
        ]

    def to_bm_e_format(self):
        return [{
            "segment1": ' '.join([self.beginning, option]),
            "segment2": self.ending
        } for option in self.middle_options]

    def to_b_me_format(self):
        return [
            {
                "segment1": self.beginning,
                "segment2": ' '.join([self.middle_options[0], self.ending])
            },
            {
                "segment1": self.beginning,
                "segment2": ' '.join([self.middle_options[1], self.ending])
            }
        ]

    def to_b2m_m2e_format(self):
        return [
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[0]
            },
            {
                "segment1": self.middle_options[0],
                "segment2": self.ending
            },
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[1]
            },
            {
                "segment1": self.middle_options[1],
                "segment2": self.ending
            }
        ]

    def to_b2m_bm2e_format(self):
        return [
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[0]
            },
            {
                "segment1": self.beginning + ' ' + self.middle_options[0],
                "segment2": self.ending
            },
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[1]
            },
            {
                "segment1": self.beginning + ' ' + self.middle_options[1],
                "segment2": self.ending
            }
        ]

    def to_b_m_e_format(self):
        return [
            {
                "segment1": self.beginning,
            },
            {
                "segment1": self.middle_options[0]
            },
            {
                "segment1": self.ending
            },
            {
                "segment1": self.beginning,
            },
            {
                "segment1": self.middle_options[1]
            },
            {
                "segment1": self.ending
            }
        ]

    def to_b2m_m2e_m1_m2_format(self):
        return self.to_b2m_m2e_format() \
               + [
                   {
                       "segment1": self.middle_options[0],
                   },
                   {
                       "segment1": self.middle_options[1],
                   }
               ]

    def to_b_m1_m2_e_format(self):
        return [
            {
                "segment1": self.beginning
            },
            {
                "segment1": self.middle_options[0]
            },
            {
                "segment1": self.middle_options[1]
            },
            {
                "segment1": self.ending
            }
        ]

    def to_b2m_format(self):
        return [
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[0]
            },
            {
                "segment1": self.beginning,
                "segment2": self.middle_options[1]
            }
        ]

    def to_m2e_format(self):
        return [
            {
                "segment1": self.middle_options[0],
                "segment2": self.ending
            },
            {
                "segment1": self.middle_options[1],
                "segment2": self.ending
            }
        ]

    def get_option_segments(self):
        return self.to_bm_e_format()


class AnliProcessor(MultiFormatDataProcessor):
    """Processor for the ANLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "train.jsonl"),
            os.path.join(data_dir, "train-labels.lst"),
            "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "dev.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "dev.jsonl"),
            os.path.join(data_dir, "dev-labels.lst"),
            "train"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "test.jsonl"),
            os.path.join(data_dir, "test-labels.lst"),
            "train"
        )

    def get_examples_from_file(self, input_file, labels_file=None, split="predict"):
        if labels_file is not None:
            return self._create_examples(
                self._read_jsonl(input_file),
                read_lines(labels_file),
                split
            )
        else:
            return self._create_examples(
                self._read_jsonl(input_file)
            )

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, records, labels=None, set_type="predict"):
        """Creates examples for the training and dev sets."""
        examples = []

        if labels is None:
            labels = [None] * len(records)

        for (i, (record, label)) in enumerate(zip(records, labels)):
            guid = "%s" % (record['story_id'])

            beginning = record['obs1']
            ending = record['obs2']

            option1 = record['hyp1']
            option2 = record['hyp2']

            examples.append(
                AnliExample(example_id=guid,
                            beginning=beginning,
                            middle_options=[option1, option2],
                            ending=ending,
                            label=label
                            )
            )
        return examples

    def label_field(self):
        return "label"


class AnliMultiDistractorProcessor(MultiFormatDataProcessor):
    """Multiple Distractor during training for the ANLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                                        "anli-train-multi-distractors.jsonl")))
        return self.get_multi_distractor_examples_from_file(
            os.path.join(data_dir, "anli-train-multi-distractors.jsonl")
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "dev.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "dev.jsonl"),
            os.path.join(data_dir, "dev-labels.lst"),
            "train"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "test.jsonl"),
            os.path.join(data_dir, "test-labels.lst"),
            "test"
        )

    def get_multi_distractor_examples_from_file(self, input_file):
        records = read_jsonl_lines(input_file)
        labels = [r['label'] for r in records]
        return self._create_examples(records, labels)

    def get_examples_from_file(self, input_file, labels_file=None, split="predict"):
        if labels_file is not None:
            return self._create_examples(
                self._read_jsonl(input_file),
                read_lines(labels_file),
                split
            )
        else:
            return self._create_examples(
                self._read_jsonl(input_file)
            )

    def get_labels(self):
        """See base class."""
        return [str(item + 1) for item in range(10)]

    def _create_examples(self, records, labels=None, set_type="predict"):
        """Creates examples for the training and dev sets."""
        examples = []

        if labels is None:
            labels = [None] * len(records)

        for (i, (record, label)) in enumerate(zip(records, labels)):
            guid = "%s" % (record['story_id'])

            beginning = record['obs1']
            ending = record['obs2']

            if 'choices' in record:
                options = record['choices']
                label = int(label) + 1
            else:
                options = [record['hyp1'], record['hyp2']]

            examples.append(
                AnliExample(example_id=guid,
                            beginning=beginning,
                            middle_options=options,
                            ending=ending,
                            label=label
                            )
            )
        return examples

    @staticmethod
    def label_field():
        return "label"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_multiple_choice_examples_to_features(examples: list,
                                                 tokenizer: BertTokenizer,
                                                 max_seq_length: int,
                                                 is_training: bool,
                                                 verbose: bool = False):
    features = []
    for idx, example in enumerate(examples):
        option_features = []
        for option in example.get_option_segments():
            context_tokens = tokenizer.tokenize(option['segment1'])
            if "segment2" in option:
                option_tokens = tokenizer.tokenize(option["segment2"])
                _truncate_seq_pair(context_tokens, option_tokens, max_seq_length - 3)
                tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + option_tokens + ["[SEP]"]
                segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(option_tokens) + 1)
            else:
                context_tokens = context_tokens[0:(max_seq_length - 2)]
                tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
                segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            option_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        if idx < 5 and verbose:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.example_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(
                    option_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            MultipleChoiceFeatures(
                example_id=example.example_id,
                option_features=option_features,
                label=label
            )
        )

    return features


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def mc_examples_to_data_loader(examples: list,
                               tokenizer,
                               max_seq_length,
                               is_train,
                               batch_size,
                               is_predict=False,
                               verbose: bool = False):
    features = convert_multiple_choice_examples_to_features(
        examples, tokenizer, max_seq_length, is_train, verbose
    )
    if verbose:
        logger.info("  Num examples = %d", len(examples))

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    if not is_predict:
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

import argparse
import json

from transformers import BertTokenizer

import tqdm

from anli.data_processors import AnliProcessor


def data_processor_by_name(task_name):
    if task_name == "anli":
        return AnliProcessor()


def main(args):
    data_dir = args.data_dir
    bert_model = args.bert_model
    task_name = args.task_name
    threshold = args.threshold

    data_processor = data_processor_by_name(task_name)

    all_examples = data_processor.get_train_examples(data_dir) + \
                   data_processor.get_dev_examples(data_dir)

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    segment_1_lengths = []
    segment_2_lengths = []
    for example in tqdm.tqdm(all_examples):
        for option in example.get_option_segments():
            context_tokens = tokenizer.tokenize(option['segment1'])
            segment_1_lengths.append(len(context_tokens))
            if "segment2" in option:
                option_tokens = tokenizer.tokenize(option["segment2"])
                segment_2_lengths.append(len(option_tokens))

    m1 = max(segment_1_lengths)
    m2 = 0
    print("Max Segment 1: {}".format(m1))
    if len(segment_2_lengths) > 0:
        m2 = max(segment_2_lengths)
        print("Max Segment 2: {}".format(m2))

        s = [x + y for x, y in zip(segment_1_lengths, segment_2_lengths)]
        print("Set max ctx >= {}".format(max(s) + 3))

        num = sum([i > threshold for i in s])
        print("No. more than {} = {}".format(threshold, num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune BERT model and save')

    # Required Parameters
    parser.add_argument('--data_dir', type=str, help='Location of data', default=None)
    parser.add_argument('--bert_model', type=str, help='Bert model', default="bert-base-uncased")
    parser.add_argument('--task_name', type=str, help='Bert model', default="anli")
    parser.add_argument('--threshold', type=int, help='Threshold for truncation', default=256)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
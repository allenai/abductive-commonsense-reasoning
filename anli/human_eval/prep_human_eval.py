import argparse
import json
import pandas as pd
from pandas import DataFrame

from utils.file_utils import read_jsonl_lines, read_lines, write_items


def main(args):
    dev_file = args.dev_file
    dev_labels_file = args.dev_labels_file
    output_file = args.output_file

    records = read_jsonl_lines(dev_file)
    labels = read_lines(dev_labels_file)

    for r, l in zip(records, labels):
        r['label'] = l

    write_items([json.dumps(r) for r in records], output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to compute corpus satistics')

    # Required Parameters
    parser.add_argument('--dev_file', type=str, help='Location of dev data', default=None)
    parser.add_argument('--dev_labels_file', type=str, help='Location of dev labels ', default=None)
    parser.add_argument('--output_file', type=str, help='Location of output file ', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
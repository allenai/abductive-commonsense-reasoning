import argparse
import json

from utils.file_utils import TsvIO
from utils.file_utils import read_lines
import pandas as pd


def jsonl_to_tsv(jsonl_file, output_file, sep):

    records = [json.loads(line) for line in read_lines(jsonl_file)]
    if sep == "\t":
        TsvIO.write(records, filename=output_file, schema=records[0].keys(), sep=sep)
    else:
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='tsv_to_jsonl.py',
        usage='%(prog)s tsv_file',
        description='Identify seed set of entities filtered by Google N-Gram counts'
    )
    parser.add_argument('--jsonl_file', type=str,
                        help='Source of seed entities',
                        default='conceptnet')
    parser.add_argument('--output_file', type=str,
                        help='Location of output file')
    parser.add_argument('--sep', type=str,
                        help='Location of output file',
                        default='\t')
    args = parser.parse_args()

    # Run seed selection if args valid
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    jsonl_to_tsv(args.jsonl_file, args.output_file, args.sep)

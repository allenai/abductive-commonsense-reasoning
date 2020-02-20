import argparse
import json
import numpy as np

from utils.file_utils import read_jsonl_lines, read_lines


def _key(r):
    return r['obs1'] + '||' + r['obs2']


def correct_middle(r):
    return r['hyp' + r['label']]


def incorrect_middle(r):
    if r['label'] == "1":
        return r['hyp2']
    else:
        return r['hyp1']


def mean_word_lens(lst):
    return round(np.mean([len(s.split()) for s in lst]),2)


def main(args):
    input_file = args.input_file
    labels_file = args.label_file

    stories = read_jsonl_lines(input_file)
    labels = read_lines(labels_file)

    all_begins = []
    all_endings = []

    stories_by_key = {}
    for s, label in zip(stories, labels):
        s['label'] = label

        key = _key(s)
        if key not in stories_by_key:
            stories_by_key[key] = []
        stories_by_key[key].append(s)

        all_begins.append(s['obs1'])
        all_endings.append(s['obs2'])

    num_correct_middles_per_story = []
    num_incorrect_middles_per_story = []
    all_correct_middles = []
    all_incorrect_middles = []

    all_begins = list(set(all_begins))
    all_endings = list(set(all_endings))

    for k, stories in stories_by_key.items():
        num_correct_middles_per_story.append(len(set([correct_middle(r) for r in stories])))
        num_incorrect_middles_per_story.append(len(set([incorrect_middle(r) for r in stories])))

        all_correct_middles.extend(list(set([correct_middle(r) for r in stories])))
        all_incorrect_middles.extend(list(set([incorrect_middle(r) for r in stories])))

    print("No. of train stories: {}".format(len(stories_by_key)))
    print("Mean of no. of correct middles = {}".format(round(np.mean(num_correct_middles_per_story), 2)))
    print("Mean of no. of incorrect middles = {}".format(round(np.mean(num_incorrect_middles_per_story), 2)))

    print("Mean of no. of words in correct middles = {}".format(mean_word_lens(all_correct_middles)))
    print("Mean of no. of words in incorrect middles = {}".format(mean_word_lens(all_incorrect_middles)))

    print("No. correct middles = {}".format(len(all_correct_middles)))
    print("No. incorrect middles = {}".format(len(all_incorrect_middles)))

    print("Mean of no. of words in Begins = {}".format(mean_word_lens(all_begins)))
    print("Mean of no. of words in Endings = {}".format(mean_word_lens(all_endings)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to compute corpus satistics')

    # Required Parameters
    parser.add_argument('--input_file', type=str, help='Location of data', default=None)
    parser.add_argument('--label_file', type=str, help='Location of data', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
import argparse
import json

from utils.file_utils import read_jsonl_lines, write_items
from collections import Counter
import hashlib


def _hash(w):
    return hashlib.sha1(w.encode()).hexdigest()


def main(args):
    input_file = args.input_file
    output_file = args.output_file

    records = read_jsonl_lines(input_file)

    per_story_votes = {}
    per_story_workers = {}
    per_story_per_vote_worktime = {}
    for r in records:
        if r['Input.story_id'] not in per_story_votes:
            per_story_votes[r['Input.story_id']] = []
            per_story_workers[r['Input.story_id']] = []
            per_story_per_vote_worktime[r['Input.story_id']] = []
        per_story_votes[r['Input.story_id']].append(r['Answer.Answer_radios'])
        per_story_workers[r['Input.story_id']].append(_hash(r['WorkerId']))
        per_story_per_vote_worktime[r['Input.story_id']].append(r['WorkTimeInSeconds'])

    stories = []
    correct = 0
    done = set()
    for r in records:
        if r['Input.story_id'] in done:
            continue
        done.add(r['Input.story_id'])

        assert len(per_story_votes[r['Input.story_id']]) == 3

        majority_vote = Counter(per_story_votes[r['Input.story_id']]).most_common(1)[0][0]
        stories.append({
            'story_id': r['Input.story_id'],
            'obs1': r['Input.obs1'],
            'obs2': r['Input.obs2'],
            'hyp1': r['Input.hyp1'],
            'hyp2': r['Input.hyp2'],
            'label': r['Input.label'],
            'votes': per_story_votes[r['Input.story_id']],
            'majority_vote': majority_vote,
            'workers': per_story_workers[r['Input.story_id']],
            'worktime': per_story_per_vote_worktime[r['Input.story_id']]
        })

        if majority_vote == r['Input.label']:
            correct += 1

    print("Human performance = {}".format(correct / len(stories)))
    print("No. of storeies = {}".format(len(stories)))

    write_items([json.dumps(r) for r in stories], output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to compute corpus satistics')

    # Required Parameters
    parser.add_argument('--input_file', type=str, help='Location of dev data', default=None)
    parser.add_argument('--output_file', type=str, help='Location of dev data', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
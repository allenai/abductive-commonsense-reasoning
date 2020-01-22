import argparse
import json
import torch

from transformers import BertTokenizer, BertConfig

from anli.data_processors import AnliExample, mc_examples_to_data_loader
from anli.run_anli import get_data_processor, model_choice_map
import numpy as np


def load_anli_model(model_name, saved_model_dir, device):
    data_processor = get_data_processor("anli")
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Pretrained Model
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=len(data_processor.get_labels()),
        finetuning_task="anli"
    )

    model = model_choice_map['BertForMultipleChoice'].from_pretrained(
        saved_model_dir,
        from_tf=bool('.ckpt' in model_name),
        config=config
    )

    model.to(device)
    model.eval()

    return data_processor, tokenizer, model


def _predict(data_processor, tokenizer, model, obs1, obs2, hyp1, hyp2, device):
    instance = AnliExample(example_id="demo-1",
                           beginning=obs1,
                           middle_options=[hyp1, hyp2],
                           ending=obs2,
                           label=None
                           )
    data_loader = mc_examples_to_data_loader([instance], tokenizer, 68, False, 1, is_predict=True, verbose=False)
    for input_ids, input_mask, segment_ids, label_ids in data_loader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        logits = model_output[1]

        logits = logits.detach().cpu().numpy()

        answer = np.argmax(logits, axis=1).tolist()
    return answer


def main(args, interactive=False):
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    data_processor, tokenizer, model = load_anli_model(args.model_name_or_path, device)

    if interactive:
        while True:
            obs1 = input("Observation 1 >>> ")
            obs2 = input("Observation 2 >>> ")
            hyp1 = input("Hypothesis 1 >>> ")
            hyp2 = input("Hypothesis 2 >>> ")

            prediction = _predict(data_processor, tokenizer, model, obs1, obs2, hyp1, hyp2, device)

            if prediction == 0:
                print("[Answer] Hyptothesis 1: {}".format(hyp1))
            else:
                print("[Answer] Hyptothesis 2: {}".format(hyp2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo for a finetuned ANLI model.')

    # Required Parameters
    parser.add_argument('--model_name',
                        type=str,
                        help="Bert pre-trained model selected for finetuning.",
                        default="bert-large-uncased")
    parser.add_argument('--saved_model_dir',
                        type=str,
                        help="Saved finetuned model dir.",
                        default=None,
                        required=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)

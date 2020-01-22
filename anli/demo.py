import argparse
import json
import torch

from transformers import BertTokenizer, BertConfig

from anli.data_processors import AnliExample, mc_examples_to_data_loader
from anli.run_anli import get_data_processor, model_choice_map
import numpy as np


def load_anli_model(model_name_or_path, device):
    data_processor = get_data_processor("anli")
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)



    # Pretrained Model
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(data_processor.get_labels()),
        finetuning_task="anli"
    )

    model = model_choice_map['BertForMultipleChoice'].from_pretrained(
        model_name_or_path,
        from_tf=bool('.ckpt' in model_name_or_path),
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


def main(args, model_path, interactive=False):
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
    parser.add_argument('--data_dir', type=str, help='Location of data', default=None)
    parser.add_argument('--task_name', type=str, help='Task Name. Currently supported: anli / '
                                                      'wsc', default=None)
    parser.add_argument('--model_name_or_path',
                        type=str,
                        help="Bert pre-trained model selected for finetuned",
                        default=None)
    parser.add_argument('--output_dir',
                        type=str,
                        help="Output directory to save model",
                        default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--finetuning_model', type=str, default='BertForMultipleChoice')
    parser.add_argument('--eval_split', type=str, default="dev")
    parser.add_argument('--run_on_test', action='store_true')

    parser.add_argument('--input_file', action='store_true')
    parser.add_argument('--predict_input_file', default=None)
    parser.add_argument('--predict_output_file', default=None)
    parser.add_argument('--metrics_out_file', default="metrics.json")

    # Hyperparams
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--epochs', type=int, help="Num epochs", default=3)
    parser.add_argument('--training_data_fraction', type=float, default=1.0)

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)

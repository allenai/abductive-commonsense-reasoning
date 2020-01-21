import argparse
import json
import logging
import math
import os
import random

import numpy as np
import torch
from pytorch_transformers import BertTokenizer, PYTORCH_PRETRAINED_BERT_CACHE, \
    BertForMultipleChoice, BertConfig, BertModel
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from pytorch_transformers import AdamW, WarmupLinearSchedule

from anli.data_processors import AnliProcessor, AnliMultiDistractorProcessor, mc_examples_to_data_loader
from utils.file_utils import write_items
from torch import nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_choice = BertForMultipleChoice

model_choice_map = {
    'BertForMultipleChoice': BertForMultipleChoice,
}


def get_data_processor(task_name):
    if task_name == "anli":
        return AnliProcessor()
    elif task_name == "anli_md":
        return AnliMultiDistractorProcessor()
    else:
        raise Exception("Invalid task")


def _model_name(dir_name):
    return os.path.join(dir_name, "pytorch_model.bin")


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train(data_dir, output_dir, data_processor, model_name_or_path, lr, batch_size, epochs,
          finetuning_model,
          max_seq_length, warmup_proportion, debug=False, tune_bert=True, gpu_id=0, tb_dir=None,
          debug_samples=20, training_data_fraction=1.0, config_name=None):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir)

    writer = None
    if tb_dir is not None:
        writer = SummaryWriter(tb_dir)

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    train_examples = data_processor.get_train_examples(data_dir)

    if training_data_fraction < 1.0:
        num_train_examples = int(len(train_examples) * training_data_fraction)
        train_examples = random.sample(train_examples, num_train_examples)

    if debug:
        logging.info("*****[DEBUG MODE]*****")
        train_examples = train_examples[:debug_samples]
    num_train_steps = int(
        len(train_examples) / batch_size * epochs
    )

    # Pretrained Model
    config = BertConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=len(data_processor.get_labels()),
        finetuning_task="anli"
    )
    model = model_choice_map[finetuning_model].from_pretrained(model_name_or_path,
                                                               from_tf=bool('.ckpt' in args.model_name_or_path),
                                                               config=config
                                                               )

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if writer:
        params_to_log_on_tb = [(k, v) for k, v in model.named_parameters() if
                               not k.startswith("bert")]

    t_total = num_train_steps

    train_dataloader = mc_examples_to_data_loader(train_examples, tokenizer, max_seq_length, True,
                                                  batch_size, verbose=True)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=math.floor(warmup_proportion * t_total),
                                     t_total=t_total)

    global_step = 0

    logging.info("\n\n\n\n****** TRAINABLE PARAMETERS = {} ******** \n\n\n\n"
                 .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for epoch_num in trange(int(epochs), desc="Epoch"):

        model.train()

        assert model.training

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        batch_tqdm = tqdm(train_dataloader)

        current_correct = 0

        for step, batch in enumerate(batch_tqdm):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            loss = model_output[0]
            logits = model_output[1]

            current_correct += num_correct(logits.detach().cpu().numpy(),
                                           label_ids.to('cpu').numpy())

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()

            if (step + 1) % 1 == 0:  # I don't know why this is here !!!
                # modify learning rate with special warm up BERT uses
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if writer:
                    writer.add_scalar("loss", tr_loss / nb_tr_steps, global_step)
                    lrs = scheduler.get_lr()
                    writer.add_scalar("lr_pg_1", lrs[0], global_step)
                    writer.add_scalar("lr_pg_2", lrs[1], global_step)
                    for n, p in params_to_log_on_tb:
                        writer.add_histogram(n, p.clone().cpu().data.numpy(), global_step)
                    writer.add_histogram("model_logits", logits.clone().cpu().data.numpy(),
                                         global_step)

            batch_tqdm.set_description(
                "Loss: {}; Iteration".format(round(tr_loss / nb_tr_steps, 3)))

        tr_acc = current_correct / nb_tr_examples
        # Call evaluate at the end of each epoch
        result = evaluate(data_dir=data_dir,
                          output_dir=output_dir,
                          data_processor=data_processor,
                          model_name_or_path=model_name_or_path,
                          finetuning_model=finetuning_model,
                          max_seq_length=max_seq_length,
                          batch_size=batch_size,
                          debug=debug,
                          gpu_id=gpu_id,
                          model=model,
                          tokenizer=tokenizer,
                          verbose=False,
                          debug_samples=debug_samples
                          )
        logging.info("****** EPOCH {} ******\n\n\n".format(epoch_num))
        logging.info("Training Loss: {}".format(round(tr_loss / nb_tr_steps, 3)))
        logging.info("Training Accuracy: {}".format(round(tr_acc, 3)))
        logging.info("Validation Loss : {}".format(round(result['dev_eval_loss'], 3)))
        logging.info("Validation Accuracy : {}".format(round(result['dev_eval_accuracy'], 3)))
        logging.info("******")
        if writer:
            writer.add_scalar("dev_val_loss", result['dev_eval_loss'], global_step)
            writer.add_scalar("dev_val_accuracy", result['dev_eval_accuracy'], global_step)
            writer.add_scalar("dev_accuracy", tr_acc, global_step)

    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Only save the model it-self
    output_model_file = _model_name(output_dir)
    torch.save(model_to_save.state_dict(), output_model_file)
    logging.info("Training Done. Saved model to: {}".format(output_model_file))
    return output_model_file


def evaluate(data_dir, output_dir, data_processor, model_name_or_path, finetuning_model, max_seq_length,
             batch_size,
             debug=False, gpu_id=0, model=None, tokenizer=None, verbose=False, debug_samples=20,
             eval_split="dev", config_name=None, metrics_out_file="metrics.json"):
    if debug:
        logging.info("*****[DEBUG MODE]*****")
        eval_examples = data_processor.get_train_examples(data_dir)[:debug_samples]
    else:
        if eval_split == "dev":
            eval_examples = data_processor.get_dev_examples(data_dir)
        elif eval_split == "test":
            eval_examples = data_processor.get_test_examples(data_dir)

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    eval_dataloader = mc_examples_to_data_loader(examples=eval_examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=max_seq_length,
                                                 is_train=False,
                                                 batch_size=batch_size,
                                                 verbose=verbose
                                                 )

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    # Load a trained model that you have fine-tuned
    if model is None:
        config = BertConfig.from_pretrained(
            config_name if config_name else model_name_or_path,
            num_labels=len(data_processor.get_labels()),
            finetuning_task="anli"
        )
        model = model_choice_map[finetuning_model].from_pretrained(output_dir,
                                                                   from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                   config=config
                                                                   )
        model.to(device)

    model.eval()

    assert not model.training

    eval_loss, eval_correct = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    eval_predictions = []
    eval_logits = []
    eval_pred_probs = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            tmp_eval_loss = model_output[0]
            logits = model_output[1]

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_correct = num_correct(logits, label_ids)

        eval_predictions.extend(np.argmax(logits, axis=1).tolist())
        eval_logits.extend(logits.tolist())
        eval_pred_probs.extend([_compute_softmax(list(l)) for l in logits])

        eval_loss += tmp_eval_loss.item()  # No need to compute mean again. CrossEntropyLoss does that by default.
        nb_eval_steps += 1

        eval_correct += tmp_eval_correct
        nb_eval_examples += input_ids.size(0)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_correct / nb_eval_examples

    result = {}
    if os.path.exists(metrics_out_file):
        with open(metrics_out_file) as f:
            existing_results = json.loads(f.read())
        f.close()
        result.update(existing_results)

    result.update(
        {
            eval_split + '_eval_loss': eval_loss,
            eval_split + '_eval_accuracy': eval_accuracy,
        }
    )

    with open(metrics_out_file, "w") as writer:
        writer.write(json.dumps(result))

    if verbose:
        logger.info("***** Eval results *****")
        logging.info(json.dumps(result))

    output_file = os.path.join(os.path.dirname(output_dir),
                               eval_split + "_output_predictions.jsonl")

    predictions = []
    for record, pred, logits, probs in zip(eval_examples, eval_predictions, eval_logits,
                                           eval_pred_probs):
        r_json = record.to_json()
        r_json['prediction'] = data_processor.get_labels()[pred]
        r_json['logits'] = logits
        r_json['probs'] = probs
        predictions.append(r_json)
    write_items([json.dumps(r) for r in predictions], output_file)

    return result


def predict(pred_input_file,
            pred_output_file,
            model_dir,
            data_processor,
            model_name_or_path,
            max_seq_length,
            batch_size,
            gpu_id,
            verbose,
            finetuning_model,
            config_name=None):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    pred_examples = data_processor.get_examples_from_file(pred_input_file)

    pred_dataloader = mc_examples_to_data_loader(examples=pred_examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=max_seq_length,
                                                 is_train=False,
                                                 is_predict=True,
                                                 batch_size=batch_size,
                                                 verbose=verbose
                                                 )

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    # Load a trained model that you have fine-tuned
    if torch.cuda.is_available():
        model_state_dict = torch.load(_model_name(model_dir))
    else:
        model_state_dict = torch.load(_model_name(model_dir), map_location='cpu')

    # Pretrained Model
    config = BertConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=len(data_processor.get_labels()),
        finetuning_task="anli"
    )

    model = model_choice_map[finetuning_model].from_pretrained(
        model_dir,
        from_tf=bool('.ckpt' in model_name_or_path),
        config=config
    )

    model.to(device)
    model.eval()

    assert not model.training

    predictions = []

    for input_ids, input_mask, segment_ids in tqdm(pred_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = model_output[0]

        logits = logits.detach().cpu().numpy()

        predictions.extend(np.argmax(logits, axis=1).tolist())

    write_items([idx + 1 for idx in predictions], pred_output_file)


def main(args):
    output_dir = args.output_dir
    seed = args.seed
    model_name_or_path = args.model_name_or_path
    data_dir = args.data_dir
    task_name = args.task_name
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    max_seq_length = args.max_seq_length
    warmup_proportion = args.warmup_proportion
    mode = args.mode
    finetuning_model = args.finetuning_model
    debug = args.debug
    tune_bert = not args.no_tune_bert
    gpu_id = args.gpu_id
    tb_dir = args.tb_dir
    debug_samples = args.debug_samples
    run_on_test = args.run_on_test
    training_data_fraction = args.training_data_fraction
    run_on_dev = True
    metrics_out_file = args.metrics_out_file

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode is None or mode == "train":
        train(data_dir=data_dir,
              output_dir=output_dir,
              data_processor=get_data_processor(task_name),
              model_name_or_path=model_name_or_path,
              lr=lr,
              batch_size=batch_size,
              epochs=epochs,
              finetuning_model=finetuning_model,
              max_seq_length=max_seq_length,
              warmup_proportion=warmup_proportion,
              debug=debug,
              tune_bert=tune_bert,
              gpu_id=gpu_id,
              tb_dir=tb_dir,
              debug_samples=debug_samples,
              training_data_fraction=training_data_fraction
              )
    if mode is None or mode == "eval":
        if run_on_dev:
            evaluate(
                data_dir=data_dir,
                output_dir=output_dir,
                data_processor=get_data_processor(task_name),
                model_name_or_path=model_name_or_path,
                finetuning_model=finetuning_model,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                debug=debug,
                gpu_id=gpu_id,
                verbose=True,
                debug_samples=debug_samples,
                eval_split="dev",
                metrics_out_file=metrics_out_file
            )

        if run_on_test:
            logger.info("*******")
            logger.info("!!!!!!! ----- RUNNING ON TEST ----- !!!!!")
            logger.info("*******")
            evaluate(
                data_dir=data_dir,
                output_dir=output_dir,
                data_processor=get_data_processor(task_name),
                model_name_or_path=model_name_or_path,
                finetuning_model=finetuning_model,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                debug=debug,
                gpu_id=gpu_id,
                verbose=True,
                debug_samples=debug_samples,
                eval_split="test",
                metrics_out_file=metrics_out_file
            )

    if mode == "predict":
        assert args.predict_input_file is not None and args.predict_output_file is not None

        predict(
            pred_input_file=args.predict_input_file,
            pred_output_file=args.predict_output_file,
            model_dir=output_dir,
            data_processor=get_data_processor(task_name),
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            gpu_id=gpu_id,
            verbose=False,
            finetuning_model=finetuning_model
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune BERT model and save')

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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--warmup_proportion',
                        type=float,
                        default=0.2,
                        help="Portion of training to perform warmup")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_samples', default=20, type=int)
    parser.add_argument('--no_tune_bert', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--tb_dir', type=str, default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)

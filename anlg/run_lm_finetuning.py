# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on WikiText-2 (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import pickle
import random

import numpy as np
import torch
from comet.data.atomic import all_categories, make_attention_mask
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, AdamW,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  WarmupLinearSchedule
                          )

from anlg.models import GPT2CometLMHeadModel
from anlg.tokenizers import AnliGpt2Tokenizer, AnliCometGpt2Tokenizer
from utils.file_utils import read_jsonl_lines
import comet.interactive.functions as comet_interactive

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'gpt2_for_anli': (GPT2Config, GPT2CometLMHeadModel, AnliGpt2Tokenizer),
    'gpt2_for_anli_comet': (GPT2Config, GPT2CometLMHeadModel, AnliCometGpt2Tokenizer)
}

restricted_comet_relations = {
    "obs1": ["xEffect", "xWant", "xReact"],
    "obs2": ["xIntent", "xNeed"]
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            while len(tokenized_text) >= block_size:  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.add_special_tokens_single_sentence(tokenized_text[:block_size]))
                tokenized_text = tokenized_text[block_size:]
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def anli_record_to_gpt_prompt(tokenizer: AnliGpt2Tokenizer, record: dict, is_eval: bool = False):
    context = [
        record['obs1'],
        record['obs2'],
        "Because,  "
    ]

    if is_eval:
        return context
    else:
        training_instance = context + [
            record['hyp' + record['label']],
        ]
        return training_instance


def record_to_text_tokens_with_comet_pred(tokenizer: AnliCometGpt2Tokenizer,
                                          record: dict,
                                          include_comet=False,
                                          comet_text_encoder=None,
                                          comet_data_loader=None,
                                          comet_as_text=False,
                                          is_eval: bool = False,
                                          restrict_comet: bool = False,
                                          sotw: bool = False
                                          ):
    comet_event_inputs = None
    comet_attention_masks = None

    context = []

    if include_comet:

        for obs in ['obs1', 'obs2']:
            for category in all_categories:

                if restrict_comet:
                    if category not in restricted_comet_relations[obs]:
                        continue

                if comet_as_text:
                    context.append(tokenizer.category_begin_tag(obs, category))
                    if category in record['comet_preds'][obs] and \
                            record['comet_preds'][obs][category]['beams'][0] != "none":
                        context.append(record['comet_preds'][obs][category]['beams'][0])
                    else:
                        context.append(tokenizer.comet_none)
                    context.append(tokenizer.category_end_tag(obs, category))
                else:
                    if comet_event_inputs is None:
                        comet_event_inputs = []
                    if comet_attention_masks is None:
                        comet_attention_masks = []
                    XMB = np.zeros(25)
                    obs1_comet_input = comet_text_encoder.encode([record[obs]], verbose=False)[0]
                    XMB[:len(obs1_comet_input)] = obs1_comet_input
                    XMB[-1] = comet_text_encoder.encoder["<{}>".format(category)]
                    attention_mask = [1 if item != 0 else 0 for item in XMB]

                    comet_event_inputs.append(XMB)
                    comet_attention_masks.append(attention_mask)

                    if sotw:
                        # only 9 placeholders if using the SOTW model
                        if obs == 'obs1':
                            context.append(tokenizer.unk_token)
                    else:
                        context.append(tokenizer.unk_token)

    context.extend([
        tokenizer.bo1_token,
        record['obs1'],
        tokenizer.eo1_token,
        tokenizer.bo2_token,
        record['obs2'],
        tokenizer.eo2_token,
        tokenizer.bexpl_token,
    ])

    if is_eval:
        return context, comet_event_inputs, comet_attention_masks
    else:
        training_instance = context + [
            record['hyp' + record['label']],
            tokenizer.eexpl_token
        ]
        return training_instance, comet_event_inputs, comet_attention_masks


def _to_hyp_only_labels(tokenizer, tokenized_text):
    hyp_start_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.bexpl_token])[0]
    hyp_end_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.eexpl_token])[0]

    start_idx = tokenized_text.index(hyp_start_token_idx)
    end_idx = tokenized_text.index(hyp_end_token_idx)

    labels = [-1] * len(tokenized_text)

    labels[start_idx + 1: end_idx + 1] = tokenized_text[start_idx + 1:end_idx + 1]
    assert len(tokenized_text) == len(labels)
    return labels


class AnliDataset(Dataset):
    def __init__(self, tokenizer, file_path="train", cache_dir=None, max_seq_len=256,
                 include_comet=False, comet_text_encoder=None, comet_data_loader=None,
                 comet_as_text=False, conditional_lm=False, restrict_comet=False, no_cache=False,
                 is_eval=False,
                 sotw=False):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)

        if include_comet and not comet_as_text:
            max_seq_len = max_seq_len + 18
            logging.info("Increasing max length to {}.".format(max_seq_len))

        if cache_dir is None:
            cached_features_file = os.path.join(directory, f'cached_lm_{filename}')
        else:
            cached_features_file = os.path.join(cache_dir, f'cached_lm_{filename}')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples, self.labels, self.comet_inputs, self.comet_masks = pickle.load(
                    handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            self.labels = []
            self.comet_inputs = []
            self.comet_masks = []
            records = read_jsonl_lines(file_path)
            # with open(file_path, encoding="utf-8") as f:
            #     text = f.read()

            idx = 0
            for record in tqdm(records, "Encoding Data"):

                text_tokens, comet_event_inputs, comet_attention_masks = \
                    record_to_text_tokens_with_comet_pred(
                        tokenizer=tokenizer,
                        record=record,
                        include_comet=include_comet,
                        comet_text_encoder=comet_text_encoder,
                        comet_data_loader=comet_data_loader,
                        comet_as_text=comet_as_text,
                        restrict_comet=restrict_comet,
                        sotw=sotw
                    )

                text = " ".join(text_tokens)
                tokens = tokenizer.tokenize(text)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                else:
                    tokens.extend([tokenizer.unk_token] * (max_seq_len - len(tokens)))

                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
                self.examples.append(tokenized_text)
                if conditional_lm or is_eval:
                    labels = _to_hyp_only_labels(tokenizer, tokenized_text)
                else:
                    unk_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]
                    labels = [-1 if t == unk_token_idx else t for t in tokenized_text]

                self.labels.append(labels)

                self.comet_inputs.append(comet_event_inputs)
                self.comet_masks.append(comet_attention_masks)

                if idx < 5:
                    print("***** Example Instance *****")
                    print("Text: {}".format(text))
                    print("Tokenized Text: {}".format(tokenized_text))
                    if comet_event_inputs is not None:
                        print("Comet Event inputs: {}".format(comet_event_inputs))
                        print("Comet Mask: {}".format(comet_attention_masks))
                    print("Labels: {}".format(labels))
                    print("********\n")

                idx += 1

            if not no_cache:
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump((self.examples, self.labels, self.comet_inputs, self.comet_masks),
                                handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), \
               torch.tensor(self.labels[item]), \
               torch.tensor(self.comet_inputs[item]) if self.comet_inputs[item] is not None else [], \
               torch.tensor(self.comet_masks[item]) if self.comet_masks[item] is not None else []


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer,
                          file_path=args.eval_data_file if evaluate else args.train_data_file,
                          block_size=args.block_size)
    return dataset


def load_and_cache_anli_examples(args, tokenizer, evaluate=False, include_comet=False,
                                 comet_text_encoder=None, comet_data_loader=None, sotw=False):
    dataset = AnliDataset(
        tokenizer,
        file_path=args.eval_data_file if evaluate else args.train_data_file,
        cache_dir=args.cache_dir,
        include_comet=include_comet,
        comet_text_encoder=comet_text_encoder,
        comet_data_loader=comet_data_loader,
        comet_as_text=args.comet_as_text,
        conditional_lm=args.conditional_lm,
        restrict_comet=args.restrict_comet,
        no_cache=args.no_cache,
        is_eval=evaluate,
        max_seq_len=args.block_size,
        sotw=sotw
    )
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer, comet_text_encoder=None, comet_data_loader=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.tb_dir, "tb/"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(
        train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        logging.info("\n\n*** Starting Epoch: {} ***\n\n".format(epoch))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])

        # Modified code to only work for ALNI now. Need to generalize later.
        assert args.task == "anli"

        for step, (inputs, labels, comet_input, comet_mask) in enumerate(epoch_iterator):
            # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, torch.clone(batch))

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            if isinstance(comet_input, list) and len(comet_input) == 0:
                comet_input = None
                comet_mask = None
            else:
                comet_input = comet_input.to(args.device)
                comet_mask = comet_mask.to(args.device)

            model.train()
            outputs = model(inputs, labels=labels, comet_input=comet_input, comet_mask=comet_mask)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1,
                                       0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps,
                                         global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1,
                                       0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logging.info("Evaluate epoch ... {}".format(epoch))
        results = evaluate(args, model, tokenizer, prefix=str(epoch), comet_text_encoder=comet_text_encoder, comet_data_loader=comet_data_loader)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key.split("_")[0]), value, global_step)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer,
             evaluate=False,
             comet_text_encoder=None,
             comet_data_loader=None,
             prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.eval_output_dir

    results = {}

    if args.task is None:
        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    elif args.task == "anli":
        eval_dataset = load_and_cache_anli_examples(args, tokenizer,
                                                    evaluate=True,
                                                    include_comet=args.include_comet,
                                                    comet_text_encoder=comet_text_encoder,
                                                    comet_data_loader=comet_data_loader,
                                                    )
    else:
        raise Exception("Task Unsopported")

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
        eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch, labels, comet_input, comet_mask in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            # labels = torch.clone(batch)
            # if args.task == "anli":
            #     labels[labels == tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]] = -1

            outputs = model(batch, labels=labels)

            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    output_eval_file = os.path.join(eval_output_dir, "metrics.json")

    if os.path.exists(output_eval_file):
        results = json.load(open(output_eval_file))
    else:
        results = {}

    if len(prefix) == 0:
        results.update({
            "perplexity": perplexity.item(),
            "eval_loss": eval_loss
        })
    else:
        results.update({
            "perplexity_{}".format(prefix): perplexity.item(),
            "loss_{}".format(prefix): eval_loss
        })

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write(json.dumps(results))
        writer.close()

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--eval_output_dir", default=None, type=str, required=False,
                        help="Directory to write results to")
    parser.add_argument("--tb_dir", default=None, type=str, required=False,
                        help="Directory to write tensorboard to")

    ## Other parameters
    parser.add_argument("--task", default=None, type=str,
                        help="The task to finetune the LM on. Currently supports None / anli")
    parser.add_argument("--include_comet", default=False, type=bool,
                        help="To include comet predictions or not")
    parser.add_argument("--comet_model_path", default="comet-model/atomic_pretrained_model.th",
                        type=str, help="Comet model path")
    parser.add_argument("--comet_vocab_path", default="comet-vocab/", type=str,
                        help="Comet model path")
    parser.add_argument("--comet_as_text", default=False, type=bool,
                        help="Comet feature encoded using text")
    parser.add_argument("--conditional_lm", default=False, type=bool,
                        help="Comet feature encoded using text")
    parser.add_argument("--restrict_comet", default=False, type=bool,
                        help="Restrict comet features to only o1's effect and o2's causes")
    parser.add_argument("--sotw", default=False, type=bool,
                        help="Use the state of the world model.")
    parser.add_argument("--no_cache", default=False, type=bool,
                        help="Restrict comet features to only o1's effect and o2's causes")

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.eval_output_dir is None:
        args.eval_output_dir = args.output_dir
    if args.tb_dir is None:
        args.tb_dir = args.output_dir

    if args.model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    comet_text_encoder = None
    comet_data_loader = None
    comet_model = None
    if args.include_comet and not args.comet_as_text:
        opt, state_dict, vocab = comet_interactive.load_model_file(args.comet_model_path)
        # print(opt)
        comet_data_loader, comet_text_encoder = \
            comet_interactive.load_data("atomic", opt, vocab, args.comet_vocab_path)

        n_ctx = comet_data_loader.max_event + comet_data_loader.max_effect
        n_vocab = len(comet_text_encoder.encoder) + n_ctx
        if not torch.cuda.is_available():
            comet_interactive.set_compute_mode("cpu")
        comet_model = comet_interactive.make_model(opt, n_vocab, n_ctx, state_dict)
        comet_model.train()
        model.set_comet_model(comet_model)
        model.set_comet_encoder(comet_text_encoder)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.task is None:
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        elif args.task == "anli":
            train_dataset = load_and_cache_anli_examples(
                args,
                tokenizer,
                evaluate=False,
                include_comet=args.include_comet,
                comet_text_encoder=comet_text_encoder,
                comet_data_loader=comet_data_loader,
                sotw=args.sotw
            )
        else:
            raise Exception("Task Unsopported")

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, comet_text_encoder, comet_data_loader)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                                                    do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        comet_model = None
        comet_text_encoder = None
        if args.include_comet and not args.comet_as_text:
            logging.info("Setting comet model")

            opt, state_dict, vocab = interactive.load_model_file(args.comet_model_path)
            # print(opt)
            comet_data_loader, comet_text_encoder = \
                interactive.load_data("atomic", opt, vocab, args.comet_vocab_path)

            n_ctx = comet_data_loader.max_event + comet_data_loader.max_effect
            n_vocab = len(comet_text_encoder.encoder) + n_ctx
            if not torch.cuda.is_available():
                interactive.set_compute_mode("cpu")
            comet_model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.set_comet_model(comet_model)
            model.set_comet_encoder(comet_text_encoder)
            model.to(args.device)
            result = evaluate(
                args,
                model,
                tokenizer,
                evaluate=False,
                comet_text_encoder=comet_text_encoder,
                comet_data_loader=comet_data_loader,
                prefix=global_step
            )
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()

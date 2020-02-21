from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from comet.data.atomic import all_categories
from comet.models.utils import prepare_position_embeddings
from onmt.modules import GlobalAttention
from pytorch_transformers import GPT2PreTrainedModel
from pytorch_transformers.modeling_bert import BertLayerNorm as LayerNorm
from pytorch_transformers.modeling_gpt2 import Block
from torch.nn import Parameter

from anlg.sotw import SymmetricAttentionSOTW


class GPT2CometAttentiveModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2CometAttentiveModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.comet_model = None
        self.comet_encoder = None

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def _comet_embs(self, comet_input, comet_mask):
        batch_size, num_comet_rels = comet_input.size(0), comet_input.size(1)
        comet_input = comet_input.view(batch_size * num_comet_rels, -1)
        comet_mask = comet_mask.view(batch_size * num_comet_rels, -1).float()

        comet_input_with_positions = prepare_position_embeddings(None, self.comet_encoder.encoder,
                                                                 comet_input.unsqueeze(-1))

        comet_embs = self.comet_model.transformer(comet_input_with_positions.unsqueeze(1),
                                                  sequence_mask=comet_mask)[:, -1, :]
        return comet_embs.view(batch_size, num_comet_rels, -1)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, head_mask=None,
                comet_input=None, comet_mask=None
                ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length,
                                        dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)

        if comet_input is not None:
            comet_embs = self._comet_embs(comet_input.long(), comet_mask)
            num_comet_rels = comet_input.size(1)
            inputs_embeds[:, :num_comet_rels, :] = self.ln_f(comet_embs)

        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, layer_past, head_mask[i])
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)

    def set_comet_model(self, comet_model):
        self.comet_model = comet_model

    def set_comet_encoder(self, comet_encoder):
        self.comet_encoder = comet_encoder


class GPT2CometLMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2CometLMHeadModel, self).__init__(config)
        self.transformer = GPT2CometAttentiveModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self,
                input_ids,
                position_ids=None,
                token_type_ids=None,
                labels=None,
                past=None,
                head_mask=None,
                comet_input=None,
                comet_mask=None
                ):
        transformer_outputs = self.transformer(input_ids,
                                               position_ids=position_ids,
                                               token_type_ids=token_type_ids,
                                               past=past,
                                               head_mask=head_mask,
                                               comet_input=comet_input,
                                               comet_mask=comet_mask
                                               )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def set_comet_model(self, comet_model):
        self.transformer.set_comet_model(comet_model)

    def set_comet_encoder(self, comet_encoder):
        self.transformer.set_comet_encoder(comet_encoder)

    def _resize_token_embeddings(self, new_num_tokens):
        self.transformer.resize_token_embeddings(new_num_tokens)


class GPT2SotwAttentiveModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2SotwAttentiveModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.comet_model = None
        self.comet_encoder = None

        self.attn_sotw_model = SymmetricAttentionSOTW(attn_dim=config.n_embd)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def _comet_embs(self, comet_input, comet_mask):
        batch_size, num_comet_rels = comet_input.size(0), comet_input.size(1)
        comet_input = comet_input.view(batch_size * num_comet_rels, -1)
        comet_mask = comet_mask.view(batch_size * num_comet_rels, -1).float()

        comet_input_with_positions = prepare_position_embeddings(None, self.comet_encoder.encoder,
                                                                 comet_input.unsqueeze(-1))

        comet_embs = self.comet_model.transformer(comet_input_with_positions.unsqueeze(1),
                                                  sequence_mask=comet_mask)[:, -1, :]
        return comet_embs.view(batch_size, num_comet_rels, -1)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, head_mask=None,
                comet_input=None, comet_mask=None
                ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length,
                                        dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)

        if comet_input is not None:
            comet_embs = self._comet_embs(comet_input.long(),
                                          comet_mask)  # Is batch_size x 18 x h_dim (2 sentences, 9 atomic relations each)

            # one of these should be the query ? How do we combine these two to create a SOTW ?
            obs1_comet_embs = comet_embs[:, :len(all_categories), :]  # Results in 9 x h_dim
            obs2_comet_embs = comet_embs[:, len(all_categories):, :]  # Results in 9 x h_dim

            sotw = self.attn_sotw_model(obs1_comet_embs, obs2_comet_embs)

            inputs_embeds[:, :len(all_categories), :] = sotw

        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, layer_past, head_mask[i])
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)

    def set_comet_model(self, comet_model):
        self.comet_model = comet_model

    def set_comet_encoder(self, comet_encoder):
        self.comet_encoder = comet_encoder


class GPT2SotwLMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2SotwLMHeadModel, self).__init__(config)
        self.transformer = GPT2SotwAttentiveModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self,
                input_ids,
                position_ids=None,
                token_type_ids=None,
                labels=None,
                past=None,
                head_mask=None,
                comet_input=None,
                comet_mask=None
                ):
        transformer_outputs = self.transformer(input_ids,
                                               position_ids=position_ids,
                                               token_type_ids=token_type_ids,
                                               past=past,
                                               head_mask=head_mask,
                                               comet_input=comet_input,
                                               comet_mask=comet_mask
                                               )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def set_comet_model(self, comet_model):
        self.transformer.set_comet_model(comet_model)

    def set_comet_encoder(self, comet_encoder):
        self.transformer.set_comet_encoder(comet_encoder)

    def _resize_token_embeddings(self, new_num_tokens):
        self.transformer.resize_token_embeddings(new_num_tokens)

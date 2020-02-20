from comet.data.atomic import all_categories
from pytorch_transformers import GPT2Tokenizer


class AnliGpt2Tokenizer(GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 bo1_token="<|beginobs1|>",
                 eo1_token="<|endobs1|>",
                 bo2_token="<|beginobs2|>",
                 eo2_token="<|endobs2|>",
                 bexpl_token="<|bexpl|>",
                 eexpl_token="<|eexpl|>",
                 **kwargs):
        super(AnliGpt2Tokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        self.bo1_token = bo1_token
        self.eo1_token = eo1_token
        self.bo2_token = bo2_token
        self.eo2_token = eo2_token
        self.bexpl_token = bexpl_token
        self.eexpl_token = eexpl_token

        self.add_special_tokens({
            "additional_special_tokens": [self.bo1_token, self.eo1_token, self.bo2_token,
                                          self.eo2_token, self.bexpl_token, self.eexpl_token]
        })

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        pass

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = super().decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
        idx = text.find(self.eexpl_token)
        if idx != -1:
            text = text[:idx]
        return text


class AnliCometGpt2Tokenizer(GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 bo1_token="<|beginobs1|>",
                 eo1_token="<|endobs1|>",
                 bo2_token="<|beginobs2|>",
                 eo2_token="<|endobs2|>",
                 bexpl_token="<|bexpl|>",
                 eexpl_token="<|eexpl|>",
                 comet_token_px="<|personx|>",
                 comet_token_py="<|persony|>",
                 comet_none="<|none|>",
                 **kwargs):
        super(AnliCometGpt2Tokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        self.bo1_token = bo1_token
        self.eo1_token = eo1_token
        self.bo2_token = bo2_token
        self.eo2_token = eo2_token
        self.bexpl_token = bexpl_token
        self.eexpl_token = eexpl_token

        self.comet_token_px = comet_token_px
        self.comet_token_py = comet_token_py

        self.begin_tags = {}
        self.end_tags = {}
        self.comet_none = comet_none

        all_special_tokens = [self.bo1_token,
                              self.eo1_token,
                              self.bo2_token,
                              self.eo2_token,
                              self.bexpl_token,
                              self.eexpl_token,
                              self.comet_token_px,
                              self.comet_token_py,
                              self.comet_none
                              ]

        for obs in ['obs1', 'obs2']:
            for category in all_categories:
                self.begin_tags[(obs, category)] = "<{}{}>".format(obs, category)
                self.end_tags[(obs, category)] = "</{}{}>".format(obs, category)

                all_special_tokens.append("<{}{}>".format(obs, category))
                all_special_tokens.append("</{}{}>".format(obs, category))

        self.add_special_tokens({
            "additional_special_tokens": all_special_tokens
        })

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        pass

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = super().decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
        idx = text.find(self.eexpl_token)
        if idx != -1:
            text = text[:idx]
        return text

    def category_begin_tag(self, obs, category):
        return self.begin_tags[(obs, category)]

    def category_end_tag(self, obs, category):
        return self.end_tags[(obs, category)]


class VCRGpt2Tokenizer(GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 begin_img="<|b_img|>",
                 end_img="<|e_img|>",
                 begin_question="<|b_qn|>",
                 end_question="<|e_qn|>",
                 begin_rationale="<|b_rtnl|>",
                 end_rationale="<|e_rtnl|>",
                 **kwargs):
        super(VCRGpt2Tokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        self.begin_img = begin_img
        self.end_img = end_img
        self.begin_question = begin_question
        self.end_question = end_question
        self.begin_rationale = begin_rationale
        self.end_rationale = end_rationale

        self.add_special_tokens({
            "additional_special_tokens": [self.begin_img, self.end_img, self.begin_question,
                                          self.end_question, self.begin_rationale, self.end_rationale]
        })

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        pass

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = super().decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
        idx = text.find(self.end_rationale)
        if idx != -1:
            text = text[:idx]
        return text
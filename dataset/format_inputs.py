import os
from enum import Enum

import torch

class TASK_TYPE(Enum):
    GENERATE_E2E = 'generate_e2e'
    DAILY_DIALOG = 'daily_dialog'
    SAMSUM = 'samsum'
    EDS = 'eds'
    BOOLQ = 'boolq',
    COMMONSENSE = 'common_170k'
    CONVAI2 = 'convai2'

token_length_map = {
    'e2e_nlg': 150,
    'dailydialog':256,
    'samsum':256,
    'e2e_cleaned': 150,
    'boolq': 256,
    'common_170k': 256,
    'piqa': 256,
    'siqa': 256,
    'hellas': 256,
    'winog': 256,
    'arce': 256,
    'arcc': 256,
    'obqa': 256,
    'eds': 256,
    'common_all': 256,
    'convai2': 512,
    'meta_math': 155,
    'gsm8k': 155,
    'mmlu': 512,
}

gen_max_new_token_map = {
    'e2e_nlg': 40,
    'dailydialog':32,
    'samsum':40,
    'e2e_cleaned': 40,
    'boolq': 16,
    'common_170k': 16,
    'piqa': 16,
    'siqa': 16,
    'hellas': 16,
    'winog': 16,
    'arce': 16,
    'arcc': 16,
    'obqa': 16,
    'common_all': 16,
    'eds': 40,
    'convai2': 32,
    'meta_math': 200,
    'gsm8k': 200,
    'mmlu': 1,
}
dataset_map = {
    'e2e_nlg': 'data_file/e2e',
    'dailydialog': 'data_file/dailydialog',
    'samsum': 'data_file/samsum',
    'e2e_cleaned': 'data_file/e2e_nlg_cleaned',
    'boolq': 'data_file/llm_adapt/boolq',
    'common_170k': 'data_file/llm_adapt/commonsense_170k',
    'piqa': 'data_file/llm_adapt/piqa',
    'siqa': 'data_file/llm_adapt/social_i_qa',
    'hellas': 'data_file/llm_adapt/hellaswag',
    'winog': 'data_file/llm_adapt/winogrande',
    'arce': 'data_file/llm_adapt/ARC-Easy',
    'arcc': 'data_file/llm_adapt/ARC-Challenge',
    'obqa': 'data_file/llm_adapt/openbookqa',
    'eds': 'data_file',
    'convai2': 'data_file/convai2',
    'meta_math': 'data_file/meta_math',
    'common_all': 'data_file/llm_adapt/boolq',
    'gsm8k': 'data_file/llm_adapt/gsm8k',
    'mmlu': 'data_file/mmlu/'
}

task_map = {
    'e2e_nlg': TASK_TYPE.GENERATE_E2E,
    'dailydialog': TASK_TYPE.DAILY_DIALOG,
    'samsum': TASK_TYPE.SAMSUM,
    'e2e_cleaned': TASK_TYPE.GENERATE_E2E,
    'boolq': TASK_TYPE.BOOLQ,
    'common_170k': TASK_TYPE.COMMONSENSE,
    'piqa': TASK_TYPE.BOOLQ,
    'siqa': TASK_TYPE.BOOLQ,
    'hellas': TASK_TYPE.BOOLQ,
    'winog': TASK_TYPE.BOOLQ,
    'arce': TASK_TYPE.BOOLQ,
    'arcc': TASK_TYPE.BOOLQ,
    'obqa': TASK_TYPE.BOOLQ,
    'eds': TASK_TYPE.EDS,
    'convai2': TASK_TYPE.CONVAI2,
    'meta_math': TASK_TYPE.BOOLQ,
    'common_all': TASK_TYPE.BOOLQ,
    'gsm8k': TASK_TYPE.BOOLQ,
    'mmlu': TASK_TYPE.BOOLQ

}

E2E_LLAMA3_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for summarize table into text<|eot_id|><|start_header_id|>user<|end_header_id|>
{cinput}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target}<|end_of_text|>
"""
BOOLQ_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{cinput}

### Response:
{target}"""
BOOLQ_LLAMA3_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Task: Provide a yes or no answer to the following question based on the given passage.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{cinput}
Instructions:
- Start your response with a simple "yes" or "no."
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{target}<|end_of_text|>
"""

CONVAI2_TEMPLATE = """
{cinput}{target}
"""




def format_causal_input(batch, left_tokenizer, right_tokenizer, max_token_length, for_test=False,
                        context_frozen=True, template_type=0, shift_target=True, template_type_list=None, return_query_token=False,
                        target_length=16):
    if template_type_list is None:
        template_type_list = [template_type]*len(batch['input'])

    template_types = [
        'CONTEXT: {cinput} TARGET: {target}',
        '{cinput} R: {target}',
        'Given table text as: {cinput}\t The summary is: {target}',
        E2E_LLAMA3_TEMPLATE,
        BOOLQ_TEMPLATE,
        BOOLQ_LLAMA3_TEMPLATE,
        CONVAI2_TEMPLATE,
        '{cinput}{target}',
    ]
    bos_token = left_tokenizer.bos_token
    eos_token = left_tokenizer.eos_token
    batch_size = len(batch['input'])
    pad_token_id = right_tokenizer.pad_token_id

    targets = [t.strip() for t in batch['target']]
    concat_input = batch['input']
    if return_query_token:
        query_tokenized = right_tokenizer(
            batch['input'], add_special_tokens=False, return_tensors='pt',
                               padding='max_length', truncation=True,
                               max_length=max_token_length
        )
        return query_tokenized
    concat_input_target = [template_types[_template_type].format(cinput=cinput, target=target) for cinput, target, _template_type in
                       zip(concat_input, targets, template_type_list)]
    bos_concat_input = [f'{bos_token}{cinput}{eos_token}' for cinput in concat_input_target]
    lm_input = right_tokenizer(bos_concat_input, add_special_tokens=False, return_tensors='pt',
                               padding='max_length', truncation=True,
                               max_length=max_token_length)
    lm_target = lm_input.copy()
    if shift_target:
        lm_target = torch.cat((lm_target['input_ids'][:, 1:], lm_target['input_ids'].new_full(
            (batch_size, 1), pad_token_id)), dim=1)
    else:
        lm_target = lm_target['input_ids'][:, :]
    if for_test:
        inference_input = [template_types[_template_type].format(cinput=cinput, target='')
                           for cinput, _template_type in zip(concat_input, template_type_list)]
        bos_concat_input = [f'{bos_token}{cinput}' for cinput in inference_input]
        inference_tokenized = left_tokenizer(bos_concat_input, add_special_tokens=False
                                             , return_tensors='pt',
                                             padding=True, truncation=True,
                                             max_length=max_token_length)
        return inference_tokenized, lm_target
    # rewrite with left_tokenizer + right_tokenizer, specially for commonsense series
    if context_frozen and template_type in [4,6]:
        assert shift_target == False, 'make sure we do not shift target by ourself'
        context_input = [template_types[_template_type].format(cinput=cinput, target='')
                          for cinput,_template_type  in zip(concat_input, template_type_list)]
        target_input = targets
        context_tokens = left_tokenizer(context_input, add_special_tokens=False, max_length=max_token_length, padding='max_length', truncation=True, return_tensors='pt')
        target_tokens = right_tokenizer(target_input, add_special_tokens=False, return_tensors='pt', padding='max_length', max_length=target_length, truncation=True)
        lm_input = context_tokens.copy()
        lm_input['input_ids'] = torch.cat((lm_input['input_ids'], target_tokens['input_ids']), dim=1)
        lm_input['attention_mask'] = torch.cat((lm_input['attention_mask'], target_tokens['attention_mask']), dim=1)
        lm_target = context_tokens.copy()['input_ids']
        lm_target[:,:]=left_tokenizer.pad_token_id
        lm_target = torch.cat((lm_target, target_tokens['input_ids']), dim=1)
    elif context_frozen:
        context_tokens = [template_types[_template_type].format(cinput=cinput, target='')
                          for cinput,_template_type  in zip(concat_input, template_type_list)]

        context_tokens = [f'{cinput}' for cinput in context_tokens]
        context_token_ids = left_tokenizer(context_tokens, add_special_tokens=False)['input_ids']
        for _lm_target, _ctx_ids in zip(lm_target, context_token_ids):
            _tokens = left_tokenizer.convert_ids_to_tokens(_lm_target)
            _token_ids = _lm_target
            _start_idx = 0
            _end_idx = len(_ctx_ids) - 2
            try:
                assert _tokens[_end_idx] == ':', 'The end idx position must be :'
            except Exception:
                _end_idx = -1
            _token_ids[:_end_idx + 1] = left_tokenizer.pad_token_id
    return lm_input, lm_target


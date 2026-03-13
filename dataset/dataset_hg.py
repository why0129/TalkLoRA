import json
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.format_inputs import TASK_TYPE, BOOLQ_TEMPLATE


def load_json_data(dataset_path,split) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'{dataset_path}/{split}.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

class HGDataset(torch.utils.data.Dataset):
    # longest first for batch finder
    def __init__(self, data_path, split, data_type: TASK_TYPE, training_ratio=1.0):
        self.path = data_path
        if split == 'validation' and data_type == TASK_TYPE.BOOLQ:
            load_split = 'test'
        else:
            load_split = split
        if data_type in [TASK_TYPE.BOOLQ, TASK_TYPE.COMMONSENSE, TASK_TYPE.CONVAI2]:
            dataset = load_json_data(data_path, load_split)
        else:
            dataset = load_dataset(data_path)[load_split]
        if training_ratio < 1.0:
            dataset = dataset[:int(len(dataset) * training_ratio)]
        else:
            dataset = dataset[:]
        if data_type == TASK_TYPE.GENERATE_E2E:
            input_x = dataset['meaning_representation']
            target = dataset['human_reference']
        elif data_type == TASK_TYPE.SAMSUM:
            input_x = dataset['dialogue']
            target = dataset['summary']
        elif data_type == TASK_TYPE.DAILY_DIALOG:
            all_dialogs = dataset['dialog']
            input_x = []
            target = []
            for dialog in all_dialogs:
                for idx, utterance in enumerate(dialog):
                    if idx == 0:
                        continue
                    previous_dialog = dialog[:idx]
                    indicators = ['Q: ','R: '] if len(previous_dialog)%2==1 else ['R: ','Q: ']
                    previous_dialog = [indicators[pidx%2]+p.strip() for pidx, p in enumerate(previous_dialog)]
                    previous_dialog = ' '.join(previous_dialog)
                    input_x.append(previous_dialog)
                    target.append(utterance.strip())
        elif data_type in [TASK_TYPE.BOOLQ] and 'math' in data_path:
            input_x = [d['sQuestion'] for d in dataset]
            target = [d['lSolutions'] for d in dataset]
        elif data_type in [TASK_TYPE.BOOLQ, TASK_TYPE.COMMONSENSE, TASK_TYPE.CONVAI2]:
            _sample = dataset[0]
            input_x = [d['instruction'] for d in dataset]
            target = [d['output'] for d in dataset]
        else:
            raise ValueError('unsupported data type!')
        self.input_x = input_x
        self.target = target
        self.split = split
        self.data_type = data_type
        assert len(input_x) == len(target)

    def length_analysis(self, tokenizer):
        input_x = self.input_x
        target = self.target
        tokenized_input_x = tokenizer(input_x)
        input_ids = tokenized_input_x['input_ids']
        tokenized_target = tokenizer(target)
        target_ids = tokenized_target['input_ids']
        tokenized_combined = tokenizer([x+y for x, y in zip(input_x, target)])
        combined_input_ids = tokenized_combined['input_ids']
        input_ids_length = np.asarray([len(i) for i in input_ids])
        target_ids_length = np.asarray([len(i) for i in target_ids])
        combined_ids_length = np.asarray([len(i) for i in combined_input_ids])
        print(f"""
        ----split: {self.split}----
        min input length: {min(input_ids_length)}
        mean input length: {input_ids_length.mean()}
        max input length: {max(input_ids_length)}
        min target length: {min(target_ids_length)}
        mean target length: {target_ids_length.mean()}
        median combined length: {np.median(target_ids_length)}
        max target length: {max(target_ids_length)}
        """)
        print(f"""
        
        min combined length: {min(combined_ids_length)}
        mean combined length: {combined_ids_length.mean()}
        median combined length: {np.median(combined_ids_length)}
        max combined length: {max(combined_ids_length)}
        """)

        if self.data_type == TASK_TYPE.BOOLQ:
            template_ids = tokenizer(BOOLQ_TEMPLATE)
            input_ids = template_ids['input_ids']
            input_ids_length = len(input_ids)
            print("BOOLQ_TEMPLATE: ", input_ids_length)



    def __getitem__(self, idx):
        input = self.input_x[idx]
        target = self.target[idx]
        return {
            'input': input,
            'target': target,
            'split': self.split,
        }

    def __len__(self):
        return len(self.input_x)


def collate_fn(sample_list):
    dont_be_a_tensor = ['input', 'target']
    to_be_flattened = [*dont_be_a_tensor]
    data = {}
    for key in to_be_flattened:
        if key not in sample_list[0].keys():
            continue
        if sample_list[0][key] is None:
            continue
        flatten_samples = [sample[key] for sample in sample_list]
        if flatten_samples[-1].__class__ == str or key in dont_be_a_tensor:
            data[key] = flatten_samples
        else:
            data[key] = torch.tensor(flatten_samples)
    return data


def collate_fn_straight(sample_list):
    sample_list = collate_fn(sample_list)
    return sample_list


def collate_fn_straight_with_fn(fn):
    def build_collate_fn(sample_list):
        sample_list = collate_fn(sample_list)
        sample_list_processed = fn(sample_list)
        return {**sample_list, **sample_list_processed}

    return build_collate_fn


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=None, collate_fn=None, sampler=None):
    if num_workers is None:
        num_workers = batch_size // 4
    # num_workers = min(num_workers, batch_size)
    if collate_fn is None:
        _collate_fn = collate_fn_straight
    else:
        _collate_fn = collate_fn_straight_with_fn(collate_fn)
    return DataLoader(dataset, batch_size=batch_size,
                      collate_fn=_collate_fn,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      sampler=sampler)

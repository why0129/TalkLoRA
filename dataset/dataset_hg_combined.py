import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.format_inputs import TASK_TYPE


class HGCombinedDataset(torch.utils.data.Dataset):
    # longest first for batch finder
    def __init__(self, data_path, split, data_type: TASK_TYPE, training_ratio=1.0):
        assert data_type == TASK_TYPE.EDS
        self.path = data_path
        dailydialog_path = self.path+os.sep+'dailydialog'
        e2e_path = self.path+os.sep+'e2e'
        samsum_path = self.path+os.sep+'samsum'
        dailydialog_dataset = load_dataset(dailydialog_path)[split]
        e2e_dataset = load_dataset(e2e_path)[split]
        samsum_dataset = load_dataset(samsum_path)[split]
        # dataset = load_dataset(data_path)[split]
        if training_ratio < 1.0:
            dailydialog_dataset = dailydialog_dataset[:int(len(dailydialog_dataset) * training_ratio)]
            e2e_dataset = e2e_dataset[:int(len(e2e_dataset) * training_ratio)]
            samsum_dataset = samsum_dataset[:int(len(samsum_dataset) * training_ratio)]
        else:
            dailydialog_dataset = dailydialog_dataset[:]
            e2e_dataset = e2e_dataset[:]
            samsum_dataset = samsum_dataset[:]

        input_x = []
        target = []
        input_type = []
        input_x += e2e_dataset['meaning_representation']
        target += e2e_dataset['human_reference']
        input_type += ['e2e']*len(e2e_dataset['meaning_representation'])
        input_x += samsum_dataset['dialogue']
        target += samsum_dataset['summary']
        input_type += ['samsum']*len(samsum_dataset['dialogue'])
        all_dialogs = dailydialog_dataset['dialog']
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
                input_type += ['dailydialog']
        self.input_x = input_x
        self.target = target
        self.split = split
        self.input_type = input_type
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
        min input length: {min(input_ids_length)}
        mean input length: {input_ids_length.mean()}
        max input length: {max(input_ids_length)}
        min target length: {min(target_ids_length)}
        mean target length: {target_ids_length.mean()}
        max target length: {max(target_ids_length)}
        """)
        print(f"""
        split: {self.split}
        min combined length: {min(combined_ids_length)}
        mean combined length: {combined_ids_length.mean()}
        median combined length: {np.median(combined_ids_length)}
        max combined length: {max(combined_ids_length)}
        """)



    def __getitem__(self, idx):
        input = self.input_x[idx]
        target = self.target[idx]
        input_type = self.input_type[idx]
        return {
            'input': input,
            'target': target,
            'split': self.split,
            'input_type': input_type
        }

    def __len__(self):
        return len(self.input_x)


def collate_fn(sample_list):
    dont_be_a_tensor = ['input', 'target', 'input_type']
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

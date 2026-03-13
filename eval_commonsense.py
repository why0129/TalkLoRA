import csv
import re
from functools import reduce
from glob import glob
import jsonlines
import numpy as np
from dotenv import load_dotenv
import evaluate
import os
import json

from tqdm import tqdm

load_dotenv('.env')

eval_folder_path = ''

def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['siqa', 'arcc', 'arce', 'obqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellas':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winog':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

eval_tasks = [
'boolq',
'piqa',
'siqa',
'arcc',
'arce',
'obqa',
'hellas',
'winog',
]
all_data = {}
for eval_task in eval_tasks:
    eval_paths = glob(eval_folder_path)
    eval_path = eval_paths[0]
    jsonl_files = glob(f'{eval_path}/*/*{eval_task}*.jsonl')
    csv_header = ['folder_name', 'acc', 'config']
    csv_data = []
    for json_file in tqdm(jsonl_files):
        print(json_file)
        json_data = []
        lines = []
        with open(json_file, 'r') as f:
            for line in f.readlines():
                lines.append(line)
        lines = [line for line in lines if len(line.strip()) > 0]
        for line in lines:
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)
        configs = json_data[:2]
        contents = [line for line in json_data if line.__class__ is dict and 'context' in line.keys()]
        answers = []
        for content in contents:
            pred = extract_answer(f'{eval_task}', content['pred'])
            gt = extract_answer(f'{eval_task}', content['gt'])
            answers.append(pred==gt)
        acc = np.asarray(answers).mean()
        _json_paths = json_file.split(os.sep)
        _json_paths[-1] = _json_paths[-1].replace(eval_task, '')
        folder_name = os.sep.join(_json_paths)
        # folder_name = json_file.replace(eval_task, '')
        csv_row = [folder_name, acc*100.0, configs]
        if folder_name in all_data.keys():
            all_data[folder_name][eval_task]=acc*100
        else:
            all_data[folder_name] = {eval_task:acc*100}
        csv_data.append(csv_row)
    with open(f'{eval_path}/results_{eval_task}.csv', 'w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

type_name = [n for n in eval_folder_path.split(os.sep) if len(n)>0][-1]
with open(f'{eval_folder_path}/{type_name}_results_all.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['model',*eval_tasks, 'avg'])
    for folder_name in all_data.keys():
        row = [folder_name]
        for eval_task in eval_tasks:
            row.append(all_data[folder_name].get(eval_task,-1))
        avg_val = reduce(lambda x,y: x+y, row[1:])/len(row[1:])
        writer.writerow(row+[avg_val])



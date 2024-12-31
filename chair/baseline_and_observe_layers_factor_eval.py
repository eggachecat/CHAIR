# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
from llamafactory.hparams.parser import HfArgumentParser, _EVAL_ARGS
from chair_common import DataCollectorFactory, get_intervention_config
import ssl
import urllib.request
import pyvene as pv

from baseline_and_observe_layers_dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

def load_csv(file_path):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    '''
    Data format:

    ,full_prefix,doc_id,completion,contradiction_0,contradiction_1,contradiction_2,longest_completions,turncated_prefixes
    0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. ",0,Whether or not it gets a second season of The Witcher is another question.,Whether or not it gets a second season of Stranger Things is another question.,Whether or not it gets a fifth season of The Witcher is another question.,Whether or not it gets a second season of Black Mirror is another question.,15.0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. "

    '''
    list_data_dict = []
    df = pd.read_csv(file_path)
    if 'news' in file_path:
        prefix_type = 'full_prefix'
    else:
        prefix_type = 'turncated_prefixes'
    for idx in range(len(df)):
        item = dict(
            prefix=df[prefix_type][idx],
            completion=df['completion'][idx],
            contradiction_0=df['contradiction_0'][idx],
            contradiction_1=df['contradiction_1'][idx],
            contradiction_2=df['contradiction_2'][idx],
        )
        list_data_dict.append(item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DataCollectorFactory:
    
    class DataCollector:
        is_source_constant = True
        keep_last_dim = True
        intervention_types = 'data_collector'
        def __init__(self, component, scope, **kwargs):
            super().__init__(**kwargs)
            self.component = component
            self.scope = scope

        def __call__(self, base, source=None, subspaces=None, model=None):
            self.scope.data_collection[self.component] = model.model.lm_head(base)
            return base

    def __init__(self):
        self.data_collection = {}

    def create(self, component):
        return self.DataCollector(component, scope=self)

    def clear(self):
        self.data_collection = {}

def print_module_tree(module, module_name, indent=0):
    print("  " * indent + module_name + ", " + str(module.__class__.__name__))
    for name, sub_module in module.named_children():
        print_module_tree(sub_module, name, indent + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(_EVAL_ARGS)
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    args = parser.parse_args()

    # Get test file
    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    list_data_dict = load_csv(fp)

    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    llm = DoLa()
    llm.set_stop_words(["Q:", "\end{code}"])
    data_collection_factory = None

    mode = "chair"
    # index 0 -> 1 layers.0.input
    # index 1,2, ... 32 -> layers.i-1.output
    data_collection_factory = DataCollectorFactory()
    intervention_config, intervented_keys = get_intervention_config(
        data_collection_factory, llm.model.config.name_or_path
    )
    llm.model = pv.IntervenableModel(intervention_config, model=llm.model)
    llm.intervented_keys = intervented_keys


    answers = []
    result_dict = {
        'is_correct': [], 
        'model_answer': [], 
        'model_completion': [], 
        'full_input_text': [], 
        'modle_true_outputs': [], 
        'modle_false_outputs': []
    }
    for sample in tqdm(list_data_dict):
        modle_true_outputs, modle_false_outputs = [], []
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens, 
            top_p=args.top_p, 
            top_k=args.top_k, 
            temperature=args.temperature, 
            repetition_penalty=args.repetition_penalty, 
            mode=mode, 
            relative_top=args.relative_top, 
            relative_top_value=args.relative_top_value, 
            data_collection_factory=data_collection_factory
        )
        answer_true_log_prob, c_dist = llm.lm_score(context, answer_true, **generate_kwargs)
        modle_true_outputs.append(c_dist)
        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob, c_dist = llm.lm_score(context, answer_false, **generate_kwargs)
            modle_false_outputs.append(c_dist)
            answer_false_log_probs.append(answer_false_log_prob)
        if args.debug:
            print(f'log prob of answers: {answer_true_log_prob}', end=' ')
            for answer_false_log_prob in answer_false_log_probs:
                print(f'{answer_false_log_prob}', end=' ')
            print()
        is_cor = True
        for answer_false_log_prob in answer_false_log_probs:
            if answer_true_log_prob < answer_false_log_prob:
                is_cor = False
                break
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)
        result_dict['modle_true_outputs'].append(modle_true_outputs)
        result_dict['modle_false_outputs'].append(modle_false_outputs)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    # save results to a json file
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"{float(sum(answers))/len(answers)}")
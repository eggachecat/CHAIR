import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from eggachecat import EggachecatClassifier

wsFolder = os.path.dirname(os.path.abspath(__file__))
base_folder = f"{wsFolder}/saves/evaluation/dola-oberservation-layer/Meta-Llama-3-8B-Instruct"
base_folder = f"{wsFolder}/saves/evaluation/dola-oberservation-layer/Mistral-7B-Instruct-v0.3"

def MC_calcs(scores_true, scores_false, ref_true, ref_best):
    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[ref_true.index(ref_best)] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0

    # compute MC3: 1vFalse -- each correct answer vs all false answers
    max_false = max(scores_false)
    onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
    scores['MC3'] = onevall

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    while sum(probs_true) == 0:
        print("WARNING: all zero scores_true")
        scores_true = [x/2.0 for x in scores_true]
        probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)
    while sum(probs_false) == 0:
        print("WARNING: all zero scores_false")
        scores_false = [x/2.0 for x in scores_false]
        probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    
    # check nan
    if np.isnan(sum(probs_true)):
        scores['MC2'] = 0.0
        print(f"WARNING: nan in probs_true: sum(probs_true)={sum(probs_true)}, sum(probs_false)={sum(probs_false)}")
    else:
        scores['MC2'] = sum(probs_true)

    return scores

def prepare_data_truthfulqa():
    layer_list = list(range(1, 33))
    true_choice_score_trace = {}
    false_choice_score_trace = {}
    question_number_list = list(range(817))
    question_description_dict = {}
    question_ture_label_list = {}
    question_false_label_dict = {}
    question_best_correct_answer_dict = {}
    question_best_correct_answer_index = {}

    with open(f"{base_folder}/layer-1.json", "r") as fp:
        layer_json = json.load(fp)

    for question_number, info in enumerate(layer_json['question']):
        question_description_dict[question_number] = info['question']
        question_ture_label_list[question_number] = info['answer_true'].split("; ")
        question_false_label_dict[question_number] = info['answer_false'].split("; ")
        question_best_correct_answer_dict[question_number] = info['answer_best']
        question_best_correct_answer_index[question_number] = question_ture_label_list[question_number].index(
            info['answer_best']
        )

    for layer_number in layer_list:
        with open(f"{base_folder}/layer-{layer_number}.json", "r") as fp:
            layer_json = json.load(fp)

        for question_number in question_number_list:

            if question_number not in true_choice_score_trace:
                true_choice_score_trace[question_number] = {}

            if question_number not in false_choice_score_trace:
                false_choice_score_trace[question_number] = {}


            output = layer_json['model_scores'][question_number]


            for choice, score in enumerate(output['scores-true']):
                true_choice_score_trace[question_number].setdefault(
                    choice, []).append(score)

            for choice, score in enumerate(output['scores-false']):
                false_choice_score_trace[question_number].setdefault(
                    choice, []).append(score)


    baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

    for question_number in question_number_list:
        original_baseline_for_true = [history[-1] for _, history in true_choice_score_trace[question_number].items()]
        original_baseline_for_false = [history[-1] for _, history in false_choice_score_trace[question_number].items()]
        scores = MC_calcs(
            original_baseline_for_true, original_baseline_for_false, 
            question_ture_label_list[question_number], question_best_correct_answer_dict[question_number]
        )
        baseline_result_dict['total_mc1'] += scores['MC1'] / len(question_number_list)
        baseline_result_dict['total_mc2'] += scores['MC2'] / len(question_number_list)
        baseline_result_dict['total_mc3'] += scores['MC3'] / len(question_number_list)

    print("verify baseline_result_dict", baseline_result_dict)

    return true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict, question_description_dict, question_false_label_dict

def train_model_truthfulqa(path_to_save_model):
    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict, question_description_dict, question_false_label_dict = prepare_data_truthfulqa()
    question_number_list_to_train = question_number_list

    true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in true_choice_score_trace[question_number].items()]
    false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in false_choice_score_trace[question_number].items()]
    
    model = EggachecatClassifier()
    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(path_to_save_model)), exist_ok=True)
    model.save(path_to_save_model)

def evaluate_model_truthfulqa(path_to_save_model):
    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict, question_description_dict, question_false_label_dict = prepare_data_truthfulqa()
    model = EggachecatClassifier.load(path_to_save_model)

    total = len(question_number_list) 
    baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    finetuned_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

    for question_number in question_number_list:
        ################################################################################
        original_baseline_for_true = [history[-1] for _, history in true_choice_score_trace[question_number].items()]
        original_baseline_for_false = [history[-1] for _, history in false_choice_score_trace[question_number].items()]
        scores = MC_calcs(
            original_baseline_for_true, original_baseline_for_false, 
            question_ture_label_list[question_number], question_best_correct_answer_dict[question_number]
        )
        baseline_result_dict['total_mc1'] += scores['MC1'] / total
        baseline_result_dict['total_mc2'] += scores['MC2'] / total
        baseline_result_dict['total_mc3'] += scores['MC3'] / total
        ################################################################################
        X_for_true = np.array([history for _, history in true_choice_score_trace[question_number].items()])
        X_for_false = np.array([history for _, history in false_choice_score_trace[question_number].items()])
        log_proba_prediction_for_true = model.predict_log_proba(X_for_true)[:,1]
        log_proba_prediction_for_false = model.predict_log_proba(X_for_false)[:,1]

        scores = MC_calcs(
            log_proba_prediction_for_true, log_proba_prediction_for_false, 
            question_ture_label_list[question_number], question_best_correct_answer_dict[question_number]
        )
        finetuned_result_dict['total_mc1'] += scores['MC1'] / total
        finetuned_result_dict['total_mc2'] += scores['MC2'] / total
        finetuned_result_dict['total_mc3'] += scores['MC3'] / total

    delta_dict = dict([(k, finetuned_result_dict[k] - baseline_result_dict[k]) for k in ['total_mc1', 'total_mc2', 'total_mc3']])
    print(pd.DataFrame([
        baseline_result_dict, 
        finetuned_result_dict,
        delta_dict
    ], index=["Original", "FineTuned", "Delta"]))

def train_model_factor(path_to_save_model):
    random.seed(42)
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola-eggachecat-observe-layers.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)

    true_choice_history_list = []
    for true_outputs in observation_json['modle_true_outputs']:
        for history in true_outputs:
            true_choice_history_list.append(np.array(history).min(axis=0).tolist())

    false_choice_history_list = []
    for false_outputs in observation_json['modle_false_outputs']:
        for history in false_outputs:
            false_choice_history_list.append(np.array(history).min(axis=0).tolist())

    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model = EggachecatClassifier()

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    # model.print_importance()
    model.save(path_to_save_model)

def evaluate_model_factor(path_to_save_model):
    model = EggachecatClassifier.load(path_to_save_model)
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/Mistral-7B-Instruct-v0.3/factor-wiki-dola-eggachecat-observe-layers.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)
    print(list(observation_json.keys()))
    result_list = []
    baseline_result_list = []
    for i, (true_outputs, false_outputs, model_completion) in tqdm(enumerate(zip(
        observation_json['modle_true_outputs'],  
        observation_json['modle_false_outputs'],
        observation_json['model_completion']
    ))):
        original_true, orignal_false_list = model_completion[0], model_completion[1:]
        if min([original_true]) > max(orignal_false_list):
            baseline_result_list.append(1)
        else:
            baseline_result_list.append(0)

        log_proba_prediction_for_true = model.predict_log_proba([np.array(history).sum(axis=0).tolist() for history in true_outputs])[:,1]
        log_proba_prediction_for_false = model.predict_log_proba([np.array(history).sum(axis=0).tolist() for history in false_outputs])[:,1]
        if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
            result_list.append(1)
        else:
            result_list.append(0)

    # baseline_score = sum(baseline_result_list) / len(baseline_result_list)
    # hardcode because the result is already the output of a new scorer
    baseline_score = 0.604

    new_score = sum(result_list) / len(result_list)
    print(f"baseline(with trained on truthfulqa): {baseline_score}")
    print(f"new_score: {new_score}")
    print(f"delta: {new_score - baseline_score}")

def train_model_mmlu(path_to_save_model):
    random.seed(42)
    
    json_path = f"{wsFolder}/saves/evaluation/mmlu_observation_shot_0/layer-observation-0.json"
    json_path = f"{wsFolder}/saves/evaluation/Mistral-7B-Instruct-v0.3/mmlu_test_eggachecat_shot_0_mmlu_layer_observation/layer-observation-0.json"

    with open(json_path, "r") as fp:
        observation_json = json.load(fp)

    true_choice_history_list = []
    false_choice_history_list = []

    for i, (content, correct_ans, choices_layer_history) in enumerate(zip(
        observation_json['content'],  
        observation_json['truth'],
        observation_json['layer_history']
    )):
        correct_index = {"A":0, "B":1, "C":2, "D":3}[correct_ans]
        true_choice_history_list.extend([history for i, history in enumerate(choices_layer_history) if i == correct_index])
        false_choice_history_list.extend([history for i, history in enumerate(choices_layer_history) if i != correct_index])


    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model = EggachecatClassifier()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    # model.print_importance()
    model.save(path_to_save_model)


def evaluate_model_mmlu(path_to_save_model):
    model = EggachecatClassifier.load(path_to_save_model)

    json_path = f"{wsFolder}/saves/evaluation/mmlu_observation_shot_0/layer-observation-0.json"
    json_path = f"{wsFolder}/saves/evaluation/Mistral-7B-Instruct-v0.3/mmlu_test_eggachecat_shot_0_mmlu_layer_observation/layer-observation-0.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)

    result_list = []
    baseline_result_list = []

    for i, (content, correct_ans, choices_layer_history) in enumerate(zip(
        observation_json['content'],  
        observation_json['truth'],
        observation_json['layer_history']
    )):
        correct_index = {"A":0, "B":1, "C":2, "D":3}[correct_ans]

        if min([history[-1] for i, history in enumerate(choices_layer_history) if i == correct_index]) > max([
           history[-1] for i, history in enumerate(choices_layer_history) if i != correct_index
        ]):
            baseline_result_list.append(1)
        else:
            baseline_result_list.append(0)

        log_proba_prediction_for_true = model.predict_log_proba([history for i, history in enumerate(choices_layer_history) if i == correct_index])[:,1]
        log_proba_prediction_for_false = model.predict_log_proba([history for i, history in enumerate(choices_layer_history) if i != correct_index])[:,1]
        if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
            result_list.append(1)
        else:
            result_list.append(0)

    # baseline_score = sum(baseline_result_list) / len(baseline_result_list)
    # hardcode because the result is already the output of a new scorer
    baseline_score = 0.5958
    new_score = sum(result_list) / len(result_list)
    print(f"baseline(with trained on truthfulqa): {baseline_score}")
    print(f"new_score: {new_score}")
    print(f"delta: {new_score - baseline_score}")

def main():

    for train_func, model_path in [
        [train_model_truthfulqa, "./saves/models/eggachecat_performance_matrix/Mistral-7B-Instruct-v0.3/trained_from_truthfulqa.pkl"],
        [train_model_factor, "./saves/models/eggachecat_performance_matrix/Mistral-7B-Instruct-v0.3/trained_from_factor.pkl"],
        [train_model_mmlu, "./saves/models/eggachecat_performance_matrix/Mistral-7B-Instruct-v0.3/trained_from_mmlu_zero_shot.pkl"]
    ]:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING WITH", train_func)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        train_func(model_path)
        for eval_func in [
            evaluate_model_truthfulqa, 
            evaluate_model_factor, 
            evaluate_model_mmlu
        ]:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EVALUATING WITH", eval_func)
            eval_func(model_path)
        print("=============================================================================================================================================================================================================================================================================================================================================")
    # train_model_mmlu()

if __name__ == "__main__":
    main()
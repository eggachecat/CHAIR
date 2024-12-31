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
from eggachecat import EggachecatClassifier

wsFolder = os.path.dirname(os.path.abspath(__file__))
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

def prepare_data():
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

    return true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict

def run_few_supervised_once(n_observation=10):
    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict = prepare_data()
    question_number_list_to_train = random.sample(question_number_list, n_observation)
    true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in true_choice_score_trace[question_number].items()]
    false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in false_choice_score_trace[question_number].items()]
    
    model = EggachecatClassifier()
    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model.fit(X, y)


    total = len(question_number_list) - len(question_number_list_to_train)
    baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    finetuned_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

    for question_number in question_number_list:
        if question_number in question_number_list_to_train:
            continue
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

    return delta_dict

def run_few_supervised(n_observation=10, times=100):
    delta_dict_list = []
    for _ in range(times):
        delta_dict_list.append(run_few_supervised_once(n_observation=n_observation))
    
    df = pd.DataFrame(delta_dict_list)
    print(df)
    df.to_csv(f"./saves/evaluation/analyze_truthfulqa_with_eggachecat/run_few_supervised_{n_observation}_{times}.csv",index=False)
    return df

def run_kfold(n_splits=5):

    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict = prepare_data()

    average_delta_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    average_baseline_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    average_finetuned_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (question_number_list_to_train, _) in enumerate(kf.split(
        np.array(question_number_list).reshape(-1, 1)
    )):
        print("fold", fold)

        true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in true_choice_score_trace[question_number].items()]
        false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in false_choice_score_trace[question_number].items()]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        model.fit(X, y)



        total = len(question_number_list) - len(question_number_list_to_train)
        baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
        finetuned_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

        for question_number in question_number_list:
            if question_number in question_number_list_to_train:
                continue
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

        print(pd.DataFrame([
            baseline_result_dict, 
            finetuned_result_dict,
            dict([(k, finetuned_result_dict[k] - baseline_result_dict[k]) for k in ['total_mc1', 'total_mc2', 'total_mc3']])
        ], index=["Original", "FineTuned", "Delta"]))

        for k in ['total_mc1', 'total_mc2', 'total_mc3']:
            average_baseline_dict[k] +=  baseline_result_dict[k] * total / len(question_number_list)
            average_finetuned_dict[k] += finetuned_result_dict[k] * total / len(question_number_list)
            average_delta_dict[k] += (finetuned_result_dict[k] -  baseline_result_dict[k]) * total / len(question_number_list)
            
    print("summary")
    print(pd.DataFrame([
        average_baseline_dict, 
        average_finetuned_dict,
        average_delta_dict
    ], index=["Original", "FineTuned", "Delta"]))

def train_and_model(path_to_save_model):
    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict = prepare_data()
    question_number_list_to_train = question_number_list

    true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in true_choice_score_trace[question_number].items()]
    false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in false_choice_score_trace[question_number].items()]
    
    model = EggachecatClassifier()
    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(path_to_save_model)), exist_ok=True)
    model.save(path_to_save_model)


def evaluate_model(path_to_save_model):
    true_choice_score_trace, false_choice_score_trace, question_number_list, question_ture_label_list, question_best_correct_answer_dict = prepare_data()
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


def main():
    random.seed(42)

    # run_kfold()
    # for n_observation in [
    #     5, 10, 20, 30, 80, 100
    # ]:
    #     run_few_supervised(
    #         n_observation=n_observation, 
    #         times=50
    #     )
    model_path = "./saves/models/eggachecat/Mistral-7B-Instruct-v0.3/trained_from_truthfulqa.pkl"
    train_and_model(model_path)
    evaluate_model(model_path)
    # evaluate_model("./saves/models/eggachecat/Mistral-7B-Instruct-v0.3/trained_from_factor.pkl")

if __name__ == "__main__":
    main()
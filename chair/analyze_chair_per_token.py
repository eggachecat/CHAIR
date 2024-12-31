import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import tqdm

RANDOM_STATE = 42

class EggachecatClassifier:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', penalty='l2', random_state=RANDOM_STATE, max_iter=1000)
        # self.model = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        # self.model = SVC(class_weight='balanced', gamma='auto', probability=True)

        self.feature_labels = ['mean', 'max', 'min', 'std', 'slope']
        self.score_scalar = 300

    def fit(self, token_score_records_list, y):
        X = [self.extract_features(x) for x in token_score_records_list]
        self.model.fit(X, y)

    def predict_log_proba(self, token_score_records_list):
        return self.model.predict_log_proba([self.extract_features(x) for x in token_score_records_list]) *  self.score_scalar

    def predict(self, token_score_records_list):
        return self.model.predict([self.extract_features(x) for x in token_score_records_list])

    def extract_features(self, token_score_records):
        """
        从单条时间序列中提取统计和趋势特征，并拼接原始值：
        - 均值、标准差、最大值、最小值、斜率
        - 拼接时间序列的原始值
        """
        last_score = token_score_records[-1]
        token_score_records = [x / last_score for x in token_score_records[:-1]]
        max_v = max(token_score_records)
        min_v = min(token_score_records)
        slope = np.polyfit(range(len(token_score_records)), token_score_records, 1)[0]  # 线性拟合斜率
        mean = np.mean(token_score_records)
        std = np.std(token_score_records)

        delta_degree_1 = []
        for i in range(len(token_score_records)-1):
            delta_degree_1.append(token_score_records[i+1]-token_score_records[i])
        delta_std_degree_1 = np.std(delta_degree_1)

        delta_degree_2 = []
        for i in range(len(token_score_records)-2):
            delta_degree_2.append(token_score_records[i+2]-token_score_records[i])
        delta_std_degree_2 = np.std(delta_degree_2)


        # 拼接原始值和提取的特征
        features = np.hstack([
            # token_score_records,
            # delta_degree_1,
            # delta_degree_2,
            [
                # last_score,
                mean, 
                max_v, 
                min_v, 
                std, 
                slope,
                delta_std_degree_1,
                delta_std_degree_2
            ]
        ])
        features = (max(features) - features) / (max(features) - min(features))
        features[np.isnan(features)] = 0

        return features
    
    def print_importance(self):
        feature_importance = sorted(
            enumerate(self.model.coef_[0]),  # [(index, coef), ...]
            key=lambda x: abs(x[1]),    # 根据系数绝对值排序
            reverse=True                # 从大到小排序
        )
        print("Sorted Feature Importance:")
        for index, coef in feature_importance:
            print(f"Feature <{self.feature_labels[index]}>: Coefficient = {coef:.4f}")

    def save(self, output_path):
        with open(output_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, output_path) -> 'EggachecatClassifier':
        with open(output_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model

class BasicAnalyist:
    def __init__(self):
        pass

class TruthfulQA_Analyist:
    def __init__(self, history_json_path, agg_func):
        self.history_json_path = history_json_path
        self.question_number_list = list(range(817))
        self.true_choice_score_trace = {}
        self.false_choice_score_trace = {}
        self.question_description_dict = {}
        self.question_ture_label_list = {}
        self.question_false_label_dict = {}
        self.question_best_correct_answer_dict = {}
        self.question_best_correct_answer_index = {}

        self.baseline_true_choice_score_trace = {}
        self.baseline_false_choice_score_trace = {}
        self.agg_func = agg_func 

        self.prepare_dataset()

    def prepare_dataset(self):

        with open(self.history_json_path, "r") as fp:
            layer_json = json.load(fp)

        for question_number, info in enumerate(layer_json['question']):
            self.question_description_dict[question_number] = info['question']
            self.question_ture_label_list[question_number] = info['answer_true'].split("; ")
            self.question_false_label_dict[question_number] = info['answer_false'].split("; ")
            self.question_best_correct_answer_dict[question_number] = info['answer_best']
            self.question_best_correct_answer_index[question_number] = self.question_ture_label_list[question_number].index(
                info['answer_best']
            )

            if question_number not in self.true_choice_score_trace:
                self.true_choice_score_trace[question_number] = {}
                self.baseline_true_choice_score_trace[question_number] = {}

            if question_number not in self.false_choice_score_trace:
                self.false_choice_score_trace[question_number] = {}
                self.baseline_false_choice_score_trace[question_number] = {}

        model_scores = layer_json['model_scores']
        for question_number in self.question_number_list:
            for choice, multi_token_history in enumerate(model_scores[question_number]['scores_true_tokens_trace']):
                self.true_choice_score_trace[question_number][choice] = self.agg_func(multi_token_history)
                self.baseline_true_choice_score_trace[question_number][choice] = np.array(multi_token_history).sum(axis=0)


            for choice, multi_token_history in enumerate(model_scores[question_number]['scores_false_tokens_trace']):
                self.false_choice_score_trace[question_number][choice] = self.agg_func(multi_token_history)
                self.baseline_false_choice_score_trace[question_number][choice] = np.array(multi_token_history).sum(axis=0)


        baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

        for question_number in self.question_number_list:
            original_baseline_for_true = [history[-1] for _, history in self.baseline_true_choice_score_trace[question_number].items()]
            original_baseline_for_false = [history[-1] for _, history in self.baseline_false_choice_score_trace[question_number].items()]
            scores = self.MC_calcs(
                original_baseline_for_true, original_baseline_for_false, 
                self.question_ture_label_list[question_number], self.question_best_correct_answer_dict[question_number]
            )
            baseline_result_dict['total_mc1'] += scores['MC1'] / len(self.question_number_list)
            baseline_result_dict['total_mc2'] += scores['MC2'] / len(self.question_number_list)
            baseline_result_dict['total_mc3'] += scores['MC3'] / len(self.question_number_list)

        print("verify baseline_result_dict", baseline_result_dict)

    def MC_calcs(self, scores_true, scores_false, ref_true, ref_best):
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

    def run_kfold(self, n_splits=5):

        average_delta_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
        average_baseline_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
        average_finetuned_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for fold, (question_number_list_to_train, _) in enumerate(kf.split(
            np.array(self.question_number_list).reshape(-1, 1)
        )):
            model = self.train(question_number_list_to_train)
            total = len(self.question_number_list) - len(question_number_list_to_train)
            baseline_result_dict, finetuned_result_dict, delta_dict = self.evaluate_model(model, question_number_list_to_train)

            for k in ['total_mc1', 'total_mc2', 'total_mc3']:
                average_baseline_dict[k] +=  baseline_result_dict[k] * total / len(self.question_number_list)
                average_finetuned_dict[k] += finetuned_result_dict[k] * total / len(self.question_number_list)
                average_delta_dict[k] += (finetuned_result_dict[k] -  baseline_result_dict[k]) * total / len(self.question_number_list)
        print("summary")
        print(pd.DataFrame([
            average_baseline_dict, 
            average_finetuned_dict,
            average_delta_dict
        ], index=["Original", "FineTuned", "Delta"]))

    def train(self, question_number_list_to_train=None, path_to_save=None):
        if question_number_list_to_train is None:
            question_number_list_to_train = self.question_number_list
        true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.true_choice_score_trace[question_number].items()]
        false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.false_choice_score_trace[question_number].items()]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        model.fit(X, y)

        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred))

        if path_to_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
            model.save(path_to_save)


        return model


    def evaluate_model(self, model, exclude_question_indices=None):
        if exclude_question_indices is None:
            exclude_question_indices = []
        
        total = len(self.question_number_list) - len(exclude_question_indices)
        baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
        finetuned_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

        for question_number in self.question_number_list:
            if question_number in exclude_question_indices:
                continue

            ################################################################################
            original_baseline_for_true = [history[-1] for _, history in self.true_choice_score_trace[question_number].items()]
            original_baseline_for_false = [history[-1] for _, history in self.false_choice_score_trace[question_number].items()]
            scores = self.MC_calcs(
                original_baseline_for_true, original_baseline_for_false, 
                self.question_ture_label_list[question_number], self.question_best_correct_answer_dict[question_number]
            )
            baseline_result_dict['total_mc1'] += scores['MC1'] / total
            baseline_result_dict['total_mc2'] += scores['MC2'] / total
            baseline_result_dict['total_mc3'] += scores['MC3'] / total
            ################################################################################
            X_for_true = np.array([history for _, history in self.true_choice_score_trace[question_number].items()])
            X_for_false = np.array([history for _, history in self.false_choice_score_trace[question_number].items()])
            log_proba_prediction_for_true = model.predict_log_proba(X_for_true)[:,1]
            log_proba_prediction_for_false = model.predict_log_proba(X_for_false)[:,1]

            scores = self.MC_calcs(
                log_proba_prediction_for_true, log_proba_prediction_for_false, 
                self.question_ture_label_list[question_number], self.question_best_correct_answer_dict[question_number]
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

        return baseline_result_dict, finetuned_result_dict, delta_dict

    def evaluate(self, model_path):
        model = EggachecatClassifier.load(model_path)
        return self.evaluate_model(model=model)


class FACTOR_WIKI_Analysist:
    def __init__(self, history_json_path, agg_func):
        with open(history_json_path, "r") as fp:
            self.observation_json = json.load(fp)
        self.agg_func = agg_func
        self.question_number_list = list(range(len(self.observation_json['modle_true_outputs'])))

    def run_kfold(self, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        improvment_history = []

        for fold, (question_number_list_to_train, _) in enumerate(kf.split(
            np.array(self.question_number_list).reshape(-1, 1)
        )):
            print("fold", fold)
            model = self.train(question_number_list_to_train)

            result = self.evaluate_model(model=model, exclude_question_indices=question_number_list_to_train)
            result['n_fold'] = fold
            improvment_history.append(result)

        print(pd.DataFrame(improvment_history))


    def evaluate_model(self, model, baseline_score=None, exclude_question_indices=None):
        if exclude_question_indices is None:
            exclude_question_indices = {}

        result_list = []
        baseline_result_list = []
        for i, (true_outputs, false_outputs, model_completion) in tqdm.tqdm(enumerate(zip(
            self.observation_json['modle_true_outputs'],  
            self.observation_json['modle_false_outputs'],
            self.observation_json['model_completion']
        )), total=len(self.question_number_list)):
            if i in exclude_question_indices:
                continue

            original_true, orignal_false_list = model_completion[0], model_completion[1:]
            if min([original_true]) > max(orignal_false_list):
                baseline_result_list.append(1)
            else:
                baseline_result_list.append(0)

            log_proba_prediction_for_true = model.predict_log_proba([self.agg_func(history) for history in true_outputs])[:,1]
            log_proba_prediction_for_false = model.predict_log_proba([self.agg_func(history) for history in false_outputs])[:,1]
            if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
                result_list.append(1)
            else:
                result_list.append(0)

        if baseline_score is None:
            baseline_score = sum(baseline_result_list) / len(baseline_result_list)

        new_score = sum(result_list) / len(result_list)
        print(f"baseline(with trained on truthfulqa): {baseline_score}")
        print(f"new_score: {new_score}")
        print(f"delta: {new_score - baseline_score}")

        return {"acc": new_score - baseline_score}

    def train(self, question_number_list_to_train=None, path_to_save=None):
        if question_number_list_to_train is None:
            question_number_list_to_train = self.question_number_list

        true_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_true_outputs'][i]]
        false_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_false_outputs'][i]]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        print(sum(y), len(y))
        model.fit(X, y)

        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred))

        if path_to_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
            model.save(path_to_save)

        return model

    def evaluate(self, model_path, baseline_score=None):
        model = EggachecatClassifier.load(model_path)
        return self.evaluate_model(model=model, baseline_score=baseline_score)




wsFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def main():
    def agg_func(multi_token_history):
        multi_token_history = np.array(multi_token_history)
        # print(multi_token_history[:,-1])
        # print(np.argmin(multi_token_history[:,-1]))
        return multi_token_history[np.argmin(multi_token_history[:,-1])]
        # return np.cumsum(multi_token_history, axis=0)
        # return np.array(multi_token_history).mean(axis=0)

    truthfulqa_model = f"{wsFolder}/saves/models/chair/truthfulqa/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_tfqa_mc_eval_classifer.pkl"
    truthfulqa_analyist = TruthfulQA_Analyist(
        history_json_path=f"{wsFolder}/saves/evaluation/chair/truthfulqa/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_tfqa_mc_eval.json",
        agg_func=agg_func
    )
    truthfulqa_analyist.run_kfold(n_splits=2)

    # truthfulqa_analyist.train(path_to_save=truthfulqa_model)

    # factor_qa_model = f"{wsFolder}/saves/models/chair/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_factor_eval_classifer.pkl"
    # factor_analysist = FACTOR_WIKI_Analysist(
    #     history_json_path=f"{wsFolder}/saves/evaluation/chair/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_factor_eval.json",
    #     agg_func=agg_func
    # )
    # factor_analysist.train(path_to_save=factor_qa_model)
    # factor_analysist.run_kfold(n_splits=2)
    
    # for model in [truthfulqa_model, factor_qa_model]:
    #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     print("Evaluating....", truthfulqa_model)
    #     for analyist in [truthfulqa_analyist, factor_analysist]:
    #         print("ON", truthfulqa_analyist)
    #         analyist.evaluate(model)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    


if __name__ == "__main__":
    main()
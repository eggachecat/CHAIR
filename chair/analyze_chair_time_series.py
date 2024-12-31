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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import time

RANDOM_STATE = 42

class TimeSeriesDataset(Dataset):
    """用于处理变长时间序列的自定义 Dataset"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# class LSTMClassifier(nn.Module):
#     """定义 LSTM 模型结构"""
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(LSTMClassifier, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x, lengths):
#         # 按长度排序，使用 pack_padded_sequence 加速计算
#         packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (hn, _) = self.lstm(packed_input)  # 取最后一个时间步的隐藏状态
#         out = self.fc(hn[-1])  # 全连接层映射到类别数
#         return out

class RNNClassifier(nn.Module):
    """RNN 时间序列分类模型"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hn = self.rnn(packed_input)  # 最后一层隐藏状态
        out = self.fc(hn[-1])  # 全连接层
        return out

from collections import Counter

def print_label_distribution(dataset, name=""):
    """统计并打印数据集中每个类别的数量"""
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"{name} Dataset Label Distribution: {dict(label_counts)}")


class EggachecatClassifier:
    def __init__(self):
        self.model = RNNClassifier(input_size=34, hidden_size=100, num_layers=1, num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


        # 打印模型的总可训练参数数量
        self._print_trainable_params()

    def _compute_class_weights(self, labels):
        """计算类别权重"""
        class_counts = np.bincount(labels)
        weights = len(labels) / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def _print_trainable_params(self):
        """打印模型的总可训练参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def fit(self, token_score_records_list, y, epochs=1, batch_size=32, val_split=0.2):
        X = [self.extract_features(x) for x in token_score_records_list]
        dataset = TimeSeriesDataset(X, y)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print_label_distribution(train_dataset, "Train")
        print_label_distribution(val_dataset, "Validation")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        class_weights = self._compute_class_weights(y)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)
            elapsed_time = time.time() - start_time

            print(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {elapsed_time:.2f}s"
            )
            print("-" * 50)


    def _train_one_epoch(self, dataloader):
        """训练一个 epoch"""
        total_loss = 0
        for inputs, targets, lengths in dataloader:
            outputs = self.model(inputs, lengths)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets, lengths in dataloader:
                outputs = self.model(inputs, lengths)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    @staticmethod
    def collate_fn(batch):
        """用于 DataLoader 的 collate_fn 处理变长序列"""
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences])
        sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_sequences, labels, lengths

    def predict_log_proba(self, sequences):
        """返回对数概率"""
        self.model.eval()
        with torch.no_grad():
            inputs, _, lengths = self.collate_fn([(seq, 0) for seq in sequences])  # 伪标签
            outputs = self.model(inputs, lengths)
            log_probs = torch.log_softmax(outputs, dim=1)
        return log_probs

    def predict(self, token_score_records_list):
        log_probs = self.predict_log_proba(token_score_records_list)
        return torch.argmax(log_probs, dim=1).numpy()
    
    def extract_features(self, token_score_records):
        return token_score_records
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
            token_score_records,
            delta_degree_1,
            delta_degree_2,
            [
                last_score,
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
        torch.save(self.model.state_dict(), output_path)


    @classmethod
    def load(cls, output_path) -> 'EggachecatClassifier':
        instance = cls()
        instance.model.load_state_dict(torch.load(output_path))
        instance.model.eval()
        return instance

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
        X = [*false_choice_history_list, *true_choice_history_list]
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
        return np.array(multi_token_history)#.mean(axis=0)

    # truthfulqa_model = f"{wsFolder}/saves/models/chair/truthfulqa/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_tfqa_mc_eval_classifer.pkl"
    # truthfulqa_analyist = TruthfulQA_Analyist(
    #     history_json_path=f"{wsFolder}/saves/evaluation/chair/truthfulqa/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_tfqa_mc_eval.json",
    #     agg_func=agg_func
    # )
    # truthfulqa_analyist.run_kfold(n_splits=2)

    # truthfulqa_analyist.train(path_to_save=truthfulqa_model)

    factor_qa_model = f"{wsFolder}/saves/models/chair/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_factor_eval_classifer.pkl"
    factor_analysist = FACTOR_WIKI_Analysist(
        history_json_path=f"{wsFolder}/saves/evaluation/chair/factor_eval/wiki_factor/Meta-Llama-3-8B-Instruct/baseline_and_observe_layers_factor_eval.json",
        agg_func=agg_func
    )
    factor_analysist.train(path_to_save=factor_qa_model)
    factor_analysist.run_kfold(n_splits=2)
    
    # for model in [truthfulqa_model, factor_qa_model]:
    #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     print("Evaluating....", truthfulqa_model)
    #     for analyist in [truthfulqa_analyist, factor_analysist]:
    #         print("ON", truthfulqa_analyist)
    #         analyist.evaluate(model)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    


if __name__ == "__main__":
    main()
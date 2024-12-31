import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.svm import SVC
import random
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import tqdm
import json
import urllib.parse
import matplotlib.gridspec as gridspec

"""
TODO:
    - 小测试的样本
    - debug factor_wiki
    - 每个数据集抽2个: 对于三个数据集都有提升
    - 去重
"""

RANDOM_STATE = 42
MODEL_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_NAME = None
# MODEL_NAME = "llama-7b"
# MODEL_NAME = "Llama-2-7b-hf"
# MODEL_NAME = "Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Llama-2-7b-hf"

HYPER_PARAMETER = {}
HYPER_PARAMETER = {
    "history_start": 18,
    "history_end": -1,
    "delta_with_last": False
}


WS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def hyper_parameter_to_folder_name():
    print('HYPER_PARAMETER', HYPER_PARAMETER)
    json_str = json.dumps(HYPER_PARAMETER, separators=(',', ':'))
    encoded_str = urllib.parse.quote(json_str, safe='')
    return encoded_str

def folder_name_to_hyper_parameter(folder_name):
    return urllib.parse.unquote(folder_name)
    # encoded_str = folder_name
    # json_str = urllib.parse.unquote(encoded_str)
    # return json.loads(json_str)

PAPER_FOLDER_NAME = None


def plot_true_false_subgraphs(true_data, false_data, file_name):
    # 设置基础行高
    base_height = 1
    # 获取所有子图的最大 token 数量，以便对齐左右高度
    max_token_length = max(max(len(trace_data) for trace_data in true_data.values()),
                           max(len(trace_data) for trace_data in false_data.values()))

    # 创建网格布局，分别为 True 和 False 数据创建独立的 GridSpec
    fig = plt.figure(figsize=(30, len(true_data) * max_token_length * base_height))
    gs_true = gridspec.GridSpec(len(true_data), 1, left=0.05, right=0.45)
    gs_false = gridspec.GridSpec(len(false_data), 1, left=0.55, right=0.95)
    
    # 获取所有值的最小值和最大值，用于统一颜色范围
    all_values = []
    for data in [true_data, false_data]:
        for trace_data in data.values():
            all_values.extend([value for values in trace_data.values() for value in values])
    vmin, vmax = min(all_values), max(all_values)
    
    # 绘制 True 答案的热力图（左侧）
    for i, (answer_key, trace_data) in enumerate(true_data.items()):
        tokens = list(trace_data.keys())
        history_values = np.array(list(trace_data.values()))

        # 如果 token 数量不足最大长度，则填充空行
        if len(tokens) < max_token_length:
            padding = max_token_length - len(tokens)
            history_values = np.vstack([history_values, np.full((padding, history_values.shape[1]), np.nan)])
            tokens.extend([''] * padding)  # 空标签填充
        
        ax = fig.add_subplot(gs_true[i, 0])
        im = ax.imshow(history_values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(history_values.shape[1]))
        ax.set_xticklabels([f'Layer {j+1}' for j in range(history_values.shape[1])], rotation=90)
        ax.set_yticks(np.arange(max_token_length))
        ax.set_yticklabels(tokens)
        ax.set_title(f"True Answer: {answer_key}")

    # 绘制 False 答案的热力图（右侧）
    for j, (answer_key, trace_data) in enumerate(false_data.items()):
        tokens = list(trace_data.keys())
        history_values = np.array(list(trace_data.values()))

        # 如果 token 数量不足最大长度，则填充空行
        if len(tokens) < max_token_length:
            padding = max_token_length - len(tokens)
            history_values = np.vstack([history_values, np.full((padding, history_values.shape[1]), np.nan)])
            tokens.extend([''] * padding)  # 空标签填充
        
        ax = fig.add_subplot(gs_false[j, 0])
        im = ax.imshow(history_values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(history_values.shape[1]))
        ax.set_xticklabels([f'Layer {j+1}' for j in range(history_values.shape[1])], rotation=90)
        ax.set_yticks(np.arange(max_token_length))
        ax.set_yticklabels(tokens)
        ax.set_title(f"False Answer: {answer_key}")

    # 添加颜色条并放置在图的右侧
    cbar = fig.colorbar(im, ax=fig.get_axes(), orientation="vertical", label="Score")
    # plt.subplots_adjust(top=0.95, bottom=0.05)

    plt.savefig(f"./output/{file_name}.png")

# def plot_heatmap(trace_data):
#     tokens = list(trace_data.keys())
#     history_values = np.array(list(trace_data.values()))

#     # 绘制热力图
#     plt.figure(figsize=(10, len(tokens) * 0.5))
#     plt.imshow(history_values, aspect='auto', cmap='viridis')

#     # 设置x轴为层数，y轴为token
#     plt.xticks(ticks=np.arange(history_values.shape[1]), labels=[f'Layer {i+1}' for i in range(history_values.shape[1])], rotation=90)
#     plt.yticks(ticks=np.arange(len(tokens)), labels=tokens)

#     # 添加颜色条
#     plt.colorbar(label="Score")

#     plt.xlabel("Layers")
#     plt.ylabel("Tokens")
#     plt.title("Token History Trace by Layer")
#     plt.tight_layout()
#     plt.savefig("./output/demo.png")

class EggachecatClassifier:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', penalty='l2', random_state=RANDOM_STATE, max_iter=1000)
        # self.model = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, max_depth=5)
        # self.model = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        # self.model = SVC(class_weight='balanced', gamma='auto', probability=True)

        self.feature_labels = [
                'last_score',
                'mean', 
                'max_v', 
                'min_v', 
                'std', 
                'slope',
                'delta_std_degree_1',
                'delta_std_degree_2'
        ]
        self.score_scalar = 300

    def fit(self, token_score_records_list, y):
        X = [self.extract_features(x) for x in token_score_records_list]
        self.model.fit(X, y)

    def predict_log_proba(self, token_score_records_list):
        return self.model.predict_log_proba([self.extract_features(x) for x in token_score_records_list]) *  self.score_scalar

    def predict(self, token_score_records_list):
        return self.model.predict([self.extract_features(x) for x in token_score_records_list])

    @classmethod
    def extract_features(cls, token_score_records):
        """
        从单条时间序列中提取统计和趋势特征，并拼接原始值：
        - 均值、标准差、最大值、最小值、斜率
        - 拼接时间序列的原始值
        """
        start, end = HYPER_PARAMETER['history_start'], HYPER_PARAMETER['history_end']
        last_score = token_score_records[-1]
        token_score_records = [x for x in token_score_records[start:end]]

        # token_score_records = [x / last_score for x in token_score_records[18:-1]]
        # token_score_records = [x / last_score for x in token_score_records[18:-1]]

        max_v = max(token_score_records)
        min_v = min(token_score_records)
        slope = np.polyfit(range(len(token_score_records)), token_score_records, 1)[0]  # 线性拟合斜率
        mean = np.mean(token_score_records)
        std = np.std(token_score_records)

        # delta_degree_1 = []
        # for i in range(len(token_score_records)-1):
        #     delta_degree_1.append(token_score_records[i+1]-token_score_records[i])
        # delta_std_degree_1 = np.std(delta_degree_1)

        # delta_degree_2 = []
        # for i in range(len(token_score_records)-2):
        #     delta_degree_2.append(token_score_records[i+2]-token_score_records[i])
        # delta_std_degree_2 = np.std(delta_degree_2)

        # delta_degree_3 = []
        # for i in range(len(token_score_records)-3):
        #     delta_degree_3.append(token_score_records[i+3]-token_score_records[i])
        # delta_std_degree_3 = np.std(delta_degree_3)

        # 拼接原始值和提取的特征
        features = np.hstack([
            # token_score_records[start:end],
            # delta_degree_1[start:end],
            # delta_degree_2[start:end],
            # delta_degree_3,
            [
                last_score,
                mean, 
                max_v, 
                min_v, 
                std, 
                slope,
                # delta_std_degree_1,
                # delta_std_degree_2,
                # delta_std_degree_3
            ]
        ])
        features = (max(features) - features) / (max(features) - min(features))
        features[np.isnan(features)] = 0

        return features
    
    def print_importance(self):
        return
        feature_importance = sorted(
            enumerate(self.model.coef_[0]),  # [(index, coef), ...]
            key=lambda x: abs(x[1]),    # 根据系数绝对值排序
            reverse=True                # 从大到小排序
        )
        print("Sorted Feature Importance:")
        for index, coef in feature_importance:
            print(f"Feature <{self.feature_labels[index]}>: Coefficient = {coef:.4f}")

    def save(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, output_path) -> 'EggachecatClassifier':
        with open(output_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model

def max_str(s, max_len=50):
    if len(s)< max_len:
        return s
    return s[:max_len] + "..."

def plot_trace(true_trace_list, false_trace_list, title, ture_label_list, false_label_list, description, save_path):
    plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 24})

    # 绘制第一组线（绿色）
    for i, line in enumerate(true_trace_list):
        x_values = range(len(line))  # x 值为索引
        plt.plot(x_values, line, color='green', label=max_str(ture_label_list[i]), marker="o")

    # 绘制第二组线（红色）
    for i, line in enumerate(false_trace_list):
        x_values = range(len(line))  # x 值为索引
        plt.plot(x_values, line, color='red', label=max_str(false_label_list[i]), marker="x")

    # 设置图形细节
    plt.xlabel('Layer Index')
    plt.ylabel('Logtis\' score')
    plt.title(f"{title}\n{description}")
    # plt.suptitle(description, y=0.9, fontsize=10)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

class TruthfulQA_Analyst:
    def __init__(self, history_json_path, agg_func):
        self.history_json_path = history_json_path
        self.question_number_list = list(range(817))
        self.true_choice_score_trace = {}
        self.false_choice_score_trace = {}
        self.question_description_dict = {}
        self.question_ture_label_dict = {}
        self.question_false_label_dict = {}
        self.question_best_correct_answer_dict = {}
        self.question_best_correct_answer_index = {}

        self.baseline_true_choice_score_trace = {}
        self.baseline_false_choice_score_trace = {}

        self._raw_baseline_true_choice_score_trace = {}
        self._raw_baseline_false_choice_score_trace = {}

        self.agg_func = agg_func 

        self.prepare_dataset()

    def prepare_dataset(self):

        with open(self.history_json_path, "r") as fp:
            layer_json = json.load(fp)

        for question_number, info in enumerate(layer_json['question']):
            self.question_description_dict[question_number] = info['question']
            self.question_ture_label_dict[question_number] = info['answer_true'].split("; ")
            self.question_false_label_dict[question_number] = info['answer_false'].split("; ")
            self.question_best_correct_answer_dict[question_number] = info['answer_best']
            self.question_best_correct_answer_index[question_number] = self.question_ture_label_dict[question_number].index(
                info['answer_best']
            )

            if question_number not in self.true_choice_score_trace:
                self.true_choice_score_trace[question_number] = {}
                self.baseline_true_choice_score_trace[question_number] = {}
                self._raw_baseline_true_choice_score_trace[question_number] = {}

            if question_number not in self.false_choice_score_trace:
                self.false_choice_score_trace[question_number] = {}
                self.baseline_false_choice_score_trace[question_number] = {}
                self._raw_baseline_false_choice_score_trace[question_number] = {}

        model_scores = layer_json['model_scores']
        for question_number in self.question_number_list:
            for choice, multi_token_history in enumerate(model_scores[question_number]['scores_true_tokens_trace']):
                self.true_choice_score_trace[question_number][choice] = self.agg_func(multi_token_history)
                self.baseline_true_choice_score_trace[question_number][choice] = np.array(multi_token_history).sum(axis=0)
                self._raw_baseline_true_choice_score_trace[question_number][choice] = multi_token_history

            for choice, multi_token_history in enumerate(model_scores[question_number]['scores_false_tokens_trace']):
                self.false_choice_score_trace[question_number][choice] = self.agg_func(multi_token_history)
                self.baseline_false_choice_score_trace[question_number][choice] = np.array(multi_token_history).sum(axis=0)
                self._raw_baseline_false_choice_score_trace[question_number][choice] = multi_token_history


        baseline_result_dict = {'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

        for question_number in self.question_number_list:
            original_baseline_for_true = [history[-1] for _, history in self.baseline_true_choice_score_trace[question_number].items()]
            original_baseline_for_false = [history[-1] for _, history in self.baseline_false_choice_score_trace[question_number].items()]
            scores = self.MC_calcs(
                original_baseline_for_true, original_baseline_for_false, 
                self.question_ture_label_dict[question_number], self.question_best_correct_answer_dict[question_number]
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
            baseline_result_dict, finetuned_result_dict, delta_dict = self.evaluate_model(model, question_number_list_to_train, return_full_metrics=True)

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


    def get_Xy(self, question_number_list_to_train):
        true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.true_choice_score_trace[question_number].items()]
        false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.false_choice_score_trace[question_number].items()]
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        return X, y

    def train(self, question_number_list_to_train=None, path_to_save=None):
        if question_number_list_to_train is None:
            question_number_list_to_train = self.question_number_list
      
        X, y = self.get_Xy(question_number_list_to_train)

        model = EggachecatClassifier()
        model.fit(X, y)

        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred))

        model.print_importance()
        
        # test_true_choice_history_list = [history for question_number in self.question_number_list for _, history in self.true_choice_score_trace[question_number].items() if question_number not in question_number_list_to_train]
        # test_false_choice_history_list = [history for question_number in self.question_number_list for _, history in self.false_choice_score_trace[question_number].items() if question_number not in question_number_list_to_train]

        # X_test = np.vstack([test_false_choice_history_list, test_true_choice_history_list])
        # y_test = np.array([0] * len(test_false_choice_history_list) + [1] * len(test_true_choice_history_list))
        # y_pred = model.predict(X_test)
        # print("[Test]Accuracy:", accuracy_score(y_test, y_pred))
        # print("\n[Test]Classification Report:\n", classification_report(y_test, y_pred))


        if path_to_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
            model.save(path_to_save)


        return model


    def run_few_supervised_once(self, n_observation=10):
        question_number_list_to_train = random.sample(self.question_number_list, n_observation)
        true_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.true_choice_score_trace[question_number].items()]
        false_choice_history_list = [history for question_number in question_number_list_to_train for _, history in self.false_choice_score_trace[question_number].items()]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        model.fit(X, y)

        return self.evaluate_model(model, exclude_question_indices=question_number_list_to_train)


    def run_few_supervised(self, n_observation=10, times=100):
        delta_dict_list = []
        for _ in tqdm.tqdm(range(times), desc="experiment"):
            delta_dict_list.append(self.run_few_supervised_once(n_observation=n_observation))
        
        df = pd.DataFrame(delta_dict_list)
        print(df)
        output_csv_path = f"{WS_FOLDER}/saves/experiment/chair/truthfulqa/{MODEL_NAME}/run_few_supervised_{n_observation}_{times}.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df

    def evaluate_model(self, model, exclude_question_indices=None, return_full_metrics=False):
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
                self.question_ture_label_dict[question_number], self.question_best_correct_answer_dict[question_number]
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
                self.question_ture_label_dict[question_number], self.question_best_correct_answer_dict[question_number]
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

        if return_full_metrics:
            return baseline_result_dict, finetuned_result_dict, delta_dict

        return delta_dict

    def evaluate(self, model_path):
        print("evaluate@TruthfulQA")
        model = EggachecatClassifier.load(model_path)
        return self.evaluate_model(model=model)


    def plot_question(self, question_number, model_path=None):
        if model_path is None:
            model_path = f"{WS_FOLDER}/downloaded_models/{MODEL_NAME}"
        df = pd.read_csv(f"{WS_FOLDER}/TruthfulQA/TruthfulQA.csv")
        print(df)
        # print("self.question_description_dict[question_number]", self.question_description_dict[question_number])
        # print("self.question_ture_label_list[question_number]", self.question_ture_label_list[question_number])
        # print("self.question_false_label_dict[question_number]", self.question_false_label_dict[question_number])
        # print("self.question_best_correct_answer_dict[question_number]", self.question_best_correct_answer_dict[question_number])
        # print("self.question_best_correct_answer_index[question_number] ", self.question_best_correct_answer_index[question_number] )
        # print("self.baseline_true_choice_score_trace[question_number] ", self.baseline_true_choice_score_trace[question_number] )
        # print("self.baseline_false_choice_score_trace[question_number] ", self.baseline_false_choice_score_trace[question_number] )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        token_with_trace_data_true = {}
        for k, trace in self._raw_baseline_true_choice_score_trace[question_number].items():
            tokens_list = [t.replace('Ġ', '_') for t in tokenizer.tokenize(
                self.question_ture_label_dict[question_number][k]
            )]
            token_with_trace_data_true[k] = {t: h for t, h in zip(tokens_list, trace)}

        token_with_trace_data_false = {}
        for k, trace in self._raw_baseline_false_choice_score_trace[question_number].items():
            tokens_list = [t.replace('Ġ', '_') for t in tokenizer.tokenize(
                self.question_false_label_dict[question_number][k]
            )]
            token_with_trace_data_false[k] = {t: h for t, h in zip(tokens_list, trace)}

        plot_true_false_subgraphs(token_with_trace_data_true, token_with_trace_data_false)


        # true_str_list = [' ' + df.loc[question_index, 'completion']]
        # 
        # true_outputs = self.observation_json['modle_true_outputs'][question_index]
        # for true_str, true_output, true_tokens in zip(true_str_list, true_outputs, true_tokens_list):
        #     print(true_str)
        #     for history, true_token in zip(true_output, true_tokens):
        #         print(true_token, "->", [v for v in history])
        #     print("--------------------")

        # false_str_list = [' ' + df.loc[question_index, f'contradiction_{i}'] for i in range(3)]
        # false_tokens_list = [[t.replace('Ġ', '_') for t in tokenizer.tokenize(s)] for s in false_str_list]
        # false_outputs =  self.observation_json['modle_false_outputs'][question_index]

        # for false_str, false_output, false_tokens in zip(false_str_list, false_outputs, false_tokens_list):
        #     print(false_str)
        #     for history, false_token in zip(false_output, false_tokens):
        #         print(false_token, "->", [v for v in history])
        #     print("--------------------")

class FACTOR_WIKI_Analyst:
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

    def plot_question(self, question_index, model_path=None):
        if model_path is None:
            model_path = f"{WS_FOLDER}/downloaded_models/Meta-Llama-3-8B-Instruct"

        df = pd.read_csv(f"{WS_FOLDER}/saves/wiki_factor/wiki_factor.csv")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        
        token_with_trace_data_true = {}
        true_str_list = [' ' + df.loc[question_index, 'completion']]
        true_tokens_list = [[t.replace('Ġ', '_') for t in tokenizer.tokenize(s)] for s in true_str_list]
        true_outputs = self.observation_json['modle_true_outputs'][question_index]
        for i, (true_str, true_output, true_tokens) in enumerate(zip(true_str_list, true_outputs, true_tokens_list)):
            print(true_str)
            for history, true_token in zip(true_output, true_tokens):
                print(true_token, "->", [v for v in history])
            print("--------------------")
            token_with_trace_data_true[i] = {t: h for t, h in zip(true_tokens, true_output)}


        token_with_trace_data_false = {}
        false_str_list = [' ' + df.loc[question_index, f'contradiction_{i}'] for i in range(3)]
        false_tokens_list = [[t.replace('Ġ', '_') for t in tokenizer.tokenize(s)] for s in false_str_list]
        false_outputs =  self.observation_json['modle_false_outputs'][question_index]

        for i, (false_str, false_output, false_tokens) in enumerate(zip(false_str_list, false_outputs, false_tokens_list)):
            print(false_str)
            for history, false_token in zip(false_output, false_tokens):
                print(false_token, "->", [v for v in history])
            print("--------------------")
            token_with_trace_data_false[i] = {t: h for t, h in zip(false_tokens, false_output)}

        plot_true_false_subgraphs(token_with_trace_data_true, token_with_trace_data_false, f"factor_wiki_{question_index}")


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

            log_proba_prediction_for_true = model.predict_log_proba([np.array(history).sum(axis=0).tolist() for history in true_outputs])[:,1]
            log_proba_prediction_for_false = model.predict_log_proba([np.array(history).sum(axis=0).tolist() for history in false_outputs])[:,1]
            if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
                result_list.append(1)
            else:
                result_list.append(0)

        if baseline_score is None:
            baseline_score = sum(baseline_result_list) / len(baseline_result_list)

        new_score = sum(result_list) / len(result_list)
        print(f"baseline: {baseline_score}")
        print(f"new_score: {new_score}")
        print(f"delta: {new_score - baseline_score}")

        return {"acc": new_score - baseline_score}

    def get_Xy(self, question_number_list_to_train):
        true_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_true_outputs'][i]]
        false_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_false_outputs'][i]]
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))

        return X, y
   

    def train(self, question_number_list_to_train=None, path_to_save=None):
        if question_number_list_to_train is None:
            question_number_list_to_train = self.question_number_list

        X, y = self.get_Xy(question_number_list_to_train)
        model = EggachecatClassifier()
        print(sum(y), len(y))
        model.fit(X, y)

        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred))

        model.print_importance()
        
        # test_true_choice_history_list = [self.agg_func(multi_token_history) for i in self.question_number_list for multi_token_history in self.observation_json['modle_true_outputs'][i] if i not in question_number_list_to_train]
        # test_false_choice_history_list = [self.agg_func(multi_token_history) for i in self.question_number_list for multi_token_history in self.observation_json['modle_false_outputs'][i] if i not in question_number_list_to_train]
        # X_test = np.vstack([test_false_choice_history_list, test_true_choice_history_list])
        # y_test = np.array([0] * len(test_false_choice_history_list) + [1] * len(test_true_choice_history_list))
        # y_pred = model.predict(X_test)
        # print("[Test]Accuracy:", accuracy_score(y_test, y_pred))
        # print("\n[Test]Classification Report:\n", classification_report(y_test, y_pred))
        
        if path_to_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
            model.save(path_to_save)

        return model

    def evaluate(self, model_path, baseline_score=None):
        print("evaluate@FACTOR_WIKI")
        model = EggachecatClassifier.load(model_path)
        return self.evaluate_model(model=model, baseline_score=baseline_score)

    def run_few_supervised_once(self, n_observation=10):
        question_number_list_to_train = random.sample(self.question_number_list, n_observation)
        true_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_true_outputs'][i]]
        false_choice_history_list = [self.agg_func(multi_token_history) for i in question_number_list_to_train for multi_token_history in self.observation_json['modle_false_outputs'][i]]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        model.fit(X, y)

        return self.evaluate_model(model, exclude_question_indices=question_number_list_to_train)


    def run_few_supervised(self, n_observation=10, times=100):
        delta_dict_list = []
        for _ in tqdm.tqdm(range(times), desc=f"experiment_{n_observation}"):
            delta_dict_list.append(self.run_few_supervised_once(n_observation=n_observation))
        
        df = pd.DataFrame(delta_dict_list)
        print(df)
        output_csv_path = f"{WS_FOLDER}/saves/experiment/chair/truthfulqa/{MODEL_NAME}/run_few_supervised_{n_observation}_{times}.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return df


class MMLU_Analyst:
    def __init__(self, history_json_path, agg_func):
        self.history_json_path = history_json_path
        self.agg_func = agg_func

        with open(self.history_json_path, "r") as fp:
            self.observation_json = json.load(fp)
        print(list(self.observation_json.keys()))
        self.question_number_list = list(range(len(self.observation_json['content'])))

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

    def get_Xy(self, question_number_list_to_train):
        true_choice_history_list = []
        false_choice_history_list = []

        for i, (content, correct_ans, choices_layer_history) in enumerate(zip(
            self.observation_json['content'],  
            self.observation_json['truth'],
            self.observation_json['layer_history']
        )):
            if i not in question_number_list_to_train:
                continue
            correct_index = {"A":0, "B":1, "C":2, "D":3}[correct_ans]
            true_choice_history_list.extend([history for i, history in enumerate(choices_layer_history) if i == correct_index])
            false_choice_history_list.extend([history for i, history in enumerate(choices_layer_history) if i != correct_index])

        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))

        return X, y

    def train(self, question_number_list_to_train=None, path_to_save=None):
        if question_number_list_to_train is None:
            question_number_list_to_train = self.question_number_list

        X, y = self.get_Xy(question_number_list_to_train)
        model = EggachecatClassifier()
        print(sum(y), len(y))
        model.fit(X, y)

        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred))

        model.print_importance()

        if path_to_save is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_to_save)), exist_ok=True)
            model.save(path_to_save)

        return model


    def evaluate_model(self, model, baseline_score=None, exclude_question_indices=None):
        if exclude_question_indices is None:
            exclude_question_indices = {}

        result_list = []
        baseline_result_list = []
        for i, (content, correct_ans, choices_layer_history) in tqdm.tqdm(enumerate(zip(
            self.observation_json['content'],  
            self.observation_json['truth'],
            self.observation_json['layer_history']
        )), total=len(self.question_number_list)):
            if i in exclude_question_indices:
                continue
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

        if baseline_score is None:
            baseline_score = sum(baseline_result_list) / len(baseline_result_list)

        new_score = sum(result_list) / len(result_list)
        print(f"baseline: {baseline_score}")
        print(f"new_score: {new_score}")
        print(f"delta: {new_score - baseline_score}")

        return {"acc": new_score - baseline_score}


    def evaluate(self, model_path):
        print("evaluate@MMLU")
        model = EggachecatClassifier.load(model_path)
        return self.evaluate_model(model=model)

def viz_performance_matrix(performance_matrix):
    rows = []
    for train_on, test_dict in performance_matrix.items():
        for test_on, metric_dict in test_dict.items():
            for metric, value in metric_dict.items():
                rows.append({
                    "train_on": train_on, 
                    "test_on": test_on,
                    "metric": metric,
                    "value": value
                })
    df = pd.DataFrame(rows).pivot_table(
        index="train_on", 
        columns=["test_on", "metric"], 
        values="value"
    )
    print(df)
    return df

def make_extract_feature_func():

    @classmethod
    def extract_features(cls, token_score_records):
        """
        从单条时间序列中提取统计和趋势特征，并拼接原始值：
        - 均值、标准差、最大值、最小值、斜率
        - 拼接时间序列的原始值
        """
        # last_score = token_score_records[-1]
        token_score_records = [x for x in token_score_records[28:]]

        # token_score_records = [x / last_score for x in token_score_records[18:-1]]
        # token_score_records = [x / last_score for x in token_score_records[18:-1]]

        max_v = max(token_score_records)
        min_v = min(token_score_records)
        slope = np.polyfit(range(len(token_score_records)), token_score_records, 1)[0]  # 线性拟合斜率
        mean = np.mean(token_score_records)
        std = np.std(token_score_records)

        # delta_degree_1 = []
        # for i in range(len(token_score_records)-1):
        #     delta_degree_1.append(token_score_records[i+1]-token_score_records[i])
        # delta_std_degree_1 = np.std(delta_degree_1)

        # delta_degree_2 = []
        # for i in range(len(token_score_records)-2):
        #     delta_degree_2.append(token_score_records[i+2]-token_score_records[i])
        # delta_std_degree_2 = np.std(delta_degree_2)


        # 拼接原始值和提取的特征
        features = np.hstack([
            token_score_records,
            # delta_degree_1,
            # delta_degree_2,
            [
                # last_score,
                mean, 
                max_v, 
                min_v, 
                std, 
                slope,
                # delta_std_degree_1,
                # delta_std_degree_2
            ]
        ])
        features = (max(features) - features) / (max(features) - min(features))
        features[np.isnan(features)] = 0

        return features

    return extract_features

def agg_func(multi_token_history):
    return np.array(multi_token_history).sum(axis=0)

def main_performance_matrix_search(extract_features_func=None):

    if extract_features_func is not None:
        EggachecatClassifier.extract_features = extract_features_func

    output_path = f"{WS_FOLDER}/saves/paper/{MODEL_NAME}/{PAPER_FOLDER_NAME}/performance_matrix.json"

    model_and_dataset = [
        {
            "cls": MMLU_Analyst,
            "model_path": f"{WS_FOLDER}/saves/models/chair/mmlu_test/{MODEL_NAME}/{PAPER_FOLDER_NAME}/baseline_and_observe_layers_mmlu_test_0_shot_eval_classifer.pkl",
            "history_json_path": f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_0_mmlu_layer_observation/layer-observation-0.json",
            "key": "mmlu_test@0-shot",
            "analyst": None
        },
        {
            "cls": MMLU_Analyst,
            "model_path": f"{WS_FOLDER}/saves/models/chair/mmlu_test/{MODEL_NAME}/{PAPER_FOLDER_NAME}/baseline_and_observe_layers_mmlu_test_1_shot_eval_classifer.pkl",
            "history_json_path": f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_1_mmlu_layer_observation/layer-observation-1.json",
            "key": "mmlu_test@1-shot",
            "analyst": None
        },
        {
            "cls": TruthfulQA_Analyst,
            "model_path": f"{WS_FOLDER}/saves/models/chair/truthfulqa/{MODEL_NAME}/{PAPER_FOLDER_NAME}/baseline_and_observe_layers_tfqa_mc_eval_classifer.pkl",
            "history_json_path": f"{WS_FOLDER}/saves/evaluation/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval.json",
            "key": "TruthfulQA",
            "analyst": None
        },
        {
            "cls": FACTOR_WIKI_Analyst,
            "model_path": f"{WS_FOLDER}/saves/models/chair/factor_eval/wiki_factor/{MODEL_NAME}/{PAPER_FOLDER_NAME}/baseline_and_observe_layers_factor_eval_classifer.pkl",
            "history_json_path": f"{WS_FOLDER}/saves/evaluation/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval.json",
            "key": "FACTOR_WIKI",
            "analyst": None
        }
    ]

    for obj in model_and_dataset:
        _analyst = obj['cls'](history_json_path=obj['history_json_path'], agg_func=agg_func) 
        _analyst.train(path_to_save=obj['model_path'])
        obj['analyst'] = _analyst

    performance_matrix = {}
    for obj_train in model_and_dataset:
        row_key = obj_train['key']
        performance_matrix[row_key] = {}
        for obj_test in model_and_dataset:
            col_key = obj_test['key']
            performance_matrix[row_key][col_key] = obj_test['analyst'].evaluate(obj_train['model_path'])


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(performance_matrix, fp)
    viz_performance_matrix(performance_matrix)
    return performance_matrix

def run_kfold(n_splits=5):

    mmlu_n_shot = 0
    mmlu_analyst_0_shot = MMLU_Analyst(
        history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_{mmlu_n_shot}_mmlu_layer_observation/layer-observation-{mmlu_n_shot}.json",
        agg_func=agg_func
    )
    mmlu_analyst_0_shot.run_kfold(n_splits=n_splits)

    mmlu_n_shot = 1
    mmlu_analyst_1_shot = MMLU_Analyst(
        history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_{mmlu_n_shot}_mmlu_layer_observation/layer-observation-{mmlu_n_shot}.json",
        agg_func=agg_func
    )
    mmlu_analyst_1_shot.run_kfold(n_splits=n_splits)

    truthfulqa_analyst = TruthfulQA_Analyst(
        history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval.json",
        agg_func=agg_func
    )
    truthfulqa_analyst.run_kfold(n_splits=n_splits)

    factor_analyst = FACTOR_WIKI_Analyst(
        history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval.json",
        agg_func=agg_func
    )
    factor_analyst.run_kfold(n_splits=n_splits)
    

def main():
    global MODEL_NAME, PAPER_FOLDER_NAME, HYPER_PARAMETER

    for hyper_parameter in [
        {
            "history_start": 16,
            "history_end": -1,
            "delta_with_last": False
        },
        {
            "history_start": 0,
            "history_end": -1,
            "delta_with_last": False
        },
        {
            "history_start": 16,
            "history_end": 32,
            "delta_with_last": False
        },
        {
            "history_start": 0,
            "history_end": 32,
            "delta_with_last": False
        },
        {
            "history_start": 0,
            "history_end": 20,
            "delta_with_last": False
        },
        {
            "history_start": 5,
            "history_end": 18,
            "delta_with_last": False
        }
        
    ]:
        for k, v in hyper_parameter.items():
            HYPER_PARAMETER[k] = v
        PAPER_FOLDER_NAME = hyper_parameter_to_folder_name()



        model_name_list = [
            "Meta-Llama-3-8B-Instruct",
            "Llama-2-7b-hf",
            "Mistral-7B-Instruct-v0.3"
        ]
        performance_matrix_list = []
        for model_name in model_name_list:
            MODEL_NAME = model_name
            performance_matrix_list.append(main_performance_matrix_search())
        
        for model_name, performance_matrix in zip(model_name_list, performance_matrix_list):
            print("model_name", model_name)
            print("performance matrix")
            df = viz_performance_matrix(performance_matrix)
            csv_path = f"{WS_FOLDER}/saves/paper/{PAPER_FOLDER_NAME}/{model_name}/performance_matrix.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path)
            print("=" * 20)


    # run_kfold(n_splits=2)


def viz_all_result():
    for dirpath, dirnames, filenames in os.walk(f"{WS_FOLDER}/saves/paper/by_hyperparameter"):
        hyper_folder, model_name = dirpath.split("/")[-2:]
        if filenames:
            print('hyper_parameter', folder_name_to_hyper_parameter(hyper_folder))
            print('model_name', model_name)
            df = pd.read_csv(os.path.join(dirpath, filenames[0]))
            print(df)
            print('-' * 40)

def plot_question():
    # truthful_qa_analyst = TruthfulQA_Analyst(
    #     history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval.json",
    #     agg_func=agg_func
    # )
    # truthful_qa_analyst.plot_question(0)

    factor_analyst = FACTOR_WIKI_Analyst(
        history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval.json",
        agg_func=agg_func
    )
    for i in range(100):
        factor_analyst.plot_question(i)

def _cross_train(n_observation, n_times):
    result_record_list = []
    for _ in tqdm.tqdm(range(n_times), desc=f"Runnkng exp@{n_observation}"):
        X_list, y_list = [], []
        mmlu_n_shot = 0
        mmlu_analyst_0_shot = MMLU_Analyst(
            history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_{mmlu_n_shot}_mmlu_layer_observation/layer-observation-{mmlu_n_shot}.json",
            agg_func=agg_func
        )
        _X, _y = mmlu_analyst_0_shot.get_Xy(random.sample(mmlu_analyst_0_shot.question_number_list, n_observation))
        X_list.extend(_X)
        y_list.extend(_y)

        mmlu_n_shot = 1
        mmlu_analyst_1_shot = MMLU_Analyst(
            history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_{mmlu_n_shot}_mmlu_layer_observation/layer-observation-{mmlu_n_shot}.json",
            agg_func=agg_func
        )
        _X, _y = mmlu_analyst_1_shot.get_Xy(random.sample(mmlu_analyst_1_shot.question_number_list, n_observation))
        X_list.extend(_X)
        y_list.extend(_y)

        truthfulqa_analyst = TruthfulQA_Analyst(
            history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval.json",
            agg_func=agg_func
        )
        _X, _y = truthfulqa_analyst.get_Xy(random.sample(truthfulqa_analyst.question_number_list, n_observation))
        X_list.extend(_X)
        y_list.extend(_y)

        factor_analyst = FACTOR_WIKI_Analyst(
            history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval.json",
            agg_func=agg_func
        )
        _X, _y = factor_analyst.get_Xy(random.sample(factor_analyst.question_number_list, n_observation))
        X_list.extend(_X)
        y_list.extend(_y)

        model = EggachecatClassifier()
        model.fit(np.vstack(X_list), np.vstack(y_list))
        cross_train_model_path = f"{WS_FOLDER}/saves/models/chair/cross_train/{MODEL_NAME}/cross_train.pkl"
        model.save(cross_train_model_path)

        result_record = {}
        for name, analyst in zip(
            ["mmlu_analyst_0_shot", "mmlu_analyst_1_shot", "truthfulqa", "factort"],
            [mmlu_analyst_0_shot, mmlu_analyst_1_shot, truthfulqa_analyst, factor_analyst]
        ):
            result_record[name] = analyst.evaluate(cross_train_model_path)

        result_record_list.append(result_record)

        output_path = f"{WS_FOLDER}/saves/paper/cross_join/{MODEL_NAME}_{n_observation}_{n_times}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fp:
            json.dump(result_record_list, fp)

def cross_train():
    global MODEL_NAME
    n_times = 50

    model_name_list = [
        "Meta-Llama-3-8B-Instruct",
        "Llama-2-7b-hf",
        "Mistral-7B-Instruct-v0.3"
    ]
    for n_observation in [1, 3, 5, 7]:
        for model_name in model_name_list:
            MODEL_NAME = model_name
            _cross_train(n_observation=n_observation, n_times=n_times)

if __name__ == "__main__":
    cross_train()
    # plot_question()
    # run_kfold(n_splits=2)



# def main_performance_matrix_search(extract_features_func):
#     EggachecatClassifier.extract_features = extract_features_func
#     def agg_func(multi_token_history):
#         return np.array(multi_token_history).sum(axis=0)

#     output_path = f"{WS_FOLDER}/saves/paper/performance_matrix.json"
#     # with open(output_path, "r") as fp:
#     #     performance_matrix = json.load(fp)
#     # viz_performance_matrix(performance_matrix)
#     # print(performance_matrix)
#     # print(pd.DataFrame(performance_matrix).transpose())
#     # return

#     run_cross = True
#     mmlu_model = f"{WS_FOLDER}/saves/models/chair/mmlu_test/{MODEL_NAME}/baseline_and_observe_layers_mmlu_test_eval_classifer.pkl"
    # mmlu_n_shot = 0
    # mmlu_analyst = MMLU_Analyst(
    #     history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/mmlu_test/{MODEL_NAME}/mmlu_test_baseline_shot_{mmlu_n_shot}_mmlu_layer_observation/layer-observation-{mmlu_n_shot}.json",
    #     agg_func=agg_func
    # )
#     if run_cross:
#         mmlu_analyst.train(path_to_save=mmlu_model)
#     # mmlu_analyst.run_kfold(2)
    
#     truthfulqa_model = f"{WS_FOLDER}/saves/models/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval_classifer.pkl"
#     truthfulqa_analyst = TruthfulQA_Analyst(
#         history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/truthfulqa/{MODEL_NAME}/baseline_and_observe_layers_tfqa_mc_eval.json",
#         agg_func=agg_func
#     )
#     # for n_observation in [
#     #     5, 10, 20, 30, 80, 100
#     # ]:
#     #     truthfulqa_analyst.run_few_supervised(n_observation=n_observation)
#     if run_cross:
#         truthfulqa_analyst.train(path_to_save=truthfulqa_model)
#     # else:
#     #     truthfulqa_analyst.run_kfold(n_splits=2)

#     factor_qa_model = f"{WS_FOLDER}/saves/models/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval_classifer.pkl"
#     factor_analyst = FACTOR_WIKI_Analyst(
#         history_json_path=f"{WS_FOLDER}/saves/evaluation/chair/factor_eval/wiki_factor/{MODEL_NAME}/baseline_and_observe_layers_factor_eval.json",
#         agg_func=agg_func
#     )
#     # for n_observation in [
#     #     5, 10, 20, 30, 80, 100
#     # ]:
#     #     factor_analyst.run_few_supervised(n_observation=n_observation)

#     # # factor_analyst.plot_question(0)
#     if run_cross:
#         factor_analyst.train(path_to_save=factor_qa_model)
#     # else:
#     #     factor_analyst.run_kfold(n_splits=2)
    

#     if run_cross:

#         performance_matrix = {}
#         for row_key, model in [
#                 ("mmlu_model", mmlu_model), 
#                 ("truthfulqa_model", truthfulqa_model), 
#                 ("factor_qa_model", factor_qa_model)
#             ]:
#             performance_matrix[row_key] = {}
#             print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#             print("Evaluating....", truthfulqa_model)
#             for col_key, analyst in [
#                 ("mmlu_analyst", mmlu_analyst), 
#                 ("truthfulqa_analyst", truthfulqa_analyst), 
#                 ("factor_analyst", factor_analyst)
#             ]:
#                 print("ON", truthfulqa_analyst)
#                 performance_matrix[row_key][col_key] = analyst.evaluate(model)
#                 print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         output_path = f"{WS_FOLDER}/saves/paper/performance_matrix.json"
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, "w") as fp:
#             json.dump(performance_matrix, fp)
#         viz_performance_matrix(performance_matrix)

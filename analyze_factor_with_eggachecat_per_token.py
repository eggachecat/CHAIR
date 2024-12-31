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
from tqdm import tqdm

wsFolder = os.path.dirname(os.path.abspath(__file__))
base_folder = f"{wsFolder}/saves/evaluation/dola-oberservation-layer/Meta-Llama-3-8B-Instruct"


def plot_trace(true_trace_list, false_trace_list, title, ture_label_list, false_label_list, description, subfolder="absolute_logits"):
    plt.figure(figsize=(20, 15))

    # 绘制第一组线（绿色）
    for i, line in enumerate(true_trace_list):
        x_values = range(len(line))  # x 值为索引
        plt.plot(x_values, line, color='green', label=ture_label_list[i])

    # 绘制第二组线（红色）
    for i, line in enumerate(false_trace_list):
        x_values = range(len(line))  # x 值为索引
        plt.plot(x_values, line, color='red', label=false_label_list[i])

    # 设置图形细节
    plt.xlabel('X-axis (Layer)')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.suptitle(description, y=0.92, fontsize=10)
    plt.legend(loc='best')

    plt.grid(True)
    output_folder = os.path.join("./saves/figures/analyze_factor", subfolder)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.close()

def draw():
    random.seed(42)
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)

    for i, (true_outputs, false_outputs) in enumerate(zip(observation_json['modle_true_outputs'],  observation_json['modle_false_outputs'])):
        if i > 100:
            break
        plot_trace(
            [np.array(history).min(axis=0).tolist() for history in true_outputs], 
            [np.array(history).min(axis=0).tolist() for history in false_outputs], 
            f"Question_{i}",
            ture_label_list=[f"right_{i}" for i in range(len(true_outputs))],
            false_label_list=[f"wrong_{i}" for i in range(len(false_outputs))],
            description="wtf"
        )

def main():
    random.seed(42)
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
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

    # false_choice_history_list = false_choice_history_list[:5000]
    # true_choice_history_list = true_choice_history_list[:5000]
    X = np.vstack([false_choice_history_list, true_choice_history_list])
    y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
    model = EggachecatClassifier()

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    model.print_importance()
    model.save("./saves/models/eggachecat/trained_from_factor.pkl")

    # y_pred = model.predict(X_train)
    # print("Accuracy:", accuracy_score(y_train, y_pred))
    # print("\nClassification Report:\n", classification_report(y_train, y_pred))

    y_pred = model.predict(X_test)
    print(len(y_test), y_test.sum())
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def run_kfold(n_splits=5):

    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)

    # agg_func = 'std'

    def agg_func(multi_token_history):
        multi_token_history = np.array(multi_token_history)
        return multi_token_history[
            np.argsort(np.sum(multi_token_history, axis=1))[:99999]
        ].mean(axis=0)
        # return np.array(multi_token_history).mean(axis=0) #/ np.array(multi_token_history).std(axis=0)

    question_number_list = list(range(len(observation_json['modle_true_outputs'])))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (question_number_list_to_train, _) in enumerate(kf.split(
        np.array(question_number_list).reshape(-1, 1)
    )):
        print("fold", fold)

        true_choice_history_list = [
            agg_func(history).tolist() 
            for question_number in question_number_list_to_train 
            for history in observation_json['modle_true_outputs'][question_number]
        ]
        false_choice_history_list = [
           agg_func(history).tolist() 
            for question_number in question_number_list_to_train 
            for history in observation_json['modle_false_outputs'][question_number]
        ]
        
        model = EggachecatClassifier()
        X = np.vstack([false_choice_history_list, true_choice_history_list])
        y = np.array([0] * len(false_choice_history_list) + [1] * len(true_choice_history_list))
        model.fit(X, y)

        result_list = []
        expected_result_list = []

        for question_number in question_number_list:
            if question_number in question_number_list_to_train:
                continue

            true_outputs = observation_json['modle_true_outputs'][question_number]
            false_outputs  = observation_json['modle_false_outputs'][question_number]
            model_completion  = observation_json['model_completion'][question_number]
            original_true, orignal_false_list = model_completion[0], model_completion[1:]
            if min([original_true]) > max(orignal_false_list):
                expected_result_list.append(1)
            else:
                expected_result_list.append(0)

            log_proba_prediction_for_true = model.predict_log_proba([agg_func(history).tolist() for history in true_outputs])[:,1]
            log_proba_prediction_for_false = model.predict_log_proba([agg_func(history).tolist() for history in false_outputs])[:,1]
            if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
                result_list.append(1)
            else:
                result_list.append(0)
        print("original", sum(expected_result_list) / len(expected_result_list))
        print("now", sum(result_list) / len(result_list))

            


def evaluate_model(path_to_save_model):
    model = EggachecatClassifier.load(path_to_save_model)
    json_path = f"{wsFolder}/saves/evaluation/factor_eval/wiki_factor/output-path-factor-wiki-dola-eggachecat-observe-layers.json"
    with open(json_path, "r") as fp:
        observation_json = json.load(fp)
    print(list(observation_json.keys()))
    result_list = []
    expected_result_list = []

    # def __predict(outputs):
    #     return np.array([
    #         np.median(log_proba_tokens)
    #         # np.array(log_proba_tokens).median()
    #         for log_proba_tokens in
    #         [model.predict_log_proba(np.array(history))[:,1] for history in outputs]
    #     ])

    def min_sum_subarray(arr, x):
        if len(arr) < x:
            raise ValueError("数组长度必须大于或等于区间长度。")

        # 初始窗口的和
        min_sum = np.sum(arr[:x])
        min_index = 0  # 记录最小和区间的起始索引
        current_sum = min_sum

        # 滑动窗口遍历数组
        for i in range(x, len(arr)):
            # 更新窗口的和：减去左边的数，加上右边的数
            current_sum = current_sum - arr[i - x] + arr[i]

            # 如果当前窗口的和更小，更新最小和和起始索引
            if current_sum < min_sum:
                min_sum = current_sum
                min_index = i - x + 1
        return min_sum
        # 返回最小和的区间及其和
        # return arr[min_index:min_index + x], min_sum

    def __predict(outputs):
        return np.array([
            min_sum_subarray(log_proba_tokens, 3)
            for log_proba_tokens in
            [model.predict_log_proba(np.array(history))[:,1] for history in outputs]
        ])

    for i, (true_outputs, false_outputs, model_completion) in tqdm(enumerate(zip(
        observation_json['modle_true_outputs'],  
        observation_json['modle_false_outputs'],
        observation_json['model_completion']
    )), total=len(observation_json['modle_true_outputs'])):
        original_true, orignal_false_list = model_completion[0], model_completion[1:]
        if min([original_true]) > max(orignal_false_list):
            expected_result_list.append(1)
        else:
            expected_result_list.append(0)
        # return
        log_proba_prediction_for_true = __predict(true_outputs)
        log_proba_prediction_for_false = __predict(false_outputs)


        if log_proba_prediction_for_true.min() > log_proba_prediction_for_false.max():
            result_list.append(1)
        else:
            result_list.append(0)
    print("baseline_result_list", sum(expected_result_list) / len(expected_result_list))
    print("now_result_list", sum(result_list) / len(result_list))

if __name__ == "__main__":
    # evaluate("./saves/models/eggachecat/trained_from_factor.pkl")
    evaluate_model("./saves/models/eggachecat/trained_from_truthfulqa.pkl")
    # run_kfold(2)
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

wsFolder = os.path.dirname(os.path.abspath(__file__))
base_folder = f"{wsFolder}/saves/evaluation/dola-oberservation-layer/Meta-Llama-3-8B-Instruct"


def load_pivot_table(field):
    layer_list = list(range(1, 33))
    row_list = []
    for layer_number in layer_list:
        with open(f"{base_folder}/layer-{layer_number}.json", "r") as fp:
            layer_json = json.load(fp)

        for question_number, record in enumerate(layer_json['model_scores']):

            if "." not in field:
                value = record[field]
            else:
                value = record
                for k in field.split("."):
                    value = value[k if not k.isdigit() else int(k)]

            row_list.append({
                "layer_number": layer_number,
                "question_number": question_number,
                field: value
            })

    df_pivot = pd.DataFrame(row_list).pivot(
        index='question_number',
        columns='layer_number',
        values=field
    )

    output_path = os.path.join(
        "./saves/analysis", "_".join(field.split(".")) + ".csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_pivot.to_csv(output_path, index=False)

    return df_pivot


def main():
    for field in [
        "MC1",
        "MC2",
        "MC3",
        "scores-true.0",
        "scores-true.1",
        "scores-false.0",
        "scores-false.1",
    ]:
        print(load_pivot_table(field))


# main()
layer_list = list(range(1, 33))
question_list = list(range(817))


def calculate_MC1_upperbound_with_select_best():
    df = load_pivot_table("MC1").reset_index()
    ctr_total = 0
    ctr_could_right = 0
    candidate_layer_list = [*layer_list]
    for _, row in df.iterrows():
        ctr_total += 1
        for layer_number in candidate_layer_list:
            if row.iloc[layer_number] > 0:
                ctr_could_right += 1
                break
    print(ctr_could_right, ctr_total, ctr_could_right / ctr_total)


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
    output_folder = os.path.join("./saves/figures/analyze_truthfulqa", subfolder)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.close()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def prepare_data(lanes1, lanes2):
    """
    将 lanes1 和 lanes2 数据合并，并生成标签：
    - lanes1 的标签为 0
    - lanes2 的标签为 1
    """
    X = np.array(lanes1 + lanes2)  # 合并数据
    y = np.array([0] * len(lanes1) + [1] * len(lanes2))  # 生成标签
    return X, y

def train_and_evaluate_model(X, y):
    """
    训练分类模型并进行评估。
    """
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 初始化分类器（使用随机森林）
    model = RandomForestClassifier(random_state=42)
    # model = LogisticRegression(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 进行预测
    y_pred = model.predict(X_test)

    # 输出准确率和分类报告
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

def change_stuff():
    layer_list = list(range(1, 33))
    true_choice_score_trace = {}
    false_choice_score_trace = {}
    question_number_list = list(range(817))
    question_description_dict = {}
    question_ture_label_list = {}
    question_false_label_dict = {}

    with open(f"{base_folder}/layer-1.json", "r") as fp:
        layer_json = json.load(fp)

    for question_number, info in enumerate(layer_json['question']):
        question_description_dict[question_number] = info['question']
        question_ture_label_list[question_number] = info['answer_true'].split("; ")
        question_false_label_dict[question_number] = info['answer_false'].split("; ")


    for layer_number in layer_list:
        with open(f"{base_folder}/layer-{layer_number}.json", "r") as fp:
            layer_json = json.load(fp)

        for question_number in question_number_list:

            if question_number not in true_choice_score_trace:
                true_choice_score_trace[question_number] = {}

            if question_number not in false_choice_score_trace:
                false_choice_score_trace[question_number] = {}


            print('layer', layer_number)
            output = layer_json['model_scores'][question_number]


            for choice, score in enumerate(output['scores-true']):
                true_choice_score_trace[question_number].setdefault(
                    choice, []).append(score)

            for choice, score in enumerate(output['scores-false']):
                false_choice_score_trace[question_number].setdefault(
                    choice, []).append(score)

            # print('\tscores-true', output['scores-true'],
            #       'min', min(output['scores-true']))
            # print('\tscores-false',
            #       output['scores-false'], 'max', max(output['scores-false']))
            # print('--------------------')

    # for question_number in question_number_list:
    #     for choice, history in true_choice_score_trace[question_number].items():
    #         print(history)

    # for question_number in question_number_list:
    #     for choice, history in false_choice_score_trace[question_number].items():
    #         print(history)

    true_choice_history_list = [history for question_number in question_number_list for _, history in true_choice_score_trace[question_number].items()]
    false_choice_history_list = [history for question_number in question_number_list for _, history in false_choice_score_trace[question_number].items()]

    # 准备数据
    X, y = prepare_data(true_choice_history_list, false_choice_history_list)

    # 训练和评估模型
    model = train_and_evaluate_model(X, y)


    
    # for question_number in question_number_list:
    #     plot_trace(
    #         [[(x-history[i-1])/history[i-1] if i != 0 else 0 for i,x in enumerate(history[1:])] for _, history in true_choice_score_trace[question_number].items()],
    #         [[(x-history[i-1])/history[i-1] if i != 0 else 0 for i,x in enumerate(history[1:])] for _, history in false_choice_score_trace[question_number].items()],
    #         title=f"Question_{question_number}",
    #         description=question_description_dict[question_number],
    #         ture_label_list=question_ture_label_list[question_number],
    #         false_label_list=question_false_label_dict[question_number],
    #         subfolder="relative_change_with_previous"
    #     )

    # print("question_true_choice_score_through_layer_history", true_choice_score_trace)
    # print("question_false_choice_score_through_layer_history", false_choice_score_trace)


change_stuff()
# calculate_MC_upperbound_with_select_best("MC1")
# calculate_MC_upperbound_with_select_best("MC2")
# calculate_MC_upperbound_with_select_best("MC3")

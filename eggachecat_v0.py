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
import pickle

class EggachecatClassifier:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', penalty='l2', random_state=42, max_iter=1000)
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
        max_v = max(token_score_records)
        min_v = min(token_score_records)
        slope = np.polyfit(range(len(token_score_records)), token_score_records, 1)[0]  # 线性拟合斜率
        mean = np.mean(token_score_records)
        std = np.std(token_score_records)
        # 拼接原始值和提取的特征
        features = np.hstack([
            [

                mean, 
                max_v, 
                min_v, 
                std, 
                slope
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

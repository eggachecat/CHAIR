import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import urllib

WS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

root_dir = f'{WS_FOLDER}/saves/paper/run_few_supervised'


def draw_robustness_of_single_dataset():

    cherry_pick_result = []

    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for hyperparam_encode in os.listdir(dataset_path):
            hyperparam_path = os.path.join(dataset_path, hyperparam_encode)
            if not os.path.isdir(hyperparam_path):
                continue
            for model_name in os.listdir(hyperparam_path):
                model_path = os.path.join(hyperparam_path, model_name)
                if not os.path.isdir(model_path):
                    continue
                n_times_path = os.path.join(model_path, 'n_times_50')
                if not os.path.isdir(n_times_path):
                    continue
                
                # 用于存储特定组合的所有 sample_size 和每个指标数据
                data = []
                
                for file in os.listdir(n_times_path):
                    if file.startswith('run_few_supervised_') and file.endswith('.csv'):
                        # 提取 sample 数量
                        sample_size = int(file.split('_')[-1].split('.')[0])
                        # if sample_size < 5:
                        #     continue
                        file_path = os.path.join(n_times_path, file)
                        
                        # 读取文件
                        df = pd.read_csv(file_path)
                        df['Sample_Size'] = sample_size  # 添加 sample_size 列
                        data.append(df)  # 将 DataFrame 添加到数据列表中
                
                # 检查是否有数据
                if data:
                    # 合并所有数据到一个 DataFrame
                    combined_df = pd.concat(data, ignore_index=True)
                    
                    # 针对每个列生成 boxplot
                    for column in combined_df.columns:
                        if column == 'Sample_Size':
                            continue  # 跳过 sample_size 列
                        
                        plt.figure(figsize=(8, 6))
                        sns.set(style="whitegrid")

                        # sns.boxplot(data=combined_df, x='Sample_Size', y=column)
                        sns.boxplot(x='Sample_Size', y=column, showfliers=False, data=combined_df, boxprops=dict(color=sns.color_palette("Paired")[1], alpha=0.5), medianprops=dict(color='red', linewidth=2, alpha=0.5))
                        sns.stripplot(x='Sample_Size', y=column, data=combined_df, jitter=True, alpha=0.5)
                        import json
                        hyperparam_dict = json.loads(urllib.parse.unquote(hyperparam_encode))

                        plt.title(f'Boxplot of performance improvement of {column} by Sample Size')
                        plt.tight_layout()
                        # plt.title(f'Boxplot of {column} by Sample Size\nDataset: {dataset}, Model: {model_name}, Hyperparam: {hyperparam_dict}')
                        plt.xlabel('Sample Size')
                        plt.ylabel(column)
                        
                        # 保存图表
                        folder_name = "_".join([f"{k}-{hyperparam_dict[k]}" for k in sorted(hyperparam_dict.keys())])

                        output_path = f"./output/{folder_name}/boxplot_{model_name}_{dataset}_{column}.pdf"
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        plt.savefig(output_path)
                        plt.close()


draw_robustness_of_single_dataset()
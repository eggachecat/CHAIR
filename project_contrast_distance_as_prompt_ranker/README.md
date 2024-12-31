# Steps

0. 选择一个数据集
1. 基于数据集的描述，指定模型生成prompt
2. 挑选几个example并对prompt进行排序
    - 对这几个example使用pyvene提取attention
    - 使用contrast decoding进行打分
    - 得到一个排名A
3. 对于每个prompt进行评估
    - 真的跑benchmark
    - 得到一个排名B
4. 比较 排名A 和 排名B
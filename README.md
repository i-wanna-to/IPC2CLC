## 基于孪生BERT网络的科技文献类目映射

《计算机研究与发展》人工智能专题论文开源数据和代码，完整项目持续更新...

> This code is based on [sentence-transformers (0.4.1.2)](https://github.com/UKPLab/sentence-transformers)


### 数据说明（训练数据\验证数据\测试数据）

1. 实验数据存储在 `dataset/` 目录下；
2. `dataset/` 目录中包括：三个损失函数所需的训练数据（ *_train.txt）\验证数据（ *_valid.txt）\测试数据（ *_test.txt）；
3. 在三个损失函数中，除排序损失函数外，其他两个损失函数所需的验证数据在训练过程中自动划分1/5的训练数据出来作为验证数据。


### 组织结构说明：

1. `error_log` 目录在实验过程中自动创建，保存对应模型映射错误的类目；
2. `output` 目录在实验过程中自动创建，保存训练好的模型和参数权重；
3. `sentence-transformers` 目录为修改过部分源码的 sentence-transformers 库。


### 训练和映射

#### 若您没有可供使用的GPU设备，可使用[Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) 提供的免费GPU设备

注: 测试环境为 `torch==1.8.0+cu101`

#### 安装依赖

> pip install -r requirements.txt

#### sia-BERT模型和sia-BERT-Zero模型

> 执行 `python run.py`

#### sia-Multi模型和sia-Multi-Zero模型

> 1. 修改配置文件 `model_args.json` 中的 `model_name`
> 2. 执行 `python run.py`


### 各模型的实验结果（%）

| 模型 | A | B | C | D | E | F | G | H | AVE | 10^-2×VAR |
| :--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Category-Sim | 60.47 | 56.50 | 57.76 | 54.05 | 66.67 | 44.96 | 65.32 | 47.66 | 56.30 | |
| Sia-Multi-Zero | 55.04 | 49.72 | 57.14 | 36.49 | 52.56 | 55.04 | 60.48 | 64.06 | 54.70 | |
| Sia-BERT-Zero | 62.79 | 52.54 | 59.01 | 37.84 | 41.03 | 61.24 | 55.65 | 56.25 | 54.90 | |
| Bi-TextCNN | 81.40 | 79.66 | 80.75 | 71.62 | 89.74 | 86.82 | 77.42 | 75.00 | 80.30 |16.86 |
| Sia-Multi | 90.70 | 86.44 | 90.06 | 79.73 | 93.59 | 89.92 | 90.32 | 88.28 | 88.80 |5.56 |
| TextCNN | 89.92 | 89.83 | 94.41 | 87.84 | 97.44 | 93.02 | 92.74 | 87.50 | 91.50 |1.70 |
| Bi-LSTM | 92.25 | 92.09 | 92.55 | 85.14 | 98.72 | 92.25 | 94.35 | 88.28 | 92.00 |0.50 |
| T-Encoder | 93.02 | 92.66 | 94.41 | 86.49 | 98.72 | 92.25 | 95.97 | 85.16 | 92.40 |1.74 |
| Sia-BERT | 94.57 | 94.35 | 94.41 | 87.84 | 98.72 | 96.12 | 94.35 | 90.63 | 94.00 |1.10 |

注：实验评价指标为准确率和方差


### 引用和作者（Citing & Authors）
如果您使用了我们的实验数据或者代码，请引用（If you use the code, feel free to cite our publication [IPC2CLC](https://github.com/i-wanna-to/IPC2CLC/)）:
``` 
@article{何贤敏:1751,
author = {何贤敏,李茂西,何彦青},
title = {基于孪生BERT网络的科技文献类目映射},
publisher = {计算机研究与发展},
year = {2021},
journal = {计算机研究与发展},
volume = {58},
number = {8},
eid = {1751},
numpages = {9},
pages = {1751},
keywords = {国际专利分类法;中国图书馆分类法;基于孪生BERT网络;类目映射;对比损失},
url = {https://crad.ict.ac.cn/CN/abstract/article_4478.shtml},
doi = {10.7544/issn1000-1239.2021.20210323}
}    
```

如果您使用了我们发布的实验数据和代码，请先与我们先取得联系，获得同意后再使用！
联系邮箱: 何贤敏, xianminhe@jxnu.edu.cn

如果有疑问可以给我发送电子邮件或者发布一个 issue。（Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.）

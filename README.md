## 基于Siamese BERT网络的科技文献IPC和CLC类目映射研究

《计算机研究与发展》人工智能专题在投论文开源数据和代码，完整项目持续更新...

> This code is based on [sentence-transformers (0.4.1.2)](https://github.com/UKPLab/sentence-transformers)


### 数据说明：训练数据\验证数据\测试数据

1. 实验所需的数据存储在 `data_KFold_1/` 目录下，1000条原始数据（raw data）存储在 `data_KFold_1/raw_data/` 目录下；

2. 除基于匹配计数的类目相似度实验和零样本迁移学习实验外，其余实验采用五折交叉验证，未处理的五折交叉验证数据存储在 `data_KFold_1/origin/` 目录下，每一折划分了训练数据（XXX_train.txt）和测试数据（XXX_valid.txt），训练时的验证集将从训练数据中划分出来（验证集占训练集的1/5）；

3. 根据 `data_KFold_1/origin/` 目录下未处理的五折交叉验证数据生成不同损失函数训练时所需的训练集和验证集，这些数据集分别存储在 `data_KFold_1/sample_ContrastiveLoss/`、`data_KFold_1/sample_MultipleNegativesRankingLoss/` 和 `data_KFold_1/sample_TripletLoss/` 目录下；


### 训练和映射（代码暂时只提供.ipynb文件，之后会陆续更新为.py文件）

#### 若您没有可供使用的GPU设备，可使用[Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) 提供的免费GPU设备

注: 测试环境为 `torch==1.8.0+cu101`

#### sia-BERT模型和sia-BERT-Zero模型

> `si_model.ipynb`

#### sia-Multi模型和sia-Multi-Zero模型

> `si_model.ipynb`

#### Bi-TextCNN模型

> `Bilinear_CNN.ipynb`

#### TextCNN模型

> `Text_CNN.ipynb`

#### Bi-LSTM模型

> `Bi_LSTM.ipynb`

#### T-Encoder模型

> `Transformer_Encoder.ipynb`

#### Category-Sim模型

> `ipc_to_clc.ipynb`


### 各模型的实验结果

| 模型 | A | B | C | D | E | F | G | H | AVE |
| :--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Category-Sim | 60.47 | 56.50 | 57.76 | 54.05 | 66.67 | 44.96 | 65.32 | 47.66 | 56.30 |
| Sia-Multi-Zero | 55.04 | 49.72 | 57.14 | 36.49 | 52.56 | 55.04 | 60.48 | 64.06 | 54.70 |
| Sia-BERT-Zero | 62.79 | 52.54 | 59.01 | 37.84 | 41.03 | 61.24 | 55.65 | 56.25 | 54.90 |
| Bi-TextCNN | 81.40 | 79.66 | 80.75 | 71.62 | 89.74 | 86.82 | 77.42 | 75.00 | 80.30 |
| Sia-Multi | 90.70 | 86.44 | 90.06 | 79.73 | 93.59 | 89.92 | 90.32 | 88.28 | 88.80 |
| TextCNN | 89.92 | 89.83 | 94.41 | 87.84 | 97.44 | 93.02 | 92.74 | 87.50 | 91.50 |
| Bi-LSTM | 92.25 | 92.09 | 92.55 | 85.14 | 98.72 | 92.25 | 94.35 | 88.28 | 92.00 |
| T-Encoder | 93.02 | 92.66 | 94.41 | 86.49 | 98.72 | 92.25 | 95.97 | 85.16 | 92.40 |
| Sia-BERT | 94.57 | 94.35 | 94.41 | 87.84 | 98.72 | 96.12 | 94.35 | 90.63 | 94.00 |

注：实验评价指标为准确率(%)


### 引用和作者（Citing & Authors）
如果您使用了我们的实验数据或者代码，请引用（If you use the code, feel free to cite our publication [IPC2CLC](https://github.com/i-wanna-to/IPC2CLC/)）:
``` 
@article{XXX-2021-XXX,
    title = "基于Siamese BERT网络的科技文献IPC和CLC类目映射研究",
    author = "XXX, XXX, XXX, ...", 
    journal= "XXX",
    month = "X",
    year = "2021",
    url = "https://github.com/i-wanna-to/IPC2CLC/",
}
```

如果您使用了我们发布的实验数据和代码，请先与我们先取得联系，获得同意后再使用！
联系邮箱: XXX, X@X.edu.cn （基于盲审要求，暂时需要匿名。）

如果有疑问可以给我发送电子邮件或者发布一个 issue。（Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.）

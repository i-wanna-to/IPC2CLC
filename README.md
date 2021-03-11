# IPC2CLC

基于预训练语言模型的孪生网络的IPC与CLC类目映射

> This code is based on [sentence-transformers (0.4.1.2)](https://github.com/UKPLab/sentence-transformers)

## 执行（Installation）

1. `pip install -r requirements.txt`
2. `python run.py`

注意（NOTE）: test in `torch==1.8.0+cu101`

## 训练数据\验证数据\测试数据

1. 实验所需的数据存储在 `data_KFold_1/` 目录下，1000条原始数据（raw data）存储在 `data_KFold_1/raw_data/` 目录下；

2. 基于类目相似度的实验数据存储在 `data_KFold_1/category_similarity_data/` 目录下；

3. 除类目相似度实验外，其余实验采用五折交叉验证，未处理的五折交叉验证数据存储在 `data_KFold_1/origin/` 目录下，每一折划分了训练数据（XXX_train.txt）和测试数据（XXX_test.txt），训练时的验证集将从训练数据中划分出来（验证集占训练集的1/5）；

4. 根据 `data_KFold_1/origin/` 目录下的未处理的五折交叉验证数据生成不同损失函数所需的训练集和验证集，这些数据集分别存储在 `data_KFold_1/sample_ContrastiveLoss/`、`data_KFold_1/sample_MultipleNegativesRankingLoss/` 和 `data_KFold_1/sample_TripletLoss/` 目录下；

## 训练（Train）

### model1 (encoder)

> `bash runs/train-wmt-en2de-deep-prenorm-baseline.sh`

### model2 (layer encoder)

> `bash runs/train-wmt-en2de-deep-postnorm-dlcl.sh`

### model3 (encoder)

> `bash runs/train-wmt-en2de-deep-prenorm-dlcl.sh`

## Results

Model | #Param. |Epoch* | BLEU 
:--|:--:|:--:|:--:|
[Transformer](https://arxiv.org/abs/1706.03762) (base) | 65M | 20 | 27.3
[Transparent Attention](https://arxiv.org/abs/1808.07561) (base, `16L`) | 137M | - | 28.0
[Transformer](https://arxiv.org/abs/1706.03762) (big) | 213M | 60 | 28.4
[RNMT+](https://arxiv.org/abs/1804.09849) (big) | 379M | 25 | 28.5
[Layer-wise Coordination](https://papers.nips.cc/paper/8019-layer-wise-coordination-between-encoder-and-decoder-for-neural-machine-translation.pdf) (big) | 210M* | - | 29.0
[Relative Position Representations](https://arxiv.org/abs/1803.02155) (big) | 210M | 60 | 29.2
[Deep Representation](https://arxiv.org/abs/1810.10181) (big) | 356M | - | 29.2
[Scailing NMT](https://arxiv.org/abs/1806.00187) (big) | 210M | 70 | 29.3
Our deep pre-norm Transformer (base, `20L`) | 106M | 20 | 28.9
Our deep post-norm DLCL (base, `25L`) | 121M | 20 | 29.2
Our deep pre-norm DLCL (base, `30L`) | 137M | 20 | 29.3


NOTE: `*` denotes approximate values.



## 结果（Results）

| Model    | STS benchmark | SentEval  |
| ----------------------------------|:-----: |:---:   |
| Avg. GloVe embeddings             | 58.02  | 81.52  |
| BERT-as-a-service avg. embeddings | 46.35  | 84.04  |
| BERT-as-a-service CLS-vector      | 16.50  | 84.66  |
| InferSent - GloVe                 | 68.03  | 85.59  |
| Universal Sentence Encoder        | 74.92  | 85.10  |
|**Sentence Transformer Models**    ||
| nli-bert-base       | 77.12  | 86.37 |
| nli-bert-large     | 79.19  | 87.78 |
| stsb-bert-base    | 85.14  | 86.07 |
| stsb-bert-large   | 85.29 | 86.66|
| stsb-roberta-base | 85.44 | - |
| stsb-roberta-large | 86.39 | - |
| stsb-distilbert-base| 85.16 | - |


## 引用和作者（Citing & Authors）
如果您使用了我们的实验数据或者代码，请引用（If you use the code, feel free to cite our publication [IPC2CLC](https://github.com/i-wanna-to/IPC2CLC/)）:
``` 
@article{XXX-2021-XXX,
    title = "IPC2CLC",
    author = "XX敏, XX西, XX文, XX青", 
    journal= "arXiv preprint arXiv: XXXX.0001",
    month = "8",
    year = "2021",
    url = "https://github.com/i-wanna-to/IPC2CLC/",
}
```

如果您使用了我们发布的实验数据，请先与我们先取得联系，获得同意后再使用！
邮箱: XX敏, xianminhe@jxnu.edu.cn

如果有疑问可以给我发送电子邮件或者发布一个 issue。（Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.）

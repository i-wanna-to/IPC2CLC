import os
import math
import logging
import torch

from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, InputExample, LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from model_args import MappingArgs


class MappingModel:
    def __init__(self, args=None, model_args_file=None, **kwargs):
        if model_args_file:
            self.args = self._load_model_args(model_args_file)

        if args:
            self.args = self.update_args(args)

        #设置随机种子，保证实验的可重复性
        if self.args.seed:
            #random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)  #为CPU设置随机种子
            torch.cuda.manual_seed(self.args.seed)  #为当前GPU设置随机种子
            #如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子
            if self.args.n_gpu > 1:
                torch.cuda.manual_seed_all(self.args.seed)

        #用以保证模型的可重复性，实验的可重复性
        torch.backends.cudnn.deterministic = True

        #设置日志格式
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])

    def creat_model(self):
        #自定义模型，包括transformers中的模型也可以用(也可以自定义模型框架)
        if self.args.custom_model:
            #加载孪生BERT模型
            word_embedding_model = models.Transformer(self.args.model_name)

            #全局平均池化层（可选用三种池化策略）
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                            pooling_mode_mean_tokens=True,
                                            pooling_mode_cls_token=False,
                                            pooling_mode_max_tokens=False)
            #搭建模型
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.args.device)
        else:
            #加载sentence_transformers中已经训练好的模型
            model = SentenceTransformer(self.args.model_name, device=self.args.device)

        return model

    def read_data(self,kfold=0):
        #存储训练数据
        train_num = 0
        train_samples = []
        #存储验证数据
        valid_num = 0
        valid_samples = []

        if self.args.loss_type == "MultipleNegativesRankingLoss":
            with open(self.args.train_dataset_path+self.args.loss_type+'/data_sample_'+str(kfold)+'/ipc_clc_train.txt', 'r', encoding='utf8') as fIn:
                for row in fIn:
                    train_num += 1
                    row = row.strip()
                    row = row.split('\t')
                    train_samples.append(InputExample(texts=[row[0], row[1]])) #MultipleNegativesRankingLoss 无label
            
            #MultipleNegativesRankingLoss直接从文件中读取，其他loss的验证集将直接从训练数据中得到，1/5的训练集数据将作为验证集
            #需要注意的是，如果只有一种label，则evaluator无法进行验证，所以是二分类的要保证验证集中两种类别都有
            with open(self.args.train_dataset_path+self.args.loss_type+'/data_sample_'+str(kfold)+'/ipc_clc_valid.txt', 'r', encoding='utf8') as fIn:
                for row in fIn:
                    valid_num += 1
                    row = row.strip()
                    row = row.split('\t')
                    valid_samples.append(InputExample(texts=[row[0], row[1]], label=int(row[2]))) #验证集需要 label
        else:
            with open(self.args.train_dataset_path+self.args.loss_type+'/data_sample_'+str(kfold)+'/ipc_clc_train.txt', 'r', encoding='utf8') as fIn:
                train_valid_number = 0
                for row in fIn:
                    row = row.strip()
                    row = row.split('\t')
                    train_valid_number = train_valid_number + 1
                    if train_valid_number % 5 == 0:
                        valid_num += 1
                        if self.args.loss_type == "ContrastiveLoss":
                            valid_samples.append(InputExample(texts=[row[0], row[1]], label=int(row[2]))) #验证集需要 label
                        else:
                            valid_samples.append(InputExample(texts=[row[0], row[1]], label=int(1))) #TripletLoss pos
                            valid_samples.append(InputExample(texts=[row[0], row[2]], label=int(0))) #TripletLoss neg
                    else:
                        train_num += 1
                        if self.args.loss_type == "ContrastiveLoss":
                            train_samples.append(InputExample(texts=[row[0], row[1]], label=int(row[2]))) #ContrastiveLoss 有label
                        else:
                            train_samples.append(InputExample(texts=[row[0], row[1], row[2]])) #TripletLoss 无label

        return train_samples, valid_samples

    def train(self, kfold=0):
        model = self.creat_model()
        train_samples, valid_samples = self.read_data(kfold=kfold)
        #构造训练数据 DataLoader
        train_data = SentencesDataset(train_samples, model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args.batch_size)

        #定义训练的损失函数(三种可选)
        if self.args.loss_type == "ContrastiveLoss":
            train_loss = losses.OnlineContrastiveLoss(model=model, margin=0.5)  #有label
        elif self.args.loss_type == "MultipleNegativesRankingLoss":
            train_loss = losses.MultipleNegativesRankingLoss(model=model)  #无label
        elif self.args.loss_type == 'TripletLoss':
            train_loss = losses.TripletLoss(model=model, triplet_margin=1)  #无label，默认triplet_margin=5
        else:
            assert 0, '不支持的损失函数：'+self.args.loss_type

        #构造验证集数据 DataLoader
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, name='ipc2clc-valid')

        #warmup步数，这个也有很大的随机性
        warmup_steps = math.ceil(len(train_samples) * self.args.num_epochs / self.args.batch_size * self.args.warmup_steps) #15% 的训练数据做 warm-up
        
        model_path = self.args.model_save_path + str(kfold) + '_fold'
        #if self.args.optimizer_class == 'Adamw':
        #    optimizer_class = transformers.optimization.AdamW

        #模型训练
        model.fit(train_objectives = [(train_dataloader, train_loss)],
                epochs = self.args.num_epochs,
                #optimizer_class = optimizer_class,
                optimizer_params = {'correct_bias': False, 'eps': self.args.optimizer_eps, 'lr': self.args.lr},
                scheduler = self.args.scheduler,
                evaluator = evaluator,
                evaluation_steps = self.args.evaluation_steps,
                save_best_model = self.args.save_best_model,
                weight_decay = self.args.weight_decay,
                warmup_steps = self.args.warmup_steps,
                max_grad_norm = self.args.max_grad_norm,
                output_path = model_path
                )

        return model_path

    def predict(self, kfold=0, model_path=None):
        """
            思路：
            利用预训练语言模型对IPC和CLC字符串进行编码，得到句子级别的语境向量，
            之后对两者进行语义相似度计算（余弦相似度）。

            步骤：
            1.我们将IPC作为查询字符串，CLC作为待匹配的字符串。

            2.一条IPC字符串需要与每一条CLC字符串进行语义相似度计算（余弦相似度）。

            3.计算完毕后，取出相似度最高的前n条CLC分类号，作为与当前IPC形成映射的CLC分类号。

            4.计算准确率，TopK准确率。
        """

        #读入IPC和CLC测试数据
        read_ipc_file = open(self.args.test_dataset_path+self.args.loss_type+'/data_sample_'+str(kfold)+'/ipc_test.txt', 'r', encoding='utf-8')
        read_clc_file = open(self.args.test_dataset_path+self.args.loss_type+'/data_sample_'+str(kfold)+'/clc_test.txt', 'r', encoding='utf-8')

        #存储IPC和CLC的分别类号(key)和匹配字符串(value)
        ipc_key_list = []
        ipc_value_list = []
        clc_key_list = []
        clc_value_list = []
        for line in read_ipc_file:
            line = line.strip()
            line = line.split(' ')
            assert len(line) == 2, 'IPC.txt文件中只应该有两列，每一列用一个空格隔开！' + str(line[0])
            ipc_key_list.append(str(line[0]))
            ipc_value_list.append(str(line[1]))
        for line in read_clc_file:
            line = line.strip()
            line = line.split(' ')
            assert len(line) == 2, 'CLC.txt文件中只应该有两列，每一列用一个空格隔开！' + str(line[0])
            clc_key_list.append(str(line[0]))
            clc_value_list.append(str(line[1]))
        assert len(ipc_key_list) == len(clc_key_list), 'IPC.txt文件和CLC.txt文件要一一对应，行数要一致！'

        match_clc_key_dict = dict()
        for ipc_key_i, target in enumerate(ipc_key_list):
            match_clc_key_dict[str(ipc_key_i)] = []
            for ipc_key_j, every_ipc_key in enumerate(ipc_key_list):
                if every_ipc_key == target:
                    match_clc_key_dict[str(ipc_key_i)].append(clc_key_list[ipc_key_j])

        #用于计算IPC每一类别包含的数量
        ipc_section_number_dict = defaultdict(int)
        for ipc_key_i in ipc_key_list:
            ipc_section_number_dict[str(ipc_key_i[0])] += 1

        #判断GPU是否可用
        device = self.args.device
        #训练好的模型路径
        model_path = model_path
        #加载预训练语言模型（可更换,传入模型名称/模型路径）
        model = SentenceTransformer(model_path, device=device)
        #将CLC查询字符串转化为语境向量，并转化为tensor
        clc_corpus_embeddings = model.encode(clc_value_list, convert_to_tensor=True, device=device)
        #可规定TopK准确率
        top_k = self.args.top_k
        #存储相似度最高的前n条CLC分类号
        clc_top_k_results = dict()
        #存储匹配正确的数量(只取匹配度最高的)
        correct = 0
        #存储TopK中匹配正确的数量
        top_k_correct = 0
        #存储top2-TopK中匹配正确的数量(包含了topK)
        top2_to_topk_correct_dict = defaultdict(int)
        #存储每一个IPC类别的正确个数
        ipc_section_correct_dict = defaultdict(int)
        #存储相似度最高匹配错误的ipc和clc句对
        error_match_ipc_clc_pair = []
        #存储topk匹配错误的ipc和clc句对
        topk_error_match_ipc_clc_pair = []
        pbar = tqdm(ipc_value_list)
        for ipc_i, ipc_query in enumerate(pbar):
            #将IPC查询字符串转化为语境向量，并转化为tensor
            ipc_query_embedding = model.encode(ipc_query, convert_to_tensor=True, device=device)
            #计算两者的语义相似度计算（余弦相似度）
            cos_scores = util.pytorch_cos_sim(ipc_query_embedding, clc_corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            #使用 torch.topk 找到相似度最高的前n条CLC分类号
            top_results = torch.topk(cos_scores, k=top_k)
            clc_top_k_results[str(ipc_i)] = []
            to_match_clc_key_list = []
            
            #只有前K个结果
            for score, idx in zip(top_results[0], top_results[1]):
                clc_top_k_results[str(ipc_i)].append((str(clc_key_list[idx]), str(clc_value_list[idx]), score))
                to_match_clc_key_list.append(str(clc_key_list[idx]))
            
            #匹配正确的数量(只取第一个匹配度最高的)
            if clc_top_k_results[str(ipc_i)][0][0] in match_clc_key_dict[str(ipc_i)]:
                correct = correct + 1
                ipc_section_correct_dict[str(ipc_key_list[ipc_i][0])] += 1
            else:
                error_match_ipc_clc_pair.append(str(ipc_key_list[ipc_i]) + '  ' + str(ipc_value_list[ipc_i]) + \
                                                '\t|||\t' + str(clc_key_list[ipc_i]) + '  ' + \
                                                str(clc_value_list[ipc_i]) + '\t|||\t' + \
                                                str(clc_top_k_results[str(ipc_i)][0][0]) + '  ' + \
                                                str(clc_top_k_results[str(ipc_i)][0][1]))
            
            #TopK中匹配正确的数量(10个的)
            if len(match_clc_key_dict[str(ipc_i)]) == 1:
                assert match_clc_key_dict[str(ipc_i)][0] == str(clc_key_list[ipc_i]), 'TopK匹配有问题！--1'
                if str(clc_key_list[ipc_i]) in to_match_clc_key_list:
                    top_k_correct = top_k_correct + 1
                else:
                    topk_error_match_ipc_clc_pair.append(str(ipc_key_list[ipc_i]) + '  ' + str(ipc_value_list[ipc_i]) + \
                                                    '\t|||\t' + str(clc_key_list[ipc_i]) + '  ' + \
                                                    str(clc_value_list[ipc_i]) + '\t|||\t' + \
                                                    str(clc_top_k_results[str(ipc_i)][0][0]) + '  ' + \
                                                    str(clc_top_k_results[str(ipc_i)][0][1]))
            elif len(match_clc_key_dict[str(ipc_i)]) > 1:
                for each_clc_key_1 in match_clc_key_dict[str(ipc_i)]:
                    if str(each_clc_key_1) in to_match_clc_key_list:
                        top_k_correct = top_k_correct + 1
                        break
                    else:
                        topk_error_match_ipc_clc_pair.append(str(ipc_key_list[ipc_i]) + '  ' + str(ipc_value_list[ipc_i]) + \
                                                        '\t|||\t' + str(clc_key_list[ipc_i]) + '  ' + \
                                                        str(clc_value_list[ipc_i]) + '\t|||\t' + \
                                                        str(clc_top_k_results[str(ipc_i)][0][0]) + '  ' + \
                                                        str(clc_top_k_results[str(ipc_i)][0][1]))
            else:
                print('TopK匹配有问题！--2')

            #计算tok2-topK每一个的准确率
            for topk_index in range(1, top_k):
                for each_clc_key_1 in match_clc_key_dict[str(ipc_i)]:
                    if str(each_clc_key_1) in to_match_clc_key_list[:topk_index+1]:
                        top2_to_topk_correct_dict[str(topk_index+1)] += 1
                        break

            pbar.set_description("Processing %s" % str(ipc_i+1))
            
        #计算准确率，TopK准确率
        ipc_count_number = len(ipc_key_list)
        acc = correct/ipc_count_number
        top_k_acc = top_k_correct/ipc_count_number
        print('\n-----------第{}折结果-------------\n'.format(kfold))
        print('\n准确率: {:.2f}% .'.format(acc*100))
        for topk_name, top_correct_num in top2_to_topk_correct_dict.items():
            print('\ntop_{} 准确率: {:.2f}% .'.format(topk_name, (top_correct_num/ipc_count_number)*100))
        print('\n---------------------------------\n')

        #结果统计
        print('总数：{}\ntop1 正确个数：{}\ntop1 错误个数：{}\n'.format(ipc_count_number, correct, len(error_match_ipc_clc_pair)))
        for topk_name, top_correct_num in top2_to_topk_correct_dict.items():
            print('总数：{}\ntop{} 正确个数：{}\ntop{} 错误个数：{}\n'.format(ipc_count_number, topk_name, top_correct_num, topk_name, ipc_count_number-top_correct_num))
        print('\n---------------------------------\n')

        #计算每一个IPC类别的正确个数和正确率（只有top1的）
        for sec_i, sec_cor in ipc_section_correct_dict.items():
            print('类别：{}，总数为：{}，正确个数为：{}，错误个数为：{}，正确率为：{:.2f}%\n'.format(sec_i, ipc_section_number_dict[sec_i], sec_cor, ipc_section_number_dict[sec_i]-sec_cor, (sec_cor/ipc_section_number_dict[sec_i])*100))

        log_path = 'error_log/' + self.args.loss_type + '_log/' + str(kfold) + '_fold/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        #相似度最高的错误匹配（topk=1）
        f_write = open(log_path + 'acc-' + str(round(acc*100, 2)) + '%-top1_error_text_log.txt', 'w', encoding='utf-8')
        for k_v in error_match_ipc_clc_pair:
            f_write.write(str(k_v) + '\n')
        f_write.close()

        #topk的错误匹配结果
        f_write = open(log_path + 'acc-' + str(round(acc*100, 2)) + '%-topk_error_text_log.txt', 'w', encoding='utf-8')
        for k_v in topk_error_match_ipc_clc_pair:
            f_write.write(str(k_v) + '\n')
        f_write.close()

        #topk的匹配结果
        f_write = open(log_path + 'acc-' + str(round(acc*100, 2)) + '%-topk_error_score_log.txt', 'w', encoding='utf-8')
        for k_index, v_list_tuple in clc_top_k_results.items():
            for each_tuple in v_list_tuple:
                f_write.write(ipc_key_list[int(k_index)] + '----' + each_tuple[0] + each_tuple[1] + str(each_tuple[2].item()) + '\n')
                f_write.write('------------------------------------------------------------------' + '\n')
        f_write.close()

    def update_args(self):
        parser = argparse.ArgumentParser(description='manual to this script.')
        parser.add_argument('--input_dir', type=str, default='./kko.100w', help='输入文件路径')
        parser.add_argument('--output_dir', type=str, default='./kko.100w.out', help=' 输出文件路径')
        parser.add_argument('--max_len', type=int, default=50, help='句子最长字符数')
        parser.add_argument('--max_clip_num', type=str, default=5, help='在基数上增加的切分个数')
        args = parser.parse_args()
        return args

    def _load_model_args(self, model_args_file):
        args = MappingArgs()
        args.load(model_args_file)
        return args


if __name__ == "__main__":
    fofo = MappingModel(model_args_file='./model_args.json')
    fofo.train()
    fofo.predict()
# TextCNN with Adversarial Training
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/elijahcn/TextCNN-AdversarialTraining/blob/main/LICENSE)

中文文本分类，TextCNN，FGSM，FGM，PGD，FreeAT，基于pytorch。

## 概述
本实验测试对抗训练方法（Adversarial Training）在中文文本分类中的应用。文本分类模型为TextCNN，对抗训练测试了FGSM、 FGM、PGD、FreeAT等几种方法。 

### 软件环境
python 3.7  
pytorch 1.9  
tqdm  
sklearn  
tensorboardX

### 目录结构
    run.py          # 运行主程序
    run_batch       # 训练批处理程序
    train_eval.py   # 训练和评测代码
    utils.py    `   # 数据集处理和词向量生成
    models/         # 各模型代码, 超参定义和模型定义在同一文件中。  
        TextCNN.py  # TextCNN文本分类模型
        FGM.py      # FGM 对抗训练模型
        FGSM.py     # FGSM 对抗训练模型
        PGD.py      # PGD 对抗训练模型
        FreeAT.py   # FreeAT 对抗训练模型
    THUCNEWS/       # THUCNEWS数据集和运行结果
        data/       # 训练测试数据集
            train.txt   # 训练数据集
            dev.txt     # 开发数据集
            test.txt    # 测试数据集
            class.txt   # 类别定义
            embedding_SougouNews.npz    # 从Sougou数据生成的词向量数据
            vocab.pkl   # 词汇表数据
        log/        # 输出log文件
        ckpt/       # 输出模型文件

## 数据说明

### 短文本分类中文数据集
从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集|数据量
训练集|18万
验证集|1万
测试集|1万

### 词向量生成
可以选以字或以词为单位输入模型。
预训练词向量使用搜狗新闻数据，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

本实验的对抗训练都是在Embedding数据上进行扰动。

## 算法说明

### Fast Gradient Method(FGM)
扰动生成：
![Image text](https://github.com/elijahcn/TextCNN-AdversarialTraining/blob/main/imgs/fgm1.svg)

### Projected Gradient Descent(PGD)

### Free Adversarial Training(FreeAT)

### 效果

模型|acc|备注
--|--|--
TextCNN|91.16%|TextCNN baseline
TextCNN+FGM|91.51%|TextCNN + FGM Adversarial Training 
TextCNN+PGD|91.41%|TextCNN + PGD(k=3) Adversarial Training 
TextCNN+FreeAT|91.19%|TextCNN + FreeAT(m=3) Adversarial Training 


## 使用说明
```
# 训练并测试：
# TextCNN + 无对抗训练
python run.py --model TextCNN

# TextCNN + FGM
python run.py --model TextRNN --adv FGM

# TextCNN + FGSM
python run.py --model TextRNN --adv FGSM

# TextCNN + PGD
python run.py --model TextRNN --adv PGD

# TextCNN + FreeAT
python run.py --model TextRNN --adv FreeAT
```


## 参考
[1] Explaining and Harnessing Adversarial Examples.
    arxiv.org/abs/1412.6572
[2] Adversarial Training Methods for Semi-Supervised Text Classification.
    arxiv.org/abs/1605.07725
[3] Fast is better than free: Revisiting adversarial training
    arxiv.org/abs/2001.03994
[4] https://github.com/locuslab/fast_adversarial
[5] https://github.com/649453932/Chinese-Text-Classification-Pytorch

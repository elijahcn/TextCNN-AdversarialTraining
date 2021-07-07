# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse

from train_eval import train, init_network

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN')
parser.add_argument('--adv', type=str, default='', required=False, help='choose a AT method: FGSM, PGD, FreeAT')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNEWS'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN
    adv_name = args.adv     # FGSM, PGD, FREEAT
    from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    y = import_module('models.' + adv_name) if len(adv_name)>0 else None
    config = x.Config(dataset, embedding, adv_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.set_default_tensor_type(torch.DoubleTensor)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.ClsModel(config).to(config.device)
    adv = y.ATModel(model) if y else None
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, adv)

# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F

EMB_NAME = 'embedding.'     # Embedding参数名称前缀

class ATModel():
    ''' FreeAT - Free Adversarial Training
    '''

    def __init__(self, model):
        self.model = model          # 文本分类模型
        self.emb_backup = {}        # Embedding数据备份
        self.epsilon = 1.0
        self.M = 3                  # FreeAT 循环次数
        self.backup_emb()           # 备份原始的embedding数据

    def train(self, train_data, labels, optimizer):
        '''
        对抗训练过程
        '''
        # 对每个样本循环执行M次
        for t in range(self.M):
            # 正常训练
            outputs = self.model(train_data)
            self.model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # 在embedding上添加对抗扰动，一直累积下去
            self.attack_emb(backup=False)
        return outputs, loss

    def attack_param(self, name, param):
        # 对某个参数的值添加扰动
        #  FreeAT: r[t+1] = r[t] + epsilon * sign(grad)
        '''
        r_at = self.epsilon * np.sign(param.grad)      # 根据梯度计算这一轮的r
        param.short.add_(r_at)  # 添加扰动
        param.short = self.project(name, param.short)
        '''
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = self.epsilon * param.grad / norm
            param.data.add_(r_at)           # 添加扰动
            param.data = self.project(name, param.data)


    def project(self, param_name, param_data):
        ''' 对参数值进行校正投射'''
        r = param_data - self.emb_backup[param_name]    # 计算和原始Embedding之间的差
        # r.clamp_(-self.epsilon, self.epsilon)  # 约束在 (-eps, eps)范围内
        if torch.norm(r) > self.epsilon:  # 如超出扰动半径eps，则投射回球面
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def attack_emb(self, backup=False):
        '''
        对所有Embedding参数值添加扰动
        :param backup: 是否对Embedding参数进行备份
        :return:
        '''
        # 遍历各参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                if backup:  # 备份参数值
                    self.emb_backup[name] = param.data.clone()
                # 调用单个参数添加扰动
                self.attack_param(name, param)

    def backup_emb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                self.emb_backup[name] = param.data.clone()

    def restore_emb(self):
        '''恢复之前备份的Embedding参数'''
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}


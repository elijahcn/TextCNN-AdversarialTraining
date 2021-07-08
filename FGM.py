# coding: UTF-8
import torch
import torch.nn.functional as F

EMB_NAME = 'embedding.'     # Embedding参数名称前缀

class ATModel():
    ''' FGM - Fast Gradient Method
    '''

    def __init__(self, model):
        self.model = model          # TextCNN文本分类模型
        self.emb_backup = {}        # Embedding数据备份
        self.epsilon = 1.0

    def train(self, train_data, labels, optimizer):
        '''
        针对每个样本的对抗训练过程。
        :param train_data:训练样本数据为Tensor
        :param labels: 标签为Tensor
        :param optimizer: 模型的优化器
        :return: outputs: 分类结果
                  loss:   损失值
        '''
        # 先计算正常样本的梯度以计算扰动
        outputs = self.model(train_data)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # 对抗训练
        self.attack_emb()  # 添加对抗扰动
        outputs = self.model(train_data)
        loss = F.cross_entropy(outputs, labels)
        self.model.zero_grad()      # 仅计算对抗样本的梯度
        loss.backward()  # 扰动后的反向传播，用对抗样本梯度进行网络参数更新
        self.restore_emb()  # 恢复参数
        optimizer.step()
        return outputs, loss

    def attack_param(self, name, param):
        '''
        对某个参数的值添加扰动
        :param name: 参数名称
        :param param: 参数
        :return:
        '''
        #  FGM: r = epsilon * grad / norm(grad)
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = self.epsilon * param.grad / norm
            param.data.add_(r_at)

    def attack_emb(self, backup=True):
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

    def restore_emb(self):
        '''恢复之前备份的Embedding参数'''
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}


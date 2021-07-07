# coding: UTF-8
import torch
import torch.nn.functional as F

EMB_NAME = 'embedding.'     # Embedding参数名称前缀

class ATModel():
    ''' PGD - Projected Gradient Descent
    '''

    def __init__(self, model):
        self.model = model          # TextCNN文本分类模型
        self.emb_backup = {}        # Embedding数据备份
        self.grad_backup = {}       # 梯度数据备份
        self.epsilon = 1.0
        self.alpha = 0.3
        self.K = 3          # PGD循环次数

    def train(self, train_data, labels, optimizer):
        '''
        对抗训练过程
        :param train_data:训练数据为Tensor
        :param labels: 标签为Tensor
        :return: outputs: 分类结果
                  loss:   损失值
        '''
        # 计算正常样本下的梯度
        outputs = self.model(train_data)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # 再进行对抗训练
        self.backup_grad()  # 梯度备份
        for t in range(self.K):
            self.attack_emb(backup=(t == 0))  # 在embedding上添加对抗扰动
            if t != self.K - 1:
                self.model.zero_grad()
            else:
                self.restore_grad()  # 最后一次恢复正常的grad，并累加对抗训练的梯度
            outputs = self.model(train_data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()  # 反向传播，计算梯度
        self.restore_emb()  # 恢复embedding参数
        optimizer.step()
        return outputs, loss

    def attack_param(self, name, param):
        '''
        对某个参数的值添加扰动
        :param name: 参数名称
        :param param: 参数
        :return:
        '''
        #  PGD: r = epsilon * grad / norm(grad)
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
            r_at = self.alpha * param.grad / norm
            param.data.add_(r_at)
            param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        '''
        对参数值进行校正，如超出扰动半径eps，则投射回球面
        :param param_name:
        :param param_data:
        :return:
        '''
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

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

    def backup_grad(self):
        '''备份所有梯度数据'''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        '''恢复之前备份的梯度数据'''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


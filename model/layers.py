import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    """
    通过GCN得到节点嵌入以后，应该输入该模块得到图嵌入。\\
    输入的节点嵌入维度为 N * D，N是节点个数，D是嵌入维度，也就是filters_3.也就是16
    """
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        self.config = config
        # 创建权重W2, 维度是D * D, 用于计算全局上下文c
        self.W2 = nn.Parameter(torch.Tensor(self.config['filters_3'], self.config['filters_3']))
        # 使用 Xavier 方法初始化权重矩阵W2, 据说可以缓解梯度下降问题
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x): 
        """
        返回图级表示矩阵 根据后面的NTN模块，这里输出的h应该是D * 1格式 \\
        x 维度：N * D 整张图的节点级表示\\
        w2 维度：D * D \\
        c == transformed_global 的维度应该是 N * D \times D * D = N * D.mean() = 1 * D\\
        SimGNN公式(2)
        """
        c = torch.tanh(torch.mm(x, self.W2).mean(dim=0)).view(1, -1) # 1 * D
        h = torch.mm(x.T, torch.sigmoid(torch.mm(x, c.T))).T # 1 * D = (D * N \times (N * D \times D * 1)).T
        return h # 1 * D

class TensorNetworkModule(torch.nn.Module):
    def __init__(self, config):
        super(TensorNetworkModule, self).__init__()
        self.config = config
        self.W3 = nn.Parameter(torch.Tensor(self.config['tensor_neurons'], self.config['filters_3'], self.config['filters_3'])) # K * D * D 特定跟论文中不同，为了更好实现代码
        self.V  = nn.Parameter(torch.Tensor(self.config['tensor_neurons'], 2 * self.config['filters_3']))                       # K * 2D
        self.b3 = nn.Parameter(torch.Tensor(1, self.config['tensor_neurons']))                                                  # 1 * K
        nn.init.xavier_uniform_(self.W3)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.b3)

    def forward(self, hi, hj):
        """
        hi: 1 * D
        hj: 1 * D
        W3: D * D * K
        """
        term_1 = []
        for W_0 in self.W3:
            term_1.append(torch.mm(torch.mm(hi, W_0), hj.T))
        term_1 = torch.cat(term_1, dim=1) # 1 * K
        term_2 = torch.mm(self.V, torch.cat((hi, hj),dim = 1).T).T # 1 * K

        scores = F.relu(term_1 + term_2 + self.b3) # SimGNN公式(3)
        return scores # 1 * K

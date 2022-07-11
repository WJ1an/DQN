import torch
import numpy as np


class QNetwork(torch.nn.Module):
    """
    Q网络，由两个全连接层组成，激活函数为Relu。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)  # 输入层
        self.out = torch.nn.Linear(hidden_size, output_size)  # 隐藏层

    def forward(self, inputs):
        x = torch.nn.functional.relu(self.lin1(inputs))
        x = self.out(x)
        return x


class ReplayBuffer:
    """
    经验回放池
    """
    def __init__(self, args):
        self.size = args.buffer_size
        self.position = 0
        self.buffer = {}

        self.current_size = 0

        self.buffer["state"] = np.empty([self.size, args.state_size])
        self.buffer["act"] = np.empty([self.size, args.act_size])
        self.buffer["reward"] = np.empty([self.size, 1])
        self.buffer["state_next"] = np.empty([self.size, args.state_size])
        self.buffer["done"] = np.empty([self.size, 1])

    def __len__(self):
        return self.current_size

    def push(self, s, a, r, s_n, d):   # 存储MDP五元组
        self.buffer["state"][self.position] = s
        self.buffer["act"][self.position] = a
        self.buffer["reward"][self.position] = r
        self.buffer["state_next"][self.position] = s_n
        self.buffer["done"][self.position] = d
        self.position = (self.position + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):   # 随机采样batch_size大小的数据
        batch = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            batch[key] = self.buffer[key][idx]
        return batch

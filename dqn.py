import torch
import os
from utils import QNetwork


class DQN:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

        self.Q_net = QNetwork(args.state_size, args.hidden_size, args.act_size).to(self.device)  # 构建Q网络
        self.Q_target = QNetwork(args.state_size, args.hidden_size, args.act_size).to(self.device)  # 构建目标网络
        self.Q_target.load_state_dict(self.Q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=args.lr)  # 也可以使用RMSprop

        self.train_step = 0

        self.model_path = "models/DQN"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def train(self, batch):
        """
        训练DQN
        :param batch: 用于训练的数据，通过经验回放池随机采样获取
        :return: loss: TD误差
        """
        if self.train_step > 0 and self.train_step % self.args.target_update_freq == 0:  # 在一定训练步后，更新目标网络参数
            self.Q_target.load_state_dict(self.Q_net.state_dict())

        state = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        act = torch.tensor(batch["act"], dtype=torch.int64, device=self.device)
        reward = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        state_next = torch.tensor(batch["state_next"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_next, a_next = self.Q_target(state_next).max(1)
            q_next = q_next.unsqueeze(-1)
            q_target = reward + (1 - done) * self.args.gamma * q_next  # 目标Q值：r + \gamma * max{Q(s', a')}

        q = self.Q_net(state)
        q = q.gather(1, act)
        loss = torch.nn.functional.mse_loss(q, q_target)  # TD误差

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1

        return loss

    def select_action(self, state):
        """
        选择动作，主要用于与环境交互
        :param state: 当前环境状态
        :return: act: 动作值
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.Q_net(state)
        act = torch.argmax(q).cpu()  # a = argmax_a q(s,a)
        return act.numpy()

    def save_model(self):
        """
        保存模型
        """
        torch.save(self.Q_net.state_dict(), self.model_path + "/q_net_parameters.pkl")

    def load_model(self):
        """
        加载模型
        """
        self.Q_net.load_state_dict(torch.load(self.model_path + "/q_net_parameters.pkl"))


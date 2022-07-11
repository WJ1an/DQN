import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from dqn import DQN
from utils import ReplayBuffer
from arguments import get_arguments


class Runner:
    def __init__(self, args):
        self.args = args
        self.agent = DQN(args)
        self.buffer = ReplayBuffer(args)
        self.log_path = "log/DQN"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        self.env = gym.make(args.env_name)
        self.evaluate_env = gym.make(args.env_name)

        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.evaluate_freq = args.evaluate_freq
        self.evaluate_episode = args.evaluate_episode
        self.target_update_freq = args.target_update_freq

    def run(self):
        log_reward = []
        loss = []
        best_reward = -99
        step = 0
        # 与环境交互的轨迹数
        for episode in range(self.args.episodes_num):
            done = False
            state = self.env.reset()  # 初始化游戏环境
            # 对于每条轨迹交互至结束
            while not done:
                # 动作选取采用epsilon-贪婪算法
                if np.random.uniform() < self.epsilon:
                    act = self.env.action_space.sample()  # 随机选取动作
                else:
                    act = self.agent.select_action(state)

                # 与环境交互
                state_next, reward, done, _ = self.env.step(act)

                # 存储MDP五元组
                self.buffer.push(state, act, reward, state_next, done)

                # 有足够数据后就开始训练
                if step > 0 and len(self.buffer) > self.args.batch_size:
                    batch = self.buffer.sample(self.args.batch_size)
                    loss.append(self.agent.train(batch))

                # 在给定交互步数后开始评估当前智能体
                if step > 0 and step % self.evaluate_freq == 0:
                    evaluate_reward = self.evaluate()
                    log_reward.append(evaluate_reward)  # 保存评估奖励用于画图

                    # 如果当前奖励是最好的，则保存模型
                    if evaluate_reward >= best_reward:
                        self.agent.save_model()
                        best_reward = evaluate_reward

                    print("step: {}, episode: {}, evaluate reward: {}, best reward: {}, epsilon: {}".format(step, episode, evaluate_reward, best_reward, self.epsilon))

                state = state_next
                step += 1
                self.epsilon = max(self.epsilon_min, self.epsilon - 0.00005)  # 更新epsilon

        plt.figure()
        plt.plot(range(len(log_reward)), log_reward)
        plt.xlabel("step * " + str(self.evaluate_freq))
        plt.ylabel("average rerurns")
        plt.savefig(self.log_path + '/DQN_train.png')
        np.save(self.log_path + '/reward.pkl', log_reward)

    def evaluate(self, load=False):
        if load:
            self.agent.load_model()
        returns = []
        for episode in range(self.evaluate_episode):
            state = self.evaluate_env.reset()
            done = False
            episode_reward = 0
            while not done:
                act = self.agent.select_action(state)
                state_next, reward, done, _ = self.evaluate_env.step(act)
                episode_reward += reward
                state = state_next
            returns.append(episode_reward)
        return sum(returns) / self.evaluate_episode


if __name__ == '__main__':
    args = get_arguments()
    env = gym.make(args.env_name)
    args.state_size = env.observation_space.shape[0]  # 获取环境的状态维度
    args.act_size = env.action_space.n  # 获取环境动作维度
    runner = Runner(args)

    if args.evaluate:  # 对保存的模型进行评估
        returns = runner.evaluate(load=True)
        print('Average returns is', returns)
    else:
        runner.run()  # 训练模型

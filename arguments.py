import argparse


def get_arguments():
    """
    需要用到的超参数
    """
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for DQN")
    parser.add_argument('--env_name', default="CartPole-v1", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)  # 512
    parser.add_argument("--buffer_size", default=int(1e5), type=int)
    parser.add_argument("--lr", default=0.008, type=float)  # 0.00025
    parser.add_argument("--batch_size", default=32, type=int)  # 32 in DQN 2015,,,1024
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--epsilon", default=0.95, type=float)
    parser.add_argument("--epsilon_min", default=0.005, type=float)
    parser.add_argument("--episodes_num", default=500, type=int)
    parser.add_argument("--evaluate", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--evaluate_freq", default=100, type=int)
    parser.add_argument("--evaluate_episode", default=20, type=int)
    parser.add_argument("--target_update_freq", default=200, type=int)  # 40000--env_name

    args = parser.parse_args()

    return args

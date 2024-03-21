import os
import argparse
from datetime import datetime
import torch

from envs_transfer.initialize_env import make_env
from airl.algo import transfer_SAC, transfer_PPO
from airl.transfer_trainer import Transfer_Trainer
torch.autograd.set_detect_anomaly(True)


def run(args):
    # Create real envs to use for training and evaluation
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    if args.algo == 'transfer_sac':
        algo = transfer_SAC(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            seed=args.seed)
        disc_path = os.path.join(
            'logs', args.airl_env_id, 'airl_sac', f'seed{args.load_seed}-{args.load_time}',
            'model', f'step{args.airl_num_steps}')
    else:
        algo = transfer_PPO(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            seed=args.seed)
        disc_path = os.path.join(
            'logs', args.airl_env_id, 'airl', f'seed{args.load_seed}-{args.load_time}',
            'model', f'step{args.airl_num_steps}')

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    transfer_trainer = Transfer_Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        disc_path=disc_path,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    transfer_trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--airl_env_id', type=str, default='PointMaze-Right')
    p.add_argument('--airl_num_steps', type=int, default=1500000)
    p.add_argument('--load_seed', type=int, default=0)
    p.add_argument('--load_time', type=str, default='20231226-1522')
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='PointMaze-Left')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--algo', type=str, default='transfer_sac')
    p.add_argument('--cuda_id', type=int, default=0)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    run(args)

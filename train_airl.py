import os
import argparse
from datetime import datetime
import torch
import gym

from torch.utils.tensorboard import SummaryWriter
from envs_transfer.initialize_env import make_env
from airl.buffer import SerializedBuffer
from airl.algo import ALGOS
from airl.trainer import Trainer


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        epoch_disc=args.epoch_disc, epoch_policy=args.epoch_policy, batch_size=args.batch_size,
        lr_critic=args.lr_critic, lr_disc=args.lr_disc
        # , lr_weight_decay=args.lr_weight_decay
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Pointmaze-Right')
    p.add_argument('--algo', type=str, default='airl')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epoch_disc', type=int, default=1)
    p.add_argument('--epoch_policy', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr_critic',type=float, default=3e-4)
    p.add_argument('--lr_disc', type=float, default=3e-4)
    p.add_argument('--lr_weight_decay', type=float,default=3e-1)
    p.add_argument('--cuda_id', type=int, default=0)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    run(args)

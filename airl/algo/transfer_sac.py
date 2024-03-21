import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .transfer_base import Algorithm
from airl.buffer import Buffer
from airl.utils import soft_update, disable_gradient
from airl.network import (
    StateDependentPolicy, StateIndependentPolicy, TwinnedStateActionFunction, AIRLDiscrim
)


class transfer_SAC(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.99,
                 batch_size=256, buffer_size=10**6, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, units_actor=(256, 256),
                 units_critic=(256, 256), units_disc_r=(100, 100), units_disc_v=(100, 100), 
                 start_steps=10000, tau=5e-3, alpha_init=0.2):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Replay buffer.
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor.
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        # Critic.
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()
        
        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
        
        # Source Actor
        self.source_actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        # self.source_actor = StateIndependentPolicy(
        #     state_shape=state_shape,
        #     action_shape=action_shape,
        #     hidden_units=(64, 64),
        #     hidden_activation=nn.Tanh()
        # ).to(device)

        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        # Entropy coefficient.
        self.alpha = alpha_init
        # We optimize log(alpha) because alpha should be always bigger than 0.
        self.log_alpha = (alpha_init * torch.ones(1, device=device)).log()
        self.log_alpha.requires_grad = True
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
    
    def load_disc(self, disc_path):
        super().load_disc(disc_path)
        self.disc.load_state_dict(torch.load(disc_path))
    
    def load_source_actor(self, disc_path):
        super().load_source_actor(disc_path)
        self.source_actor.load_state_dict(torch.load(disc_path))

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, step):
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)[0]

        next_state, _, done, _ = env.step(action)
        
        reward = 0
        
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)
        
        source_log_pis = self.source_actor.sample(states)[1]
        rewards = self.disc.calculate_reward(states, dones, source_log_pis, next_states)
        
        self.update_critic(
            states, actions, rewards, dones, next_states, writer)
        self.update_actor(states, writer)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states,
                      writer):
        curr_qs1, curr_qs2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic1', loss_critic1.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic2', loss_critic2.item(), self.learning_steps)

    def update_actor(self, states, writer):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        for name, param in self.actor.named_parameters():
            param.retain_grad()
        loss_actor.backward(retain_graph=True)
        self.optim_actor.step()

        entropy = -log_pis.detach_().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        with torch.no_grad():
            self.alpha = self.log_alpha.exp().item()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'loss/alpha', loss_alpha.item(), self.learning_steps)
            writer.add_scalar(
                'stats/alpha', self.alpha, self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)
            for name, param in self.actor.named_parameters():
                param_grad_cpu = param.grad
                if param_grad_cpu is not None:
                    param_grad_cpu = param_grad_cpu.cpu().numpy()
                    max_grad = np.max(np.abs(np.reshape(param_grad_cpu, [-1])))
                    writer.add_scalar(
             	       'loss/actor_grad/%s' % name, max_grad, self.learning_steps)

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )



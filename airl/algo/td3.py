import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import os

from .base import Algorithm
from airl.network import (DeterministicCritic, DeterministicActor)
from airl.utils import soft_update, disable_gradient
from airl.buffer import DeterministicBuffer


class TD3(Algorithm):
    def __init__(
            self, state_shape, action_shape, device, seed, gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            expl_noise=0.1,
            noise_clip=0.5,
            lr_actor=1e-3,
            lr_critic=1e-3,
            beta=0.05,
            num_noise_samples=50,
            with_importance_sampling=0,
            batch_size=100,
            buffer_size=10 ** 6,
            start_steps=10000,
            units_actor=(400, 300),
            units_critic=(400, 300),
        ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.device = device

        # Actor.
        self.actor = DeterministicActor(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor_target = DeterministicActor(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()
        self.actor_optimizer1 = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic1 = DeterministicCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target1 = DeterministicCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)

        self.critic2 = DeterministicCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic_target2 = DeterministicCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device).eval()
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        soft_update(self.actor_target, self.actor, 1.0)
        disable_gradient(self.actor_target)

        soft_update(self.critic_target1, self.critic1, 1.0)
        disable_gradient(self.critic_target1)
        soft_update(self.critic_target2, self.critic2, 1.0)
        disable_gradient(self.critic_target2)

        self.buffer = DeterministicBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.beta = beta
        self.num_noise_samples = num_noise_samples
        self.with_importance_sampling = with_importance_sampling

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.expl_noise = expl_noise
        self.buffer_size = buffer_size

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def exploretd3(self, state):
        action = (self.select_action(state) + np.random.normal(0, self.expl_noise, size=self.action_shape)).clip(-1, 1)
        return action

    def exploit(self, state):
        action = self.select_action(state)
        return action

    def step(self, env, state, t, step):
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.exploretd3(state)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        self.buffer.add(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)
            
        self.update_td3(
            states, actions, rewards, dones, next_states, writer)

    def update_td3(self, states, actions, rewards, dones, next_states,
                   writer):
        self.learning_steps += 1
        self.update_critic(
            states, actions, rewards, dones, next_states, writer)
        if self.learning_steps % 2 == 0:
            self.update_actor(states, writer)
            self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states,
                      writer):
                      
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_states)+noise).clamp(-1, 1)
        next_Q1 = self.critic_target1(next_states, next_actions)
        next_Q2 = self.critic_target2(next_states, next_actions)
        
        next_Q = torch.min(next_Q1, next_Q2)

        target_Q = rewards + (1.0 - dones) * self.gamma * next_Q

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer1.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic2_loss.backward()
        self.critic_optimizer2.step()
        
        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic1', critic1_loss.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic2', critic2_loss.item(), self.learning_steps)

    def update_actor(self, states, writer):
        actor_loss = -self.critic1(states, self.actor(states)).mean()

        self.actor_optimizer1.zero_grad()
        for name, param in self.actor.named_parameters():
            param.retain_grad()
            
        actor_loss.backward(retain_graph=True)
        
        self.actor_optimizer1.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', actor_loss.item(), self.learning_steps_disc*self.rollout_length)
            for name, param in self.actor.named_parameters():
                param_grad_cpu = param.grad
                if param_grad_cpu is not None:
                    param_grad_cpu = param_grad_cpu.cpu().numpy()
                    max_grad = np.max(np.abs(np.reshape(param_grad_cpu, [-1])))
                    writer.add_scalar(
             	       'loss/actor_grad/%s' % name, max_grad, self.learning_steps_disc*self.rollout_length)

    def update_target(self):
        soft_update(self.critic_target1, self.critic1, self.tau)
        soft_update(self.critic_target2, self.critic2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def calc_pdf(self, samples, mu=0):
        pdfs = 1 / (self.policy_noise * np.sqrt(2 * np.pi)) * torch.exp(
            - (samples - mu) ** 2 / (2 * self.policy_noise ** 2))
        pdf = torch.prod(pdfs, dim=2)
        return pdf

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


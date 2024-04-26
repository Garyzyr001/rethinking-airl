import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .td3 import TD3
from airl.network import AIRLDiscrim


class AIRLTD3(TD3):
    def __init__(self, buffer_exp, state_shape, action_shape, device, seed, gamma=0.99,
                 tau=0.005, policy_noise=0.2, expl_noise=0.1, noise_clip=0.5, lr_actor=1e-3,
                 lr_critic=1e-3, beta=0.005, num_noise_samples=50, with_importance_sampling=0,
                 batch_size=100, start_steps=10000, units_actor=(400, 300), units_critic=(400, 300),
                 rollout_length=2048, buffer_size=10**6, lr_disc=3e-4, 
                 units_disc_r=(100, 100), units_disc_v=(100, 100), 
                 epoch_disc=5, epoch_policy=50):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, tau, policy_noise,
            expl_noise, noise_clip, lr_actor, lr_critic, beta, 
            num_noise_samples, with_importance_sampling, batch_size,
            buffer_size, start_steps,
            units_actor, units_critic
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
        
        self.start_steps = start_steps

        self.learning_steps_disc = 0
        self.rollout_length = rollout_length
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.epoch_policy = epoch_policy
    
    def explore(self, state):
        state = state.reshape(self.batch_size, -1)
        state = state.unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return log_pi

    def exploit(self, state):
        action = self.select_action(state)
        return action
    
    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            # Samples from current policy's trajectories.
            states, actions, _, dones, next_states = \
                self.buffer.sample(self.batch_size)
            with torch.no_grad():
                log_pis = self.actor.evaluate_log_pi(states, actions)
            
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)
                
            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )
        
        for _ in range(self.epoch_policy):
            # We don't use reward signals here,
            states, actions, _, dones, next_states = self.buffer.sample(self.batch_size)
            # Calculate rewards.
            with torch.no_grad():
                log_pis = self.actor.evaluate_log_pi(states, actions)
            rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)
            self.update_td3(states, actions, rewards, dones, next_states, writer)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps_disc*self.rollout_length)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps_disc*self.rollout_length)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps_disc*self.rollout_length)
    
    def is_update(self, step):
        return step % self.rollout_length == 0
    
    def save_models(self, save_dir):
        super().save_models(save_dir)
        
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )

        torch.save(
            self.disc.state_dict(),
            os.path.join(save_dir, 'disc.pth')
        )

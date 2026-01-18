import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(
        self,
        actor,
        critic,
        lr_actor=5e-5,          # Reduced from 5e-4 to match RLlib
        lr_critic=5e-5,         # Reduced from 1e-3 to match RLlib
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.3,         # Changed from 0.2 to match RLlib default
        value_loss_coef=1.0,    # Changed from 0.5 to match RLlib
        entropy_coef=0.05,      # Increased back to 0.05 for exploration
        max_grad_norm=0.5,
        ppo_epochs=10,          # NEW: Multiple epochs per update
        num_minibatches=8,      # NEW: Number of minibatches
        device="cpu"
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.device = device

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_action(self, obs, deterministic=False):
        """
        Get action from observation.
        """
        obs = obs.to(self.device)
        mean, std = self.actor(obs)
        if deterministic:
            return mean, None
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_value(self, state):
        """
        Get state value.
        For IPPO, state = obs
        For MAPPO, state = global_state
        """
        state = state.to(self.device)
        value = self.critic(state)
        return value

    def update(self, rollouts):
        """
        Update networks using collected data with MULTIPLE EPOCHS and MINIBATCHES.
        This is the key fix to match RLlib's PPO implementation.
        """
        obs_batch = rollouts['obs'].to(self.device)
        state_batch = rollouts['state'].to(self.device)
        actions_batch = rollouts['actions'].to(self.device)
        old_log_probs_batch = rollouts['log_probs'].to(self.device)
        returns_batch = rollouts['returns'].to(self.device)
        advantages_batch = rollouts['advantages'].to(self.device)

        # Normalize advantages (do this ONCE before epochs, not inside)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        batch_size = obs_batch.shape[0]
        minibatch_size = batch_size // self.num_minibatches
        
        # Track losses for logging
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            # Shuffle indices for this epoch
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = obs_batch[mb_indices]
                mb_state = state_batch[mb_indices]
                mb_actions = actions_batch[mb_indices]
                mb_old_log_probs = old_log_probs_batch[mb_indices]
                mb_returns = returns_batch[mb_indices]
                mb_advantages = advantages_batch[mb_indices]

                # --- Update Actor ---
                mean, std = self.actor(mb_obs)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # --- Update Critic ---
                values = self.critic(mb_state).squeeze(-1)
                value_loss = F.mse_loss(values, mb_returns) * self.value_loss_coef

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Return average losses across all minibatch updates
        return total_actor_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.optimizer_actor.state_dict(),
            'opt_critic': self.optimizer_critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer_actor.load_state_dict(state_dict['opt_actor'])
        self.optimizer_critic.load_state_dict(state_dict['opt_critic'])

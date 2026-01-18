import argparse
import time
import os

import torch
import numpy as np
import wandb
from vmas import make_env
from vmas.simulator.utils import save_video
from marl.model import Actor, Critic
from marl.ppo import PPO

def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """
    计算 GAE (Generalized Advantage Estimation)
    """
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    # values 长度为 T, next_value 是标量或 (N, 1)
    
    nb_steps = rewards.shape[0]
    
    for t in reversed(range(nb_steps)):
        if t == nb_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    returns = advantages + values
    return advantages, returns

def train(args):
    # WandB 初始化
    wandb.init(
        project="vmas_marl_repro",
        config=vars(args),
        name=f"{args.algo}_{args.scenario}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建环境
    env = make_env(
        scenario=args.scenario,
        num_envs=args.num_envs,
        device=device,
        continuous_actions=True,
        seed=args.seed,
        clamp_actions=True,
        max_steps=args.max_episode_steps,  # 超时强制结束，防止僵尸环境
        terminated_truncated=True  # 区分真正完成和超时
    )

    n_agents = env.n_agents
    obs_dim = env.observation_space[0].shape[0] # 每个智能体的观测维度
    act_dim = env.action_space[0].shape[0] # 每个智能体的动作维度

    # 全局状态维度 (MAPPO/CPPO)
    # CPPO: Actor 看全局状态, Critic 看全局状态
    # MAPPO: Actor 看局部观测, Critic 看全局状态
    # IPPO: Actor 看局部观测, Critic 看局部观测
    state_dim = obs_dim * n_agents
    
    print(f"Scenario: {args.scenario}")
    print(f"Algorithm: {args.algo}")
    print(f"Agents: {n_agents}, Obs Dim: {obs_dim}, Act Dim: {act_dim}, Global State Dim: {state_dim}")

    # 初始化模型
    if args.algo == "cppo":
        # CPPO: 集中式 Actor (超级智能体)
        # 输入: 全局状态 (N_agents * Obs_dim)
        # 输出: 联合动作 (N_agents * Act_dim)
        print("Initializing CPPO: Centralized Actor & Critic")
        actor_input_dim = state_dim
        actor_output_dim = act_dim * n_agents
        critic_input_dim = state_dim
    elif args.algo == "mappo":
        # MAPPO: 分布式 Actor (共享参数), 集中式 Critic
        print("Initializing MAPPO: Decentralized Actor & Centralized Critic")
        actor_input_dim = obs_dim
        actor_output_dim = act_dim
        critic_input_dim = state_dim
    elif args.algo == "ippo":
        # IPPO: 分布式 Actor (共享参数), 分布式 Critic (共享参数)
        print("Initializing IPPO: Decentralized Actor & Decentralized Critic")
        actor_input_dim = obs_dim
        actor_output_dim = act_dim
        critic_input_dim = obs_dim
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    actor = Actor(actor_input_dim, actor_output_dim, hidden_dim=64)
    critic = Critic(critic_input_dim, hidden_dim=64)

    agent = PPO(
        actor, critic, 
        lr_actor=args.lr, 
        lr_critic=args.lr, 
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        device=device
    )

    # 训练循环
    total_steps = 0
    num_updates = args.max_steps // args.steps_per_update
    
    obs = env.reset()
    # obs: List[Tensor(num_envs, obs_dim)] -> 堆叠为 (num_envs, n_agents, obs_dim)
    obs = torch.stack(obs, dim=1)
    
    # Episode 返回跟踪 (Running Return): 用于记录每个环境的累计奖励
    running_return = torch.zeros(args.num_envs, device=device)
    
    # 视频保存计数器（最多保存10个成功视频）
    video_count = 0
    max_videos = 10 

    for update in range(num_updates):
        start_time = time.time()
        
        # 数据缓冲区
        b_obs = []      # IPPO/MAPPO Actor 输入
        b_state = []    # MAPPO/CPPO Critic/Actor 输入
        b_actions = []
        b_log_probs = []
        b_rewards = []
        b_dones = []
        b_values = []
        
        # 课程学习参数
        curriculum_done = (not args.use_curriculum) or (update >= args.curriculum_steps)
        
        # 视频录制：只在课程学习完成后录制，且未达到最大视频数
        frame_list = []
        recording_env_idx = None  # 正在录制的环境索引
        video_saved_this_update = False  # 本次更新是否已保存视频
        
        # Done 计数器 (只统计真正完成，不含超时)
        terminated_count = 0
        truncated_count = 0
        # 完成的 Episode 回报列表
        episode_returns = []

        for step in range(args.steps_per_update):
            # 准备输入数据
            # global_state: (num_envs, state_dim)
            global_state = obs.reshape(args.num_envs, -1)
            
            with torch.no_grad():
                if args.algo == "cppo":
                    # CPPO: Actor 接收全局状态
                    # 输出: (num_envs, n_agents * act_dim)
                    joint_actions, joint_log_probs = agent.get_action(global_state)
                    
                    # 将动作 reshape 为环境需要的格式: (num_envs, n_agents, act_dim)
                    actions_reshaped = joint_actions.reshape(args.num_envs, n_agents, act_dim)
                    log_probs = joint_log_probs # (num_envs,) 联合动作的对数概率
                    
                    # Critic 接收全局状态
                    values = agent.get_value(global_state) # (num_envs, 1)

                elif args.algo == "mappo":
                    # MAPPO: Actor 接收局部观测
                    # obs: (num_envs, n_agents, obs_dim) -> 展平为 (num_envs * n_agents, obs_dim)
                    flat_obs = obs.reshape(-1, obs_dim)
                    actions, log_probs = agent.get_action(flat_obs)
                    
                    actions_reshaped = actions.reshape(args.num_envs, n_agents, act_dim)
                    # Critic 接收全局状态 (如果使用简单 Critic 则为每个智能体重复)
                    # 或者 Attention Critic 接收特殊输入
                    
                    # 集中式 Critic: 输入是全局状态 (为每个智能体重复)
                    state_rep = global_state.unsqueeze(1).repeat(1, n_agents, 1).reshape(-1, state_dim)
                    values = agent.get_value(state_rep)

                elif args.algo == "ippo":
                    # IPPO: Actor 接收局部观测, Critic 接收局部观测
                    flat_obs = obs.reshape(-1, obs_dim)
                    actions, log_probs = agent.get_action(flat_obs)
                    actions_reshaped = actions.reshape(args.num_envs, n_agents, act_dim)
                    values = agent.get_value(flat_obs)

            # 环境步进
            actions_list = [actions_reshaped[:, i, :] for i in range(n_agents)]
            next_obs, rewards, terminated, truncated, _ = env.step(actions_list)
            dones = terminated | truncated  # 合并用于重置逻辑
            
            # 视频录制: 只在课程学习完成后且未达到最大视频数时录制
            if not args.no_video and curriculum_done and video_count < max_videos and recording_env_idx is not None and not video_saved_this_update:
                frame = env.render(mode="rgb_array", env_index=recording_env_idx, visualize_when_rgb=True)
                frame_list.append(frame)
            
            # 自动重置逻辑
            # 先更新 running_return (累加本步奖励)
            step_rewards = torch.stack(rewards, dim=1).sum(dim=1)  # (num_envs,) 所有智能体奖励之和
            running_return += step_rewards
            
            # 分别统计真正完成和超时
            terminated_indices = torch.where(terminated)[0]
            truncated_indices = torch.where(truncated & ~terminated)[0]  # 超时但非真正完成
            terminated_count += len(terminated_indices)
            truncated_count += len(truncated_indices)
            
            if dones.any():
                # dones: (num_envs,)
                done_indices = torch.where(dones)[0]
                
                # 记录完成的 Episode 回报
                episode_returns.extend(running_return[done_indices].tolist())
                # 重置这些环境的 running_return
                running_return[done_indices] = 0.0
                
                # 视频录制：只在课程学习完成后，且是真正完成（非超时）时录制
                if not args.no_video and curriculum_done and video_count < max_videos:
                    # 如果正在录制的环境真正完成了（terminated），保存视频
                    if recording_env_idx is not None and recording_env_idx in terminated_indices and not video_saved_this_update:
                        if len(frame_list) > 10:  # 至少有足够的帧
                            os.makedirs("assets", exist_ok=True)
                            video_name = f"assets/{args.algo}_{args.scenario}_success_{video_count}"
                            save_video(video_name, frame_list, fps=1 / env.scenario.world.dt)
                            print(f"Video saved: {video_name}.mp4 ({video_count + 1}/{max_videos})")
                            video_saved_this_update = True
                            video_count += 1
                            frame_list = []  # 清空帧列表
                    
                    # 如果正在录制的环境超时了（非成功），放弃录制，重新开始
                    if recording_env_idx is not None and recording_env_idx in truncated_indices:
                        recording_env_idx = None
                        frame_list = []
                    
                    # 如果还没开始录制，选择第一个真正完成的环境开始录制下一个 episode
                    if recording_env_idx is None and len(terminated_indices) > 0:
                        recording_env_idx = terminated_indices[0]
                        frame_list = []  # 开始新录制
                
                for idx in done_indices:
                    new_obs_list = env.reset_at(idx.item())
                    # new_obs_list 是张量列表 (每个智能体一个), 每个形状为 (num_envs, obs_dim)
                    # VMAS reset_at 返回所有环境的观测, 所以需要按索引取值
                    for agent_i, agent_obs_batch in enumerate(new_obs_list):
                        next_obs[agent_i][idx] = agent_obs_batch[idx]
            
            # --- 课程学习 ---
            # 将包裹质量从 5 逐渐增加到 50
            if args.scenario == "transport" and args.use_curriculum:
                target_mass = 50.0
                start_mass = 5.0
                
                current_mass = start_mass + (target_mass - start_mass) * min(1.0, update / args.curriculum_steps)
                
                # 直接应用到物理实体
                for package in env.scenario.packages:
                    package.mass = current_mass
            
                if update % 10 == 0:
                    wandb.log({"package_mass": current_mass}, step=total_steps)
            
            # 存储数据
            # 注意: 我们需要存储一致的形状用于更新

            if args.algo == "cppo":
                # CPPO 将全局状态存储给 Actor/Critic
                # 奖励需要求和以对应联合动作吗?
                # 论文说 "单一超级智能体"。通常优化团队奖励。
                # VMAS 奖励是按智能体的。对于 CPPO 我们将它们求和。
                # "它将多智能体问题视为单智能体问题... 一个超级智能体"
                # 所以是的, 求和奖励。
                
                rewards_tensor = torch.stack(rewards, dim=1).sum(dim=1) # (num_envs,)
                
                b_obs.append(global_state) # Actor 输入
                b_state.append(global_state) # Critic 输入
                b_actions.append(joint_actions)
                b_log_probs.append(log_probs)
                b_rewards.append(rewards_tensor)
                b_values.append(values.squeeze(-1)) # (num_envs,)
                
                # Dones: 任一智能体完成? 还是全局完成?
                # VMAS done 通常是按环境的。
                if isinstance(dones, torch.Tensor):
                     b_dones.append(dones.float())
                else:
                     b_dones.append(torch.stack(dones, dim=1).float().any(dim=1).float())

            else:
                # MAPPO/IPPO 存储每个智能体的数据
                b_obs.append(obs.reshape(-1, obs_dim)) # Actor 输入 (用于 MAPPO/IPPO)
                
                if args.algo == "mappo":
                    state_rep = global_state.unsqueeze(1).repeat(1, n_agents, 1).reshape(-1, state_dim)
                    b_state.append(state_rep) # Critic 输入
                else: # ippo
                    b_state.append(obs.reshape(-1, obs_dim)) # Critic 输入 (局部观测)

                b_actions.append(actions.reshape(-1, act_dim))
                b_log_probs.append(log_probs.reshape(-1))
                b_values.append(values.reshape(-1))
                
                # 每个智能体的奖励
                rewards_tensor = torch.stack(rewards, dim=1).reshape(-1) # (num_envs * n_agents)
                b_rewards.append(rewards_tensor)
                
                # 每个智能体的 Dones
                if isinstance(dones, torch.Tensor):
                     dones = dones.unsqueeze(1).repeat(1, n_agents).reshape(-1)
                else:
                     dones = torch.stack(dones, dim=1).reshape(-1)
                b_dones.append(dones.float())

            # 更新观测
            obs = torch.stack(next_obs, dim=1)
            total_steps += args.num_envs

        # --- GAE 和更新 ---
        with torch.no_grad():
            global_state = obs.reshape(args.num_envs, -1)
            
            if args.algo == "cppo":
                next_val = agent.get_value(global_state).reshape(-1)
            elif args.algo == "mappo":
                state_rep = global_state.unsqueeze(1).repeat(1, n_agents, 1).reshape(-1, state_dim)
                next_val = agent.get_value(state_rep).reshape(-1)
            else: # ippo
                next_val = agent.get_value(obs.reshape(-1, obs_dim)).reshape(-1)
        
        # 堆叠数据
        # 展平时间和批次维度 (T * N_Envs * N_Agents) 或 (T * N_Envs) 对于 CPPO
        b_obs = torch.stack(b_obs)
        b_state = torch.stack(b_state)
        b_actions = torch.stack(b_actions)
        b_log_probs = torch.stack(b_log_probs)
        b_rewards = torch.stack(b_rewards)
        b_values = torch.stack(b_values)
        b_dones = torch.stack(b_dones)
        
        advantages, returns = compute_gae(b_rewards, b_values, next_val, b_dones)
        
        # 用于展平维度以进行 PPO 更新的辅助函数
        def flatten(x):
            if x.dim() == 3:
                return x.reshape(-1, x.shape[-1])
            else:
                return x.reshape(-1)

        rollouts = {
            'obs': flatten(b_obs),       # Actor 输入
            'state': flatten(b_state),   # Critic 输入
            'actions': flatten(b_actions),
            'log_probs': flatten(b_log_probs),
            'returns': flatten(returns),
            'advantages': flatten(advantages)
        }
        
        actor_loss, val_loss, entropy = agent.update(rollouts)

        # 日志记录
        avg_step_reward = b_rewards.mean().item()
        max_reward = b_rewards.max().item()
        min_reward = b_rewards.min().item()
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        
        log_data = {
            "step": total_steps,
            "training_iteration": update,  # 论文图 4 的横轴
            "reward": avg_step_reward,
            "reward_max": max_reward,
            "reward_min": min_reward,
            "actor_loss": actor_loss,
            "value_loss": val_loss,
            "entropy": entropy,
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "terminated_count": terminated_count,  # 真正通关的次数
            "truncated_count": truncated_count,  # 超时的次数
            "mean_episode_return": np.mean(episode_returns) if len(episode_returns) > 0 else 0.0  # 论文图 4 的纵轴
        }
        wandb.log(log_data)
        
        if update % args.log_interval == 0:
            mass_str = f", Mass: {env.scenario.packages[0].mass:.2f}" if args.scenario == "transport" else ""
            ep_ret_str = f", EpRet: {np.mean(episode_returns):.2f}" if len(episode_returns) > 0 else ", EpRet: N/A"
            print(f"Update {update}/{num_updates}, Step {total_steps}, "
                  f"Reward: {avg_step_reward:.4f} (max: {max_reward:.2f}, min: {min_reward:.2f}), "
                  f"Actor: {actor_loss:.4f}, Val: {val_loss:.4f}, Ent: {entropy:.4f}, "
                  f"Term: {terminated_count}, Trunc: {truncated_count}{ep_ret_str}{mass_str}")

    wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="transport", help="VMAS scenario name")
    parser.add_argument("--algo", type=str, default="ippo", choices=["ippo", "mappo", "cppo"], help="Algorithm")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of vectorized environments")
    parser.add_argument("--max_steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--steps_per_update", type=int, default=100, help="Steps per PPO update")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of PPO epochs per update")
    parser.add_argument("--num_minibatches", type=int, default=8, help="Number of minibatches per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=8, help="Log interval")
    parser.add_argument("--render_update", type=int, default=None, help="Update number at which to record a video (None to disable)")
    parser.add_argument("--render_steps", type=int, default=100, help="Number of steps to record for video")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="Max steps per episode before timeout (prevents zombie envs)")
    parser.add_argument("--use_curriculum", default=True, action="store_true", help="Enable curriculum learning (gradually increase package mass)")
    parser.add_argument("--curriculum_steps", type=int, default=100, help="Number of updates for curriculum learning")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for better performance")

    
    args = parser.parse_args()
    train(args)

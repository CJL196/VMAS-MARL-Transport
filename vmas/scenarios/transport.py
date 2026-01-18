#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transport 场景实现
==================
本场景是 VMAS 论文中的核心多智能体协作任务之一。

任务描述:
    N 个智能体需要协作将 M 个包裹推到目标位置。
    由于包裹质量较大（默认 50），单个智能体无法独自推动，
    必须多个智能体同时施力才能完成任务。

核心挑战:
    1. 协作探索: 智能体需要学会同时靠近包裹并协调施力方向
    2. 信用分配: 所有智能体共享奖励，需要学会分工
    3. 高维状态空间: 随着智能体数量增加，协调难度指数增长

论文实验结果:
    在此场景中，IPPO 表现最好，因为分布式策略更容易进行初期探索；
    CPPO 因联合状态空间过大而难以泛化；
    MAPPO 介于两者之间。
"""

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    """
    Transport 场景类，继承自 BaseScenario。
    
    实现了 VMAS 场景接口的五个核心方法:
    - make_world: 创建世界、智能体和地标
    - reset_world_at: 重置指定环境
    - reward: 计算奖励
    - observation: 返回观测
    - done: 判断是否完成
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        创建仿真世界，包括智能体、包裹和目标。
        
        Args:
            batch_dim: 并行环境数量，VMAS 的核心特性是向量化，
                       可以同时模拟数千个环境
            device: 运行设备 (CPU/GPU)
            **kwargs: 场景配置参数
        
        Returns:
            World: 配置好的仿真世界对象
        
        场景配置参数:
            n_agents: 智能体数量，默认 4
            n_packages: 包裹数量，默认 1
            package_width: 包裹宽度，默认 0.15
            package_length: 包裹长度，默认 0.15
            package_mass: 包裹质量，默认 50（需要多智能体协作）
        """
        # =================================================================
        # 1. 解析场景参数
        # =================================================================
        n_agents = kwargs.pop("n_agents", 4)              # 智能体数量
        self.n_packages = kwargs.pop("n_packages", 1)     # 包裹数量
        self.package_width = kwargs.pop("package_width", 0.15)   # 包裹宽度
        self.package_length = kwargs.pop("package_length", 0.15) # 包裹长度
        self.package_mass = kwargs.pop("package_mass", 50)       # 包裹质量（关键参数）
        ScenarioUtils.check_kwargs_consumed(kwargs)  # 检查是否有未使用的参数

        # 奖励塑形系数：将距离转换为奖励的比例因子
        # 较大的值会产生更强的梯度信号
        self.shaping_factor = 100
        
        # 世界半边长：定义智能体可活动的范围 [-1, 1]
        self.world_semidim = 1
        
        # 智能体半径
        self.agent_radius = 0.03

        # =================================================================
        # 2. 创建仿真世界
        # =================================================================
        # World 是 VMAS 的核心类，管理所有物理实体和仿真步进
        # x_semidim/y_semidim 定义了世界边界，超出边界的物体会被限制
        world = World(
            batch_dim,  # 并行环境数量
            device,     # 运行设备
            x_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),  # 考虑实体大小的边界
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
        )
        
        # =================================================================
        # 3. 创建智能体
        # =================================================================
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),  # 圆形智能体
                u_multiplier=0.6,  # 动作力的乘数，控制智能体的最大推力
            )
            world.add_agent(agent)
            
        # =================================================================
        # 4. 创建目标点（goal）
        # =================================================================
        goal = Landmark(
            name="goal",
            collide=False,   # 目标不参与碰撞
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,  # 浅绿色表示目标
        )
        world.add_landmark(goal)
        
        # =================================================================
        # 5. 创建包裹（packages）
        # =================================================================
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}",
                collide=True,    # 包裹参与碰撞（与智能体碰撞产生推力）
                movable=True,    # 包裹可以被推动
                mass=self.package_mass,  # 质量决定了推动难度
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,  # 红色表示未到达目标
            )
            package.goal = goal  # 每个包裹关联其目标
            self.packages.append(package)
            world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        """
        重置指定环境（或所有环境）的状态。
        
        Args:
            env_index: 要重置的环境索引，None 表示重置所有环境
        
        重置逻辑:
            1. 随机放置所有智能体
            2. 随机放置目标和包裹（避开智能体位置）
            3. 初始化奖励塑形的基准值
        """
        # =================================================================
        # 1. 随机放置智能体
        # =================================================================
        # ScenarioUtils.spawn_entities_randomly 是 VMAS 提供的工具函数
        # 自动处理向量化环境的随机初始化，确保实体之间不重叠
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,  # 要放置的实体列表
            self.world,         # 世界对象
            env_index,          # 环境索引
            min_dist_between_entities=self.agent_radius * 2,  # 最小间距
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )
        
        # 记录智能体的位置，用于后续放置时避开
        agent_occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)

        # =================================================================
        # 2. 随机放置目标和包裹
        # =================================================================
        goal = self.world.landmarks[0]
        ScenarioUtils.spawn_entities_randomly(
            [goal] + self.packages,  # 目标和包裹一起放置
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages
            ),
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_occupied_positions,  # 避开智能体
        )

        # =================================================================
        # 3. 初始化奖励塑形基准值
        # =================================================================
        for package in self.packages:
            # 检查包裹是否已经在目标上（初始时通常不在）
            package.on_goal = self.world.is_overlapping(package, package.goal)

            # global_shaping 保存上一时刻的距离值，用于计算奖励差值
            # 这是 reward shaping 的关键：奖励 = 前一距离 - 当前距离
            if env_index is None:
                # 重置所有环境
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                # 只重置指定环境
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

    def reward(self, agent: Agent):
        """
        计算智能体的奖励。
        
        Args:
            agent: 当前计算奖励的智能体
        
        Returns:
            torch.Tensor: 形状为 (batch_dim,) 的奖励张量
        
        奖励设计 (Reward Shaping):
            - 使用势函数差值作为奖励: r = φ(s) - φ(s')
            - φ(s) = -distance(package, goal) * shaping_factor
            - 这样设计保证：包裹越接近目标，奖励越高
            - 所有智能体共享相同的团队奖励（合作设定）
        
        注意:
            由于所有智能体共享奖励，只需在第一个智能体时计算一次。
            这是 VMAS 的优化设计，避免重复计算。
        """
        # 只在第一个智能体时计算奖励（避免重复计算）
        is_first = agent == self.world.agents[0]

        if is_first:
            # 初始化奖励张量
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            for package in self.packages:
                # 计算包裹到目标的距离
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                
                # 检查包裹是否到达目标（用于 done 判断和颜色更新）
                package.on_goal = self.world.is_overlapping(package, package.goal)
                
                # 动态更新包裹颜色：红色=未到达，绿色=已到达
                package.color = torch.tensor(
                    Color.RED.value,
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value,
                    device=self.world.device,
                    dtype=torch.float32,
                )

                # === 核心奖励计算 ===
                # 当前距离的势函数值
                package_shaping = package.dist_to_goal * self.shaping_factor
                
                # 奖励 = 上一时刻势函数 - 当前势函数
                # 如果距离减小，奖励为正；距离增大，奖励为负
                # 只对未到达目标的环境计算奖励
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                
                # 更新上一时刻的势函数值
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        """
        返回智能体的观测。
        
        Args:
            agent: 需要获取观测的智能体
        
        Returns:
            torch.Tensor: 形状为 (batch_dim, obs_dim) 的观测张量
        
        观测空间设计:
            对于每个智能体，观测包含:
            - agent.state.pos: 智能体自身位置 (2,)
            - agent.state.vel: 智能体自身速度 (2,)
            对于每个包裹:
            - package.pos - goal.pos: 包裹到目标的相对位置 (2,)
            - package.pos - agent.pos: 包裹到智能体的相对位置 (2,)
            - package.vel: 包裹速度 (2,)
            - on_goal: 包裹是否到达目标 (1,)
        
        总维度: 4 + 7 * n_packages
        """
        # 收集所有包裹相关的观测
        package_obs = []
        for package in self.packages:
            # 包裹到目标的相对位置（用于判断任务进度）
            package_obs.append(package.state.pos - package.goal.state.pos)
            # 包裹到智能体的相对位置（用于导航到包裹）
            package_obs.append(package.state.pos - agent.state.pos)
            # 包裹的速度（用于预测运动轨迹）
            package_obs.append(package.state.vel)
            # 包裹是否已到达目标
            package_obs.append(package.on_goal.unsqueeze(-1))

        # 拼接所有观测
        return torch.cat(
            [
                agent.state.pos,  # 自身位置
                agent.state.vel,  # 自身速度
                *package_obs,     # 包裹相关观测
            ],
            dim=-1,
        )

    def done(self):
        """
        判断任务是否完成。
        
        Returns:
            torch.Tensor: 形状为 (batch_dim,) 的布尔张量
        
        完成条件:
            所有包裹都到达各自的目标位置
        """
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )


class HeuristicPolicy(BaseHeuristicPolicy):
    """
    手工设计的启发式策略，用于基准测试和调试。
    
    该策略使用 Hermite 样条曲线规划智能体的运动轨迹，
    引导智能体绕到包裹后方，然后将包裹推向目标。
    
    这个策略可以作为 MARL 算法的性能下界参考。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = 0.0  # 样条曲线上的评估点
        self.start_vel_dist_from_target_ratio = 0.5  # 起始速度方向计算参数
        self.start_vel_behind_ratio = 0.5  # 绕到目标后方的比例
        self.start_vel_mag = 1.0  # 起始速度大小
        self.hit_vel_mag = 1.0   # 撞击速度大小
        self.package_radius = 0.15 / 2  # 包裹半径估计
        self.agent_radius = -0.02  # 智能体半径（负值表示更靠近包裹）
        self.dribble_slowdown_dist = 0.0  # 接近时减速的距离阈值
        self.speed = 0.95  # 整体速度系数

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        """
        根据观测计算动作。
        
        Args:
            observation: 观测张量，格式与 Scenario.observation 一致
            u_range: 动作范围限制
        
        Returns:
            torch.Tensor: 形状为 (batch_dim, 2) 的动作张量
        """
        self.n_env = observation.shape[0]
        self.device = observation.device
        
        # 从观测中解析各个分量
        agent_pos = observation[:, :2]      # 智能体位置
        package_pos = observation[:, 6:8] + agent_pos  # 包裹绝对位置
        goal_pos = -observation[:, 4:6] + package_pos  # 目标绝对位置
        
        # 使用 dribble 策略计算控制输入
        control = self.dribble(agent_pos, package_pos, goal_pos)
        control *= self.speed * u_range
        
        return torch.clamp(control, -u_range, u_range)

    def dribble(self, agent_pos, package_pos, goal_pos, agent_vel=None):
        """
        运球策略：计算如何将包裹推向目标。
        
        策略思路:
            1. 计算包裹应该被推动的方向（指向目标）
            2. 计算智能体应该撞击包裹的位置（包裹后方）
            3. 使用 Hermite 样条规划到达撞击点的轨迹
        """
        # 包裹到目标的位移向量
        package_disp = goal_pos - package_pos
        ball_dist = package_disp.norm(dim=-1)
        direction = package_disp / ball_dist[:, None]
        
        # 计算撞击位置（包裹后方）
        hit_pos = package_pos - direction * (self.package_radius + self.agent_radius)
        hit_vel = direction * self.hit_vel_mag
        
        # 计算起始速度
        start_vel = self.get_start_vel(
            hit_pos, hit_vel, agent_pos, self.start_vel_mag * 2
        )
        
        # 接近目标时减速
        slowdown_mask = ball_dist <= self.dribble_slowdown_dist
        hit_vel[slowdown_mask, :] *= (
            ball_dist[slowdown_mask, None] / self.dribble_slowdown_dist
        )
        
        return self.get_action(
            target_pos=hit_pos,
            target_vel=hit_vel,
            curr_pos=agent_pos,
            curr_vel=agent_vel,
            start_vel=start_vel,
        )

    def hermite(self, p0, p1, p0dot, p1dot, u=0.0, deriv=0):
        """
        Hermite 样条插值。
        
        给定起点、终点及其导数，计算样条曲线上的点。
        这是一种平滑的轨迹规划方法。
        
        Args:
            p0: 起点位置
            p1: 终点位置
            p0dot: 起点速度
            p1dot: 终点速度
            u: 参数 t ∈ [0, 1]
            deriv: 求导阶数
        """
        u = u.reshape((-1,))

        U = torch.stack(
            [
                self.nPr(3, deriv) * (u ** max(0, 3 - deriv)),
                self.nPr(2, deriv) * (u ** max(0, 2 - deriv)),
                self.nPr(1, deriv) * (u ** max(0, 1 - deriv)),
                self.nPr(0, deriv) * (u**0),
            ],
            dim=1,
        ).float()
        
        # Hermite 基函数矩阵
        A = torch.tensor(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=U.device,
        )
        P = torch.stack([p0, p1, p0dot, p1dot], dim=1)
        ans = U[:, None, :] @ A[None, :, :] @ P
        ans = ans.squeeze(1)
        return ans

    def nPr(self, n, r):
        """计算排列数 P(n, r) = n! / (n-r)!"""
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans

    def get_start_vel(self, pos, vel, start_pos, start_vel_mag):
        """
        计算起始速度向量。
        
        策略：让智能体绕到包裹后方，而不是直接冲向包裹。
        """
        start_vel_mag = torch.as_tensor(start_vel_mag, device=self.device).view(-1)
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        goal_dir = goal_disp / goal_dist[:, None]

        # 计算垂直于速度方向的向量
        vel_dir_normal = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
        dot_prod = (goal_dir * vel_dir_normal).sum(dim=1)
        vel_dir_normal[dot_prod > 0, :] *= -1

        dist_behind_target = self.start_vel_dist_from_target_ratio * goal_dist
        point_dir = -vel_dir * self.start_vel_behind_ratio + vel_dir_normal * (
            1 - self.start_vel_behind_ratio
        )

        target_pos = pos + point_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * start_vel_mag[:, None]
        return start_vel

    def get_action(
        self,
        target_pos,
        target_vel=None,
        start_pos=None,
        start_vel=None,
        curr_pos=None,
        curr_vel=None,
    ):
        """
        基于 Hermite 样条的轨迹跟踪控制。
        """
        if curr_pos is None:
            curr_pos = torch.zeros(target_pos.shape, device=self.device)
        if curr_vel is None:
            curr_vel = torch.zeros(target_pos.shape, device=self.device)
        if start_pos is None:
            start_pos = curr_pos
        if target_vel is None:
            target_vel = torch.zeros(target_pos.shape, device=self.device)
        if start_vel is None:
            start_vel = self.get_start_vel(
                target_pos, target_vel, start_pos, self.start_vel_mag * 2
            )

        u_start = torch.ones(curr_pos.shape[0], device=self.device) * self.lookahead
        
        # 计算期望位置和速度
        des_curr_pos = self.hermite(
            start_pos, target_pos, start_vel, target_vel, u=u_start, deriv=0
        )
        des_curr_vel = self.hermite(
            start_pos, target_pos, start_vel, target_vel, u=u_start, deriv=1
        )
        
        des_curr_pos = torch.as_tensor(des_curr_pos, device=self.device)
        des_curr_vel = torch.as_tensor(des_curr_vel, device=self.device)
        
        # PD 控制：位置误差 + 速度误差
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)

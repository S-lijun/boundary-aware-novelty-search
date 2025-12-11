"""
Boundary-aware Novelty Search (BANS) Model

按照论文公式实现：
1. Trajectory Embedding: b(τ)
2. Boundary Detection: 识别轨迹中的边界状态
3. Boundary-aware Novelty: N_b(τ) = (1/k) * Σ_i d(b(τ), NN_i(b(τ); A_safe))
4. Normalization: Ñ = (N - median(N)) / (IQR(N) + ε)
5. Reward Shaping: r''_t = r^ext_t + β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
6. Lambda Update: λ ← [λ + η(E[c_t] - c̄)]_+
"""

import numpy as np
import torch
from typing import List, Optional, Tuple


class BoundaryAwareNS:
    """
    Boundary-aware Novelty Search (BANS) 模型。
    
    按照论文实现：
    - 使用 safe archive A_safe = {τ | C(τ) ≤ c̄}
    - median-IQR 归一化
    - 奖励公式：r''_t = r^ext_t + β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
    - Lambda 自适应更新
    """

    def __init__(
        self,
        feature_extractor,  # RNDCNN 或其他特征提取器
        archive_k: int = 10,
        boundary_threshold: float = 0.1,
        device: str = "cpu",
        cost_threshold: float = 0.5,  # c̄: cost threshold for safe archive
        lambda_init: float = 0.1,  # 初始 lambda 值
        lambda_lr: float = 0.01,  # η: lambda 学习率
    ):
        """
        Args:
            feature_extractor: 特征提取器（如 RNDCNN），用于计算观测的 embedding
            archive_k: k-nearest neighbors 的 k 值
            boundary_threshold: 边界检测阈值（观测变化率）
            device: 计算设备
            cost_threshold: c̄，用于定义 safe archive 的 cost 阈值
            lambda_init: 初始 lambda 值
            lambda_lr: lambda 的学习率 η
        """
        self.feature_extractor = feature_extractor
        self.archive_k = archive_k
        self.boundary_threshold = boundary_threshold
        self.device = device
        self.cost_threshold = cost_threshold
        self.lambda_val = lambda_init
        self.lambda_lr = lambda_lr
        
        # Archive: 存储所有轨迹的 embedding 和累积 cost
        # 格式：[(z_boundary, cumulative_cost), ...]
        self.archive: List[Tuple[np.ndarray, float]] = []
        
        # Safe archive: A_safe = {τ | C(τ) ≤ c̄}
        self.safe_archive: List[np.ndarray] = []  # 只存储 embedding
        
        # Novelty history: 用于 median-IQR 归一化
        self.novelty_history: List[float] = []
        
        # Cost history: 用于更新 lambda
        self.cost_history: List[float] = []
        self.episode_costs: List[float] = []  # 每个 episode 的累积 cost
        
        # Novelty clip（可选）
        self.novelty_clip_min = None
        self.novelty_clip_max = None

    def detect_boundary_states(self, obs_seq: np.ndarray) -> np.ndarray:
        """
        检测轨迹中的边界状态。
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列，uint8 格式
        
        Returns:
            boundary_mask: (T,) bool array，True 表示该时刻是边界状态
        """
        T = len(obs_seq) - 1
        if T == 0:
            return np.array([], dtype=bool)
        
        boundary_mask = np.zeros(T, dtype=bool)
        
        # 计算相邻帧之间的差异
        for t in range(T):
            obs_t = obs_seq[t].astype(np.float32)
            obs_t_next = obs_seq[t + 1].astype(np.float32)
            
            # 计算观测变化率（归一化的平均绝对差异）
            diff = np.abs(obs_t - obs_t_next)
            change_rate = np.mean(diff) / 255.0  # 归一化到 [0, 1]
            
            # 如果变化率超过阈值，认为是边界状态
            if change_rate > self.boundary_threshold:
                boundary_mask[t] = True
        
        return boundary_mask

    def compute_boundary_embedding(
        self, obs_seq: np.ndarray, boundary_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算边界状态的 embedding。
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列，uint8 格式
            boundary_mask: (T,) bool array，边界状态掩码。如果为 None，则自动检测
        
        Returns:
            z_boundary: (feature_dim,) 边界状态的 embedding，已 L2 归一化
        """
        if boundary_mask is None:
            boundary_mask = self.detect_boundary_states(obs_seq)
        
        T = len(obs_seq) - 1
        if T == 0:
            raise ValueError("Empty observation sequence")
        
        if not np.any(boundary_mask):
            # 如果没有边界状态，使用整个轨迹的平均 embedding
            boundary_obs = obs_seq[:-1]  # (T, H, W, C)
        else:
            # 只使用边界状态的观测
            boundary_indices = np.where(boundary_mask)[0]
            boundary_obs = obs_seq[boundary_indices]  # (N_boundary, H, W, C)
        
        with torch.no_grad():
            # 归一化到 [0,1]，转 torch
            obs_tensor = torch.tensor(boundary_obs, dtype=torch.float32, device=self.device) / 255.0
            # 特征提取，使用 target_network（固定随机权重）
            target_features, _ = self.feature_extractor.forward(obs_tensor)
            # target_features: (N_boundary, feature_dim)
            feat_np = target_features.cpu().numpy()
            z = feat_np.mean(axis=0)  # 平均作为边界 embedding
            # L2 归一化
            norm = np.linalg.norm(z) + 1e-8
            z = z / norm
            return z

    def compute_cost(self, obs_seq: np.ndarray, actions: np.ndarray) -> float:
        """
        计算 cost 函数 c(s_t, a_t)。
        
        根据论文，cost 可以是：
        - 违反安全约束的惩罚
        - 能量消耗
        - 其他需要限制的行为
        
        这里实现一个简化版本：基于风险行为（面向墙、死胡同、停滞）
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列
            actions: (T,) 动作序列
        
        Returns:
            cost: float，累积 cost C(τ) = Σ_t c(s_t, a_t)
        """
        T = len(actions)
        if T == 0:
            return 0.0
        
        cumulative_cost = 0.0
        
        # 计算每步的 cost（简化版：基于风险行为）
        for t in range(T):
            step_cost = 0.0
            
            # Cost #1: 面向墙（高风险行为）
            if t < len(obs_seq) - 1:
                obs = obs_seq[t]
                if len(obs.shape) == 3:
                    H, W, C = obs.shape
                    center_h_start, center_h_end = H // 3, 2 * H // 3
                    center_w_start, center_w_end = W // 2, W
                    center_region = obs[center_h_start:center_h_end, center_w_start:center_w_end, :]
                    
                    if C == 3:  # RGB
                        gray = np.mean(center_region, axis=2)
                        dark_ratio = np.sum(gray < 50) / gray.size
                    else:
                        dark_ratio = np.sum(center_region < 50) / center_region.size
                    
                    if dark_ratio > 0.5:  # 面向墙
                        step_cost += 0.1
            
            # Cost #2: 重复动作（可能卡住）
            if t > 0 and actions[t] == actions[t - 1]:
                step_cost += 0.05
            
            cumulative_cost += step_cost
        
        return float(cumulative_cost)

    def compute_boundary_novelty(self, z_boundary: np.ndarray, use_safe_archive: bool = True) -> float:
        """
        计算 boundary-aware novelty。
        
        论文公式：N_b(τ) = (1/k) * Σ_i d(b(τ), NN_i(b(τ); A_safe))
        
        Args:
            z_boundary: (feature_dim,) 边界状态的 embedding
            use_safe_archive: 是否使用 safe archive（默认 True）
        
        Returns:
            novelty: float，边界新颖度
        """
        # 选择使用的 archive
        if use_safe_archive:
            archive_to_use = self.safe_archive
        else:
            # 如果没有 safe archive，使用普通 archive
            archive_to_use = [z for z, _ in self.archive]
        
        if len(archive_to_use) == 0:
            return 0.0
        
        A = np.array(archive_to_use)  # (N, dim)
        dists = np.linalg.norm(A - z_boundary[None, :], axis=1)
        k = min(self.archive_k, len(archive_to_use))
        topk = np.partition(dists, k - 1)[:k]
        return float(topk.mean())

    def normalize_novelty(self, novelty: float) -> float:
        """
        Median-IQR 归一化（按照论文公式）。
        
        论文公式：Ñ = (N - median(N)) / (IQR(N) + ε)
        
        Args:
            novelty: 原始新颖度值
        
        Returns:
            novelty_hat: 归一化后的新颖度（允许负值）
        """
        if len(self.novelty_history) <= 1:
            return 0.0
        
        # 计算 median 和 IQR
        med = np.median(self.novelty_history)
        q75 = np.percentile(self.novelty_history, 75)
        q25 = np.percentile(self.novelty_history, 25)
        iqr = q75 - q25
        
        if iqr < 1e-8:
            # IQR 太小，使用相对比较
            if med > 1e-8:
                novelty_hat = float((novelty - med) / (med + 1e-8))
            else:
                novelty_hat = 0.0
        else:
            novelty_hat = float((novelty - med) / (iqr + 1e-8))
        
        # 改进：确保 novelty_hat 有一个最小值，避免完全为0
        # 如果 novelty_hat 为负或0但 novelty > 0，给予一个小的基础值
        if novelty_hat <= 0.0 and novelty > 0:
            # 使用相对新颖度作为基础值
            if len(self.novelty_history) > 1:
                min_novelty = min(self.novelty_history)
                max_novelty = max(self.novelty_history)
                if max_novelty > min_novelty:
                    relative_novelty = (novelty - min_novelty) / (max_novelty - min_novelty + 1e-8)
                    novelty_hat = max(0.1, relative_novelty * 0.5)  # 至少 0.1，最多 0.5
                else:
                    novelty_hat = 0.1  # 基础值
            else:
                novelty_hat = 0.1  # 基础值
        
        # 可选：clip 到合理范围
        if self.novelty_clip_min is not None:
            novelty_hat = max(novelty_hat, self.novelty_clip_min)
        if self.novelty_clip_max is not None:
            novelty_hat = min(novelty_hat, self.novelty_clip_max)
        
        return float(novelty_hat)

    def update_lambda(self, expected_cost: float):
        """
        更新 lambda（按照论文公式）。
        
        论文公式：λ ← [λ + η(E[c_t] - c̄)]_+
        
        Args:
            expected_cost: E[c_t]，期望 cost（可以使用最近几个 episode 的平均 cost）
        """
        # λ ← [λ + η(E[c_t] - c̄)]_+
        delta = self.lambda_lr * (expected_cost - self.cost_threshold)
        self.lambda_val = max(0.0, self.lambda_val + delta)

    def process_episode(
        self,
        obs_seq: np.ndarray,
        actions: np.ndarray,
        add_to_archive: bool = True,
    ) -> dict:
        """
        处理一个完整的 episode，计算边界新颖度和归一化值。
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列，uint8 格式
            actions: (T,) 动作序列
            add_to_archive: 是否将边界 embedding 添加到 archive
        
        Returns:
            result: dict，包含以下键：
                - boundary_mask: (T,) bool array，边界状态掩码
                - z_boundary: (feature_dim,) 边界 embedding
                - novelty: float，原始新颖度（基于 safe archive）
                - novelty_hat: float，归一化后的新颖度
                - cumulative_cost: float，累积 cost C(τ)
        """
        # 1. 边界检测
        boundary_mask = self.detect_boundary_states(obs_seq)
        
        # 2. 计算边界 embedding
        z_boundary = self.compute_boundary_embedding(obs_seq, boundary_mask)
        
        # 3. 计算累积 cost
        cumulative_cost = self.compute_cost(obs_seq, actions)
        self.episode_costs.append(cumulative_cost)
        
        # 4. 更新 safe archive：A_safe = {τ | C(τ) ≤ c̄}
        # 先添加到普通 archive
        if add_to_archive:
            self.archive.append((z_boundary.astype(np.float32), cumulative_cost))
            
            # 如果 cost 低于阈值，添加到 safe archive
            if cumulative_cost <= self.cost_threshold:
                self.safe_archive.append(z_boundary.astype(np.float32))
        
        # 5. 计算 boundary-aware novelty（使用 safe archive）
        novelty = self.compute_boundary_novelty(z_boundary, use_safe_archive=True)
        
        # 6. 更新历史并归一化
        # 第一个 episode 特殊处理
        if len(self.archive) == 1:
            novelty = 1.0  # 给予基础值
        
        self.novelty_history.append(novelty)
        novelty_hat = self.normalize_novelty(novelty)
        
        # 7. 更新 lambda（使用最近几个 episode 的平均 cost）
        if len(self.episode_costs) >= 10:
            expected_cost = float(np.mean(self.episode_costs[-10:]))  # 最近10个episode的平均
            self.update_lambda(expected_cost)
        
        return {
            "boundary_mask": boundary_mask,
            "z_boundary": z_boundary,
            "novelty": novelty,
            "novelty_hat": novelty_hat,
            "cumulative_cost": cumulative_cost,
        }

    def compute_intrinsic_reward(
        self,
        obs_seq: np.ndarray,
        actions: np.ndarray,
        beta: float = 2.0,  # β(s_t, t)，论文中的权重因子
        episode_length: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        计算BANS内在奖励（按照论文公式）。
        
        论文公式：r''_t = r^ext_t + β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
        
        注意：这里返回的是内在奖励部分：β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
        以及每步的 cost 惩罚：λc(s_t, a_t)
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列
            actions: (T,) 动作序列
            beta: β(s_t, t)，权重因子（论文中可能随时间/状态变化，这里简化为常数）
            episode_length: episode 长度。如果为 None，使用 actions 的长度
        
        Returns:
            (intrinsic_reward, cost_penalty): 
                - intrinsic_reward: β(s_t, t) Ñ_b(τ)（每步的奖励）
                - cost_penalty: λc(s_t, a_t)（每步的惩罚）
        """
        result = self.process_episode(obs_seq, actions, add_to_archive=True)
        
        T = episode_length if episode_length is not None else len(actions)
        
        # 论文公式：r''_t = r^ext_t + β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
        # 内在奖励部分：β(s_t, t) Ñ_b(τ)
        # 这里简化：β(s_t, t) = β（常数），每步加奖励
        # 如果要在 episode 结束时加，可以乘以 T
        novelty_bonus = beta * result["novelty_hat"]
        
        # Cost 惩罚部分：λc(s_t, a_t)
        # 累积 cost 平均到每步
        cost_per_step = result["cumulative_cost"] / max(T, 1)
        cost_penalty = self.lambda_val * cost_per_step
        
        # 每步的内在奖励（正值）和惩罚（负值）
        intrinsic_reward_per_step = novelty_bonus - cost_penalty
        
        # 如果要在 episode 结束时一次性加，返回总和
        intrinsic_reward_total = intrinsic_reward_per_step * T
        
        return float(intrinsic_reward_total), float(cost_penalty * T)

    def reset(self):
        """重置 archive 和历史记录（用于新的训练任务）。"""
        self.archive = []
        self.safe_archive = []
        self.novelty_history = []
        self.cost_history = []
        self.episode_costs = []
        self.lambda_val = 0.1  # 重置为初始值

    def get_archive_size(self) -> int:
        """返回 archive 的大小。"""
        return len(self.archive)

    def get_safe_archive_size(self) -> int:
        """返回 safe archive 的大小。"""
        return len(self.safe_archive)

    def get_lambda(self) -> float:
        """返回当前的 lambda 值。"""
        return self.lambda_val

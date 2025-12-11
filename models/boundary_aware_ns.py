"""
Boundary-aware Novelty Search (BANS) Model

按照论文公式实现：
1. Trajectory Embedding: b(τ)
2. Boundary Detection: 识别轨迹中的边界状态
3. Boundary-aware Novelty: N̂b(τ) = (1/k) * Σ_i d(b(τ_boundary), NN_i(b(τ_boundary); A))
4. Normalization: N̂ = (N - median(N)) / (IQR(N) + ε)
5. Reward Shaping: r''t = r_t^ext + β(st, t) N̂b(τ) - λc(st, at)
"""

import numpy as np
import torch
from typing import List, Optional


class BoundaryAwareNS:
    """
    Boundary-aware Novelty Search (BANS) 模型。
    
    用于计算边界状态的新颖度，并提供奖励塑形功能。
    """

    def __init__(
        self,
        feature_extractor,  # RNDCNN 或其他特征提取器
        archive_k: int = 10,
        boundary_threshold: float = 0.1,
        device: str = "cpu",
        novelty_clip_min: Optional[float] = None,  # 新颖度下界，None表示不限制
        novelty_clip_max: Optional[float] = None,  # 新颖度上界，None表示不限制
    ):
        """
        Args:
            feature_extractor: 特征提取器（如 RNDCNN），用于计算观测的 embedding
            archive_k: k-nearest neighbors 的 k 值
            boundary_threshold: 边界检测阈值（观测变化率）
            device: 计算设备
            novelty_clip_min: 归一化后新颖度的下界（默认None，允许负值）
                            - 如果设为0.0，等价于ReLU
                            - 如果设为-1.0，允许轻微负值但限制极端负值
                            - 如果设为None，不限制下界
            novelty_clip_max: 归一化后新颖度的上界（默认None，不限制上界）
        """
        self.feature_extractor = feature_extractor
        self.archive_k = archive_k
        self.boundary_threshold = boundary_threshold
        self.device = device
        self.novelty_clip_min = novelty_clip_min
        self.novelty_clip_max = novelty_clip_max
        
        # Archive: 存储历史边界状态的 embedding
        self.archive: List[np.ndarray] = []
        
        # Novelty history: 用于运行均值归一化（适配Maze环境）
        self.novelty_history: List[float] = []
        
        # 运行统计（用于均值-标准差归一化，适配Maze环境）
        self.novelty_mean = 0.0
        self.novelty_std = 1.0
        self.novelty_count = 0

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

    def compute_boundary_novelty(self, z_boundary: np.ndarray) -> float:
        """
        计算 boundary-aware novelty。
        
        公式：N(b(τ_boundary)) = (1/k) * Σ_i d(b(τ_boundary), NN_i(b(τ_boundary); A))
        
        Args:
            z_boundary: (feature_dim,) 边界状态的 embedding
        
        Returns:
            novelty: float，边界新颖度
        """
        if len(self.archive) == 0:
            return 0.0
        
        A = np.array(self.archive)  # (N, dim)
        dists = np.linalg.norm(A - z_boundary[None, :], axis=1)
        k = min(self.archive_k, len(self.archive))
        topk = np.partition(dists, k - 1)[:k]
        return float(topk.mean())

    def normalize_novelty(self, novelty: float) -> float:
        """
        运行均值-标准差归一化（适配Maze环境）。
        
        公式：N̂ = (N - mean(N)) / (std(N) + ε)
        然后使用ReLU确保非负：N̂ = max(0, N̂)
        
        Args:
            novelty: 原始新颖度值
        
        Returns:
            novelty_hat: 归一化后的新颖度（>= 0）
        """
        if self.novelty_count == 0:
            return 0.0
        
        # 归一化：n_norm = (n - mean) / (std + eps)
        if self.novelty_count > 1 and self.novelty_std > 1e-8:
            novelty_norm = (novelty - self.novelty_mean) / (self.novelty_std + 1e-8)
        else:
            novelty_norm = 0.0
        
        # ReLU确保非负（Maze环境关键：避免负奖励堆积）
        novelty_hat = max(0.0, novelty_norm)
        
        return float(novelty_hat)

    def compute_risk(self, obs_seq: np.ndarray, actions: np.ndarray) -> float:
        """
        计算探索风险 R(s) ∈ [0, 1]，用于BANS的乘法缩放。
        
        根据文档，Maze环境的risk包括：
        1. Agent facing a wall: risk = 1.0
        2. Agent in deadend: risk = 0.6
        3. Agent stagnating: risk = 0.3
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列
            actions: (T,) 动作序列
        
        Returns:
            risk: float，风险值 [0, 1]
        """
        T = len(actions)
        if T == 0:
            return 0.0
        
        risk = 0.0
        
        # Risk #1: Agent stagnating（长时间不前进）
        # 检测最后10步的位置变化
        if T >= 10:
            # 使用观测的简单hash来检测位置变化
            obs_start = obs_seq[max(0, T-10)].flatten()
            obs_end = obs_seq[T].flatten()
            # 计算观测差异
            obs_diff = np.mean(np.abs(obs_start - obs_end))
            if obs_diff < 0.01:  # 几乎没有变化
                risk = max(risk, 0.3)
        
        # Risk #2: 重复动作过多（可能卡在墙边）
        repeated_actions = 0
        for t in range(1, T):
            if actions[t] == actions[t - 1]:
                repeated_actions += 1
        repeat_ratio = repeated_actions / max(T, 1)
        if repeat_ratio > 0.7:  # 70%以上动作重复
            risk = max(risk, 0.6)  # deadend-like behavior
        
        # Risk #3: 检测是否面向墙（简化版：基于观测边缘的像素值）
        # Procgen Maze中，墙通常是深色像素
        # 检测观测中心区域（agent前方）是否有大量深色像素
        last_obs = obs_seq[-1]  # (H, W, C)
        if len(last_obs.shape) == 3:
            H, W, C = last_obs.shape
            # 检测中心区域（agent前方）
            center_h_start, center_h_end = H // 3, 2 * H // 3
            center_w_start, center_w_end = W // 2, W
            center_region = last_obs[center_h_start:center_h_end, center_w_start:center_w_end, :]
            
            # 计算深色像素比例（假设墙是低亮度）
            if C == 3:  # RGB
                gray = np.mean(center_region, axis=2)
                dark_ratio = np.sum(gray < 50) / gray.size  # 像素值 < 50 认为是墙
            else:
                dark_ratio = np.sum(center_region < 50) / center_region.size
            
            if dark_ratio > 0.5:  # 50%以上是深色（可能是墙）
                risk = max(risk, 1.0)  # facing wall
        
        return float(min(risk, 1.0))  # 确保在[0,1]范围内

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
                - novelty: float，原始新颖度
                - novelty_hat: float，归一化后的新颖度
                - risk: float，探索风险 [0, 1]
        """
        # 1. 边界检测
        boundary_mask = self.detect_boundary_states(obs_seq)
        
        # 2. 计算边界 embedding
        z_boundary = self.compute_boundary_embedding(obs_seq, boundary_mask)
        
        # 3. 计算 boundary-aware novelty
        novelty = self.compute_boundary_novelty(z_boundary)
        
        # 4. 更新历史并归一化
        self.novelty_history.append(novelty)
        
        # 使用Welford's online algorithm更新均值和标准差
        self.novelty_count += 1
        delta = novelty - self.novelty_mean
        self.novelty_mean += delta / self.novelty_count
        delta2 = novelty - self.novelty_mean
        # 使用样本标准差
        self.novelty_std = np.sqrt(
            ((self.novelty_count - 1) * self.novelty_std**2 + delta * delta2) / max(self.novelty_count, 1)
        )
        
        novelty_hat = self.normalize_novelty(novelty)
        
        # 5. 添加到 archive
        if add_to_archive:
            self.archive.append(z_boundary.astype(np.float32))
        
        # 6. 计算探索风险（用于BANS乘法缩放）
        risk = self.compute_risk(obs_seq, actions)
        
        return {
            "boundary_mask": boundary_mask,
            "z_boundary": z_boundary,
            "novelty": novelty,
            "novelty_hat": novelty_hat,
            "risk": risk,
        }

    def compute_intrinsic_reward(
        self,
        obs_seq: np.ndarray,
        actions: np.ndarray,
        alpha: float = 0.05,  # NS基础奖励系数（Maze推荐值）
        beta: float = 0.25,  # BANS风险缩放系数
        episode_length: Optional[int] = None,
    ) -> float:
        """
        计算BANS内在奖励（使用乘法缩放而非减法惩罚）。
        
        公式：r_BANS = r_NS * (1 - β * R(s))
        其中：r_NS = α * N̂b(τ) * T
        
        这确保奖励永远 >= 0，且在危险区域降低NS强度。
        
        Args:
            obs_seq: (T+1, H, W, C) 观测序列
            actions: (T,) 动作序列
            alpha: NS基础奖励系数（Maze推荐0.05）
            beta: BANS风险缩放系数（推荐0.1-0.3）
            episode_length: episode 长度（用于缩放奖励）。如果为 None，使用 actions 的长度
        
        Returns:
            intrinsic_reward: float，内在奖励（>= 0）
        """
        result = self.process_episode(obs_seq, actions, add_to_archive=True)
        
        T = episode_length if episode_length is not None else len(actions)
        
        # NS基础奖励：α * N̂b(τ) * T
        r_ns = alpha * result["novelty_hat"] * T
        
        # BANS缩放：r_BANS = r_NS * (1 - β * R(s))
        # 确保 (1 - β * R(s)) >= 0，即 R(s) <= 1/β
        # 由于 R(s) ∈ [0,1] 且 β ∈ [0.1, 0.3]，所以 1 - β * R(s) >= 0.7 > 0
        risk_scale = 1.0 - beta * result["risk"]
        risk_scale = max(0.0, risk_scale)  # 安全保护，确保非负
        
        intrinsic_reward = r_ns * risk_scale
        
        return float(intrinsic_reward)

    def reset(self):
        """重置 archive 和历史记录（用于新的训练任务）。"""
        self.archive = []
        self.novelty_history = []
        self.novelty_mean = 0.0
        self.novelty_std = 1.0
        self.novelty_count = 0

    def get_archive_size(self) -> int:
        """返回 archive 的大小。"""
        return len(self.archive)


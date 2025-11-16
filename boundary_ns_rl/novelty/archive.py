import numpy as np


class BehaviorArchive:
    """
    简单的行为归档（Archive）
    用于基于 embedding z 计算 novelty。

    archive 存储所有历史 embedding。
    novelty = 平均最近 k 个邻居的距离。
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.archive = []  # 存储所有过去的 z

    def add(self, z: np.ndarray):
        """
        添加一条 embedding。
        z: shape (latent_dim,)
        """
        self.archive.append(z.astype(np.float32))

    def compute_novelty(self, z: np.ndarray) -> float:
        """
        基于 Euclidean distance 计算 novelty。
        如果 archive 为空 → novelty = 0。
        """
        if len(self.archive) == 0:
            return 0.0

        A = np.array(self.archive)  # (N, latent_dim)
        z = z.reshape(1, -1)

        dists = np.linalg.norm(A - z, axis=1)  # (N,)

        k = min(self.k, len(self.archive))
        topk = np.partition(dists, k-1)[:k]     # k 个最近邻
        return float(topk.mean())

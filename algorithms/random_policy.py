"""
随机弱基线：在可行动作集合上均匀采样 (Random Policy)

该策略不利用任何缓存、时延或负载信息，只在满足内存约束的节点中随机选择一个。
它作为论文中的下界对照，体现"不感知系统状态"时的性能水平。
"""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """
    随机路由策略，接口与 SB3 的 .predict() 兼容。

    为避免因无效动作过多而让基线失真，该策略优先在可行节点中均匀采样；
    若所有节点都不可行，则退化为全节点随机采样，由环境施加无效动作惩罚。
    """

    def __init__(self, env, seed: int | None = 42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def _get_core_env(self):
        """解包 DummyVecEnv，返回底层 ServerlessEdgeEnv。"""
        return self.env.envs[0] if hasattr(self.env, "envs") else self.env

    def predict(self, obs, deterministic: bool = True):
        """
        在当前请求对应的可行节点集合中随机选取一个目标节点。

        Parameters
        ----------
        deterministic : bool
            保留该参数以兼容 SB3 接口；随机策略默认忽略该标志。
        """
        core_env = self._get_core_env()

        cursor = min(core_env.cursor, len(core_env.requests) - 1)
        request = core_env.requests[cursor]

        feasible_nodes = [
            n
            for n in range(core_env.num_nodes)
            if core_env._is_memory_feasible(node=n, request=request)
        ]

        candidate_nodes = (
            feasible_nodes if feasible_nodes else list(range(core_env.num_nodes))
        )
        action = int(self.rng.choice(candidate_nodes))
        return action, None

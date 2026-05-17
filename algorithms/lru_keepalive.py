"""
工业级强基线：缓存感知优先路由策略 (LRU Keepalive Policy)

策略语义与 AWS Lambda / Azure Functions 的实际行为一致：
  - 优先路由到已预热容器的节点（热启动，0 额外时延）
  - 其次路由到已缓存基础镜像层的节点（温启动，仅拉取业务代码）
  - 最后选剩余内存最大且可行的节点（冷启动，最小化内存溢出风险）

"LRU" 体现在环境内部的 keepalive TTL 机制上：容器按最近最少使用原则自动过期，
本策略在路由时充分利用这一缓存状态，是 PPO 需要超越的高质量对比基线。
"""

from __future__ import annotations

import numpy as np


class LRUKeepalivePolicy:
    """
    缓存感知贪心路由策略，接口与 SB3 的 .predict() 兼容。

    Parameters
    ----------
    env : ServerlessEdgeEnv 或 DummyVecEnv
        可传入原始环境，也可传入 DummyVecEnv 包装后的向量化环境。
    """

    def __init__(self, env):
        self.env = env

    def _get_core_env(self):
        """解包 DummyVecEnv，返回底层 ServerlessEdgeEnv。"""
        return self.env.envs[0] if hasattr(self.env, "envs") else self.env

    def predict(self, obs, deterministic: bool = True):
        """
        根据当前缓存状态，按热→温→冷三级规则选择目标节点。

        Returns
        -------
        action : int
            选择的节点编号
        state : None
            占位，保持与 SB3 policy.predict() 接口一致
        """
        core_env = self._get_core_env()

        cursor = min(core_env.cursor, len(core_env.requests) - 1)
        req = core_env.requests[cursor]
        fn_idx = req.function_idx
        layer_idx = req.layer_idx
        num_nodes = core_env.num_nodes

        def is_feasible(n: int) -> bool:
            return core_env._is_memory_feasible(node=n, request=req)

        def mem_used(n: int) -> float:
            return core_env._node_memory_used(n)

        # --- 规则一：热启动节点（已有函数容器 + 内存可行）---
        hot_nodes = [
            n for n in range(num_nodes)
            if core_env.container_count[n, fn_idx] > 0 and is_feasible(n)
        ]
        if hot_nodes:
            return int(hot_nodes[np.argmin([mem_used(n) for n in hot_nodes])]), None

        # --- 规则二：温启动节点（已缓存基础镜像层 + 内存可行）---
        warm_nodes = [
            n for n in range(num_nodes)
            if core_env.layer_cache[n, layer_idx] > 0 and is_feasible(n)
        ]
        if warm_nodes:
            return int(warm_nodes[np.argmin([mem_used(n) for n in warm_nodes])]), None

        # --- 规则三：冷启动，只在可行节点里选剩余内存最大的 ---
        feasible_nodes = [n for n in range(num_nodes) if is_feasible(n)]
        if feasible_nodes:
            return int(feasible_nodes[np.argmin([mem_used(n) for n in feasible_nodes])]), None

        # --- 兜底：所有节点均不可行（极端内存压力），选已用内存最少的 ---
        # 环境会施加 invalid_action_penalty，但至少不会崩溃
        return int(np.argmin([mem_used(n) for n in range(num_nodes)])), None

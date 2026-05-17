"""
贪心弱基线：逐请求最小化即时系统代价 (Greedy Policy)

该策略不学习长期缓存演化，只在当前时刻基于环境的真实状态，估计每个节点的
单步总代价，并选择最优动作。它对应论文中的"短视启发式"基线，用于和 PPO 的
时序决策能力形成对比。
"""

from __future__ import annotations

import numpy as np


class GreedyPolicy:
    """
    单步贪心路由策略，接口与 SB3 的 .predict() 兼容。

    评分函数遵循 V3 数学建模里的即时优化目标：
        单步总代价 = 启动时延 + 传输时延 + 执行时延 + 动态内存惩罚

    与 PPO 的区别在于：Greedy 只看当前一步，不考虑未来请求序列对缓存状态的影响。
    """

    def __init__(self, env):
        self.env = env

    def _get_core_env(self):
        """解包 DummyVecEnv，返回底层 ServerlessEdgeEnv。"""
        return self.env.envs[0] if hasattr(self.env, "envs") else self.env

    def _estimate_dynamic_penalty(self, core_env, node: int, request) -> float:
        """
        估计把当前请求放到指定节点后产生的动态内存惩罚。

        这里不直接修改环境状态，而是按环境中的内存计算逻辑做一次前瞻估计，
        保证 Greedy 的行为与 `ServerlessEdgeEnv.step()` 的奖励语义一致。
        """
        memory_used = np.array(
            [core_env._node_memory_used(i) for i in range(core_env.num_nodes)],
            dtype=np.float32,
        )

        if core_env.container_count[node, request.function_idx] == 0:
            memory_used[node] += request.memory_mb
        if core_env.layer_cache[node, request.layer_idx] == 0:
            memory_used[node] += core_env.layer_size_mb[request.layer_idx]

        memory_ratio = np.clip(memory_used / core_env.node_memory_mb, 0.0, 1.0)
        system_pressure = float(memory_ratio.mean())
        beta_t = core_env.beta_base * np.exp(core_env.beta_growth * system_pressure)
        idle_ratio = system_pressure
        return float(beta_t * idle_ratio * 100.0)

    def _estimate_startup_latency(self, core_env, node: int, request) -> float:
        """
        无副作用地估计启动时延。

        不能直接调用环境的 `_startup_latency()`，因为那个方法会更新命中计数，
        进而污染评估统计结果。
        """
        fn = request.function_idx
        layer = request.layer_idx

        has_hot = core_env.container_count[node, fn] > 0
        has_layer = core_env.layer_cache[node, layer] > 0

        if has_hot:
            return 0.0
        if has_layer:
            return float(core_env.warm_code_pull_ms + core_env.warm_instantiate_ms)

        full_image_mb = request.memory_mb + core_env.layer_size_mb[layer]
        io_non_linear = core_env.io_penalty_coef * np.square(full_image_mb / 64.0)
        cold_ms = full_image_mb * core_env.cold_pull_per_mb_ms + io_non_linear
        return float(cold_ms)

    def _estimate_total_cost(self, core_env, node: int, request) -> float:
        """估计将当前请求调度到某节点的单步总代价。"""
        if not core_env._is_memory_feasible(node=node, request=request):
            return float(core_env.invalid_action_penalty)

        startup_ms = self._estimate_startup_latency(core_env, node, request)
        network_ms = core_env._network_latency(request.payload_mb)
        execution_ms = core_env._execution_latency(request.exec_ms)
        dynamic_penalty = self._estimate_dynamic_penalty(core_env, node, request)
        return float(startup_ms + network_ms + execution_ms + dynamic_penalty)

    def predict(self, obs, deterministic: bool = True):
        """
        穷举所有节点，选择即时总代价最小的动作。

        Returns
        -------
        action : int
            选择的节点编号
        state : None
            占位，保持与 SB3 policy.predict() 接口一致
        """
        core_env = self._get_core_env()

        cursor = min(core_env.cursor, len(core_env.requests) - 1)
        request = core_env.requests[cursor]
        costs = [
            self._estimate_total_cost(core_env=core_env, node=n, request=request)
            for n in range(core_env.num_nodes)
        ]
        best_node = int(np.argmin(np.asarray(costs, dtype=np.float32)))
        return best_node, None

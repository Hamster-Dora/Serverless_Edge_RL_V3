from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class Request:
    """Single request sample consumed by the environment at each step."""

    function_idx: int
    layer_idx: int
    memory_mb: float
    exec_ms: float
    payload_mb: float


class ServerlessEdgeEnv(gym.Env):
    """
    V3 environment with:
    - Three-state startup model: hot / warm / cold
    - Layer sharing cache matrix G_{n,l}
    - Dynamic beta memory pressure penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_df: pd.DataFrame,
        function_names: List[str] | None = None,
        layer_names: List[str] | None = None,
        num_nodes: int = 4,
        node_memory_mb: float = 8192.0,
        node_cpu_ghz: float = 3.2,
        node_bandwidth_mbps: float = 1000.0,
        keepalive_steps: int = 8,
        beta_base: float = 0.2,
        beta_growth: float = 3.0,
        io_penalty_coef: float = 0.02,
        warm_code_pull_ms: float = 12.0,
        warm_instantiate_ms: float = 18.0,
        cold_pull_per_mb_ms: float = 0.25,
        step_penalty_for_invalid_action: float = 1500.0,
        enable_layer_sharing: bool = True,
        enable_dynamic_beta: bool = True,
        fixed_beta_value: float | None = None,
        reward_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.num_nodes = int(num_nodes)
        self.node_memory_mb = float(node_memory_mb)
        self.node_cpu_ghz = float(node_cpu_ghz)
        self.node_bandwidth_mbps = float(node_bandwidth_mbps)
        self.keepalive_steps = int(keepalive_steps)

        self.beta_base = float(beta_base)
        self.beta_growth = float(beta_growth)
        self.io_penalty_coef = float(io_penalty_coef)
        self.enable_layer_sharing = bool(enable_layer_sharing)
        self.enable_dynamic_beta = bool(enable_dynamic_beta)
        self.fixed_beta_value = None if fixed_beta_value is None else float(fixed_beta_value)

        self.warm_code_pull_ms = float(warm_code_pull_ms)
        self.warm_instantiate_ms = float(warm_instantiate_ms)
        self.cold_pull_per_mb_ms = float(cold_pull_per_mb_ms)
        self.invalid_action_penalty = float(step_penalty_for_invalid_action)
        self.reward_scale = float(reward_scale)

        self.requests, self.function_names, self.layer_names = self._build_requests(
            trace_df,
            function_names=function_names,
            layer_names=layer_names,
        )
        self.num_functions = len(self.function_names)
        self.num_layers = len(self.layer_names)

        # Action: choose one target node for the incoming request.
        self.action_space = spaces.Discrete(self.num_nodes)

        # Observation follows the V3 requirement and explicitly contains G_{n,l}.
        self.observation_space = spaces.Dict(
            {
                "hot_containers": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_nodes, self.num_functions),
                    dtype=np.float32,
                ),
                "layer_cache": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_nodes, self.num_layers),
                    dtype=np.float32,
                ),
                "memory_usage_ratio": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_nodes,), dtype=np.float32
                ),
                "current_request": spaces.Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )

        self._reset_runtime_state()

    def _build_requests(
        self,
        trace_df: pd.DataFrame,
        function_names: List[str] | None = None,
        layer_names: List[str] | None = None,
    ) -> Tuple[List[Request], List[str], List[str]]:
        required_cols = {
            "HashFunction",
            "Layer_ID",
            "AverageAllocatedMb_pct50",
            "percentile_Average_50",
            "Invocations",
        }
        missing = required_cols.difference(trace_df.columns)
        if missing:
            raise ValueError(f"trace_df missing required columns: {sorted(missing)}")

        frame = trace_df.copy()
        frame["HashFunction"] = frame["HashFunction"].astype(str)
        frame["Layer_ID"] = frame["Layer_ID"].astype(str)
        frame["AverageAllocatedMb_pct50"] = frame["AverageAllocatedMb_pct50"].clip(lower=64).fillna(256)
        frame["percentile_Average_50"] = frame["percentile_Average_50"].clip(lower=1).fillna(50)
        frame["Invocations"] = frame["Invocations"].clip(lower=1).fillna(1).astype(int)

        if function_names is None:
            function_names = sorted(frame["HashFunction"].unique().tolist())
        else:
            function_names = [str(name) for name in function_names]

        if layer_names is None:
            layer_names = sorted(frame["Layer_ID"].unique().tolist())
        else:
            layer_names = [str(name) for name in layer_names]

        fn_to_idx = {name: i for i, name in enumerate(function_names)}
        layer_to_idx = {name: i for i, name in enumerate(layer_names)}

        unknown_functions = sorted(set(frame["HashFunction"].unique()).difference(fn_to_idx))
        unknown_layers = sorted(set(frame["Layer_ID"].unique()).difference(layer_to_idx))
        if unknown_functions or unknown_layers:
            details = []
            if unknown_functions:
                details.append(f"unknown HashFunction values: {unknown_functions[:5]}")
            if unknown_layers:
                details.append(f"unknown Layer_ID values: {unknown_layers[:5]}")
            raise ValueError(
                "trace_df contains functions/layers outside the fixed vocabulary. "
                "Please rebuild the vocabulary to cover both training and evaluation days. "
                + "; ".join(details)
            )

        # Approximate app payload from execution/memory to preserve monotonic behavior.
        payload_base = np.sqrt(frame["AverageAllocatedMb_pct50"].to_numpy(dtype=np.float32))
        frame["PayloadMB"] = np.clip(payload_base / 4.0, 1.0, 64.0)

        requests: List[Request] = []
        for row in frame.itertuples(index=False):
            repeat = int(min(row.Invocations, 20))
            for _ in range(repeat):
                requests.append(
                    Request(
                        function_idx=fn_to_idx[row.HashFunction],
                        layer_idx=layer_to_idx[row.Layer_ID],
                        memory_mb=float(row.AverageAllocatedMb_pct50),
                        exec_ms=float(row.percentile_Average_50),
                        payload_mb=float(row.PayloadMB),
                    )
                )

        if not requests:
            raise ValueError("No valid requests generated from trace_df.")

        return requests, function_names, layer_names

    def _reset_runtime_state(self) -> None:
        self.cursor = 0
        self.container_count = np.zeros((self.num_nodes, self.num_functions), dtype=np.int32)
        self.container_ttl = np.zeros((self.num_nodes, self.num_functions), dtype=np.int32)
        self.layer_cache = np.zeros((self.num_nodes, self.num_layers), dtype=np.int8)

        self.fn_memory_mb = np.zeros(self.num_functions, dtype=np.float32)
        for req in self.requests:
            self.fn_memory_mb[req.function_idx] = max(
                self.fn_memory_mb[req.function_idx], req.memory_mb
            )

        # Layer size is approximated from average function memory that depends on this layer.
        layer_sizes = np.full(self.num_layers, 96.0, dtype=np.float32)
        layer_max = np.zeros(self.num_layers, dtype=np.float32)
        layer_count = np.zeros(self.num_layers, dtype=np.float32)
        for req in self.requests:
            layer_max[req.layer_idx] = max(layer_max[req.layer_idx], req.memory_mb)
            layer_count[req.layer_idx] += 1.0
        valid = layer_count > 0
        layer_sizes[valid] = np.clip(layer_max[valid] * 0.35, 32.0, 512.0)
        self.layer_size_mb = layer_sizes

        self.total_latency_ms = 0.0
        self.hot_hits = 0
        self.warm_hits = 0
        self.cold_hits = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_runtime_state()
        return self._observation(), self._info()

    def step(self, action: int):
        node = int(action)
        request = self.requests[self.cursor]

        self._decay_keepalive()

        feasible = self._is_memory_feasible(node=node, request=request)
        if not feasible:
            latency_ms = self.invalid_action_penalty
            reward = -latency_ms * self.reward_scale
            info = self._info(start_state="invalid", latency_ms=latency_ms)
        else:
            startup_ms, start_state = self._startup_latency(node, request)
            network_ms = self._network_latency(request.payload_mb)
            execution_ms = self._execution_latency(request.exec_ms)
            latency_ms = network_ms + execution_ms + startup_ms

            self._apply_placement(node, request)
            dynamic_penalty = self._dynamic_memory_penalty()
            reward = -(latency_ms + dynamic_penalty) * self.reward_scale
            info = self._info(start_state=start_state, latency_ms=latency_ms)

        self.total_latency_ms += latency_ms
        self.cursor += 1
        terminated = self.cursor >= len(self.requests)
        truncated = False
        obs = self._observation() if not terminated else self._terminal_observation()
        return obs, float(reward), terminated, truncated, info

    def _startup_latency(self, node: int, request: Request) -> Tuple[float, str]:
        fn = request.function_idx
        layer = request.layer_idx

        has_hot = self.container_count[node, fn] > 0
        has_layer = self.enable_layer_sharing and self.layer_cache[node, layer] > 0

        if has_hot:
            self.hot_hits += 1
            return 0.0, "hot"
        if has_layer:
            self.warm_hits += 1
            warm_ms = self.warm_code_pull_ms + self.warm_instantiate_ms
            return warm_ms, "warm"

        self.cold_hits += 1
        full_image_mb = request.memory_mb + self.layer_size_mb[layer]
        io_non_linear = self.io_penalty_coef * np.square(full_image_mb / 64.0)
        cold_ms = full_image_mb * self.cold_pull_per_mb_ms + io_non_linear
        return float(cold_ms), "cold"

    def _dynamic_memory_penalty(self) -> float:
        memory_ratio = self._node_memory_usage_ratio()
        system_pressure = float(memory_ratio.mean())
        if self.enable_dynamic_beta:
            beta_t = self.beta_base * np.exp(self.beta_growth * system_pressure)
        else:
            beta_t = self.fixed_beta_value if self.fixed_beta_value is not None else self.beta_base

        # Idle waste is modeled as the current warm pool occupancy ratio.
        idle_ratio = system_pressure
        return float(beta_t * idle_ratio * 100.0)

    def _network_latency(self, payload_mb: float) -> float:
        mb_per_ms = max(self.node_bandwidth_mbps / 8.0 / 1000.0, 1e-4)
        return float(payload_mb / mb_per_ms)

    def _execution_latency(self, exec_ms: float) -> float:
        return float(exec_ms / max(self.node_cpu_ghz, 1e-3))

    def _is_memory_feasible(self, node: int, request: Request) -> bool:
        new_container = 1 if self.container_count[node, request.function_idx] == 0 else 0
        new_layer = (
            1
            if self.enable_layer_sharing and self.layer_cache[node, request.layer_idx] == 0
            else 0
        )

        current_used = self._node_memory_used(node)
        projected = (
            current_used
            + new_container * request.memory_mb
            + new_layer * self.layer_size_mb[request.layer_idx]
        )
        return projected <= self.node_memory_mb

    def _apply_placement(self, node: int, request: Request) -> None:
        fn = request.function_idx
        layer = request.layer_idx

        if self.enable_layer_sharing and self.layer_cache[node, layer] == 0:
            self.layer_cache[node, layer] = 1

        if self.container_count[node, fn] == 0:
            self.container_count[node, fn] = 1

        self.container_ttl[node, fn] = self.keepalive_steps

    def _decay_keepalive(self) -> None:
        active = self.container_count > 0
        self.container_ttl[active] -= 1
        expired = self.container_ttl <= 0
        self.container_count[expired] = 0
        self.container_ttl[expired] = 0

    def _node_memory_used(self, node: int) -> float:
        container_mem = (self.container_count[node] * self.fn_memory_mb).sum()
        layer_mem = (
            (self.layer_cache[node] * self.layer_size_mb).sum() if self.enable_layer_sharing else 0.0
        )
        return float(container_mem + layer_mem)

    def _node_memory_usage_ratio(self) -> np.ndarray:
        used = np.array([self._node_memory_used(i) for i in range(self.num_nodes)], dtype=np.float32)
        return np.clip(used / self.node_memory_mb, 0.0, 1.0)

    def _request_features(self, request: Request) -> np.ndarray:
        fn_ratio = request.function_idx / max(self.num_functions - 1, 1)
        layer_ratio = request.layer_idx / max(self.num_layers - 1, 1)
        mem_ratio = np.clip(request.memory_mb / self.node_memory_mb, 0.0, 1.0)
        exec_ratio = np.clip(request.exec_ms / 1000.0, 0.0, 1.0)
        return np.array([fn_ratio, layer_ratio, mem_ratio, exec_ratio], dtype=np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        current = self.requests[min(self.cursor, len(self.requests) - 1)]
        return {
            "hot_containers": (self.container_count > 0).astype(np.float32),
            "layer_cache": self.layer_cache.astype(np.float32),
            "memory_usage_ratio": self._node_memory_usage_ratio().astype(np.float32),
            "current_request": self._request_features(current),
        }

    def _terminal_observation(self) -> Dict[str, np.ndarray]:
        return {
            "hot_containers": (self.container_count > 0).astype(np.float32),
            "layer_cache": self.layer_cache.astype(np.float32),
            "memory_usage_ratio": self._node_memory_usage_ratio().astype(np.float32),
            "current_request": np.zeros((4,), dtype=np.float32),
        }

    def _info(self, start_state: str = "none", latency_ms: float = 0.0) -> Dict[str, float]:
        total_hits = self.hot_hits + self.warm_hits + self.cold_hits
        hit_rate = (self.hot_hits + self.warm_hits) / total_hits if total_hits > 0 else 0.0
        return {
            "cursor": float(self.cursor),
            "start_state": start_state,
            "latency_ms": float(latency_ms),
            "memory_pressure": float(self._node_memory_usage_ratio().mean()),
            "hot_hits": float(self.hot_hits),
            "warm_hits": float(self.warm_hits),
            "cold_hits": float(self.cold_hits),
            "cache_hit_rate": float(hit_rate),
            "avg_latency_ms": float(self.total_latency_ms / max(self.cursor, 1)),
        }

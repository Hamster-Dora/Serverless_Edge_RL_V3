# 基于强化学习的无服务器边缘计算资源调度优化算法设计与实现

> 面向 `Serverless Edge` 场景的毕业设计项目，聚焦冷启动、层缓存共享、容器保活与内存约束下的资源调度优化。

- 项目类型：毕业设计 / 强化学习 / 系统调度仿真
- 核心方法：`PPO + ServerlessEdgeEnv + 启发式基线对比 + 消融实验`
- 主要关注指标：平均时延、尾时延、缓存命中率、内存压力
- 仓库用途：工程实现、实验复现、答辩展示

## 仓库亮点

- 自定义 `ServerlessEdgeEnv`，显式建模 `hot / warm / cold` 三态启动过程
- 将镜像层共享与动态 `beta` 内存压力惩罚纳入统一调度环境
- 提供 `PPO / Greedy / LRU / Random` 的统一评测框架
- 支持多天训练、跨天评估、批量对比与消融实验
- 适合作为毕业设计代码仓库与后续论文实验支撑基础

## 项目简介

本项目面向无服务器边缘计算场景，研究函数请求在多边缘节点上的资源调度问题。围绕冷启动开销、镜像层复用、容器保活和节点内存约束等关键因素，项目构建了一个可复现的 `ServerlessEdgeEnv` 仿真环境，并在该环境中设计和实现了基于 PPO 的调度方法，同时与 `Greedy`、`LRU`、`Random` 等基线策略进行了统一对比和消融分析。

本仓库可作为毕业设计的工程实现、实验复现和答辩展示代码基础。

## 快速导航

- 训练入口：`main_train.py`
- 评估入口：`main_evaluate.py`
- 对比实验：`experiments/run_comparison.py`
- 消融实验：`experiments/run_ablation.py`
- 环境定义：`envs/serverless_env.py`
- 数据解析：`utils/trace_parser.py`

## 研究背景与问题

在 Serverless 场景下，函数实例通常按需启动。部署到边缘侧后，虽然能够降低网络传输时延，但也会带来新的调度挑战，例如：

- 边缘节点资源有限，尤其是内存约束更强
- 函数冷启动会显著增加请求时延
- 不同函数之间存在镜像层共享关系，可影响启动成本
- 若只追求平均时延，可能忽略尾时延和系统稳定性

因此，本项目希望回答的问题是：

**在考虑冷启动、层缓存共享和内存压力的前提下，能否利用强化学习学习到更合理的 Serverless Edge 资源调度策略。**

## 研究内容与方法设计

### 1. 环境建模

项目实现了 `envs/serverless_env.py` 中的 `ServerlessEdgeEnv`，核心建模包括：

- 三态启动机制：`hot / warm / cold`
- 镜像层缓存矩阵 `G_{n,l}`
- 容器保活时间 `keepalive_steps`
- 动态 `beta` 内存压力惩罚
- 严格内存可行性检查

环境输入使用 Azure Functions Trace 数据，模拟函数请求在多个边缘节点上的到达与调度过程。

### 2. 强化学习策略

项目采用 PPO 作为核心强化学习算法，使用 `MultiInputPolicy` 处理 Dict 观测空间，并支持：

- 单日或多日训练
- 固定词表，避免跨天训练/评估时观测维度漂移
- checkpoint 保存
- TensorBoard 日志记录

训练入口：`main_train.py`

### 3. 基线对比方法

为了验证 PPO 的有效性，项目实现了三类对比策略：

- `Random`：随机选择节点
- `Greedy`：基于贪心思想进行调度
- `LRU Keepalive`：基于保活与最近最少使用思想进行调度

统一评估入口：`main_evaluate.py`

### 4. 实验设计

项目支持两类实验：

- 对比实验：`experiments/run_comparison.py`
- 消融实验：`experiments/run_ablation.py`

消融实验主要考察以下设计因素：

- 完整模型 `FullV3`
- 去除镜像层共享 `NoLayerSharing`
- 将动态 `beta` 改为固定值 `FixedBeta`

## 当前阶段结论概述

基于当前版本的阶段性实验结果，可以得到以下现象性结论：

- PPO 在平均时延指标上不一定全面优于启发式基线
- PPO 在尾时延稳定性、缓存命中率等指标上表现出一定优势
- 镜像层共享机制对抑制长尾时延具有明显作用
- 动态 `beta` 的收益与训练预算、场景复杂度和流量规模有关，还值得继续深入分析

因此，当前项目更适合表述为：**通过强化学习与系统机制联合设计，在部分牺牲均值指标的前提下，换取更稳定的尾部表现与更合理的资源利用特征。**

## 项目结构

```text
.
├─ algorithms/              # 启发式基线策略
├─ data/                    # 原始数据目录（默认不上传 GitHub）
├─ envs/                    # 环境定义
├─ experiments/             # 批量对比与消融实验脚本
├─ results/                 # 实验结果、模型、图表（默认不上传 GitHub）
├─ utils/                   # 数据解析、词表、指标统计、绘图工具
├─ main_train.py            # PPO 训练入口
├─ main_evaluate.py         # 统一评估入口
├─ generate_paper_figures.py
├─ requirements.txt
└─ TODO.md                  # 项目推进记录与阶段性结论
```

## 运行环境

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖包括：

- `pandas`
- `numpy`
- `gymnasium`
- `torch`
- `stable-baselines3`
- `tensorboard`
- `tqdm`
- `matplotlib`

## 数据准备

默认数据目录为：

```text
data/azure_functions_trace/
```

解析脚本 `utils/trace_parser.py` 默认读取以下 Azure Trace CSV 文件：

- `invocations_per_function_md.anon.dXX.csv`
- `function_durations_percentiles.anon.dXX.csv`
- `app_memory_percentiles.anon.dXX.csv`

说明：

- 当前项目已兼容 `d13/d14` 缺失内存文件的情况
- 为避免仓库过大和实验资产过早公开，原始数据默认不上传 GitHub

## 快速开始

### 1. 训练 PPO

单日烟测示例：

```bash
python main_train.py --day d01 --rows 5000 --timesteps 100000 --nodes 3 --save-path results/models/ppo_smoke
```

多日训练示例：

```bash
python main_train.py --train-days d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12 --rows 50000 --timesteps 200000 --nodes 3 --save-path results/models/ppo_phase6_final
```

### 2. 统一评估

```bash
python main_evaluate.py --day d13 --rows 20000 --nodes 3 --model-path results/models/ppo_phase6_final.zip --policies ppo,lru,greedy,random
```

### 3. 运行批量对比实验

```bash
python experiments/run_comparison.py --days d13,d14 --rows 20000 --seeds 42 --model-path results/models/ppo_phase6_final.zip --tag github_demo
```

### 4. 运行批量消融实验

```bash
python experiments/run_ablation.py --train-days d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12 --eval-days d13,d14 --train-rows 50000 --eval-rows 50000 --timesteps 200000 --seeds 42 --tag github_demo
```

## GitHub 公开说明

考虑到项目目前仍处于答辩前阶段，本仓库默认采用“**代码公开，实验资产本地保留**”的管理方式。

当前默认不上传的内容包括：

- 原始 Azure Trace 数据
- `results/` 下的完整实验结果
- 模型权重与 checkpoints
- TensorBoard 日志
- 仅供内部写作和答辩使用的文档文件

这样做的目的是：

- 保留本地完整实验资产，避免误删
- 控制仓库体积，提升上传稳定性
- 在答辩前保留对公开范围的主动控制权

## 后续完善方向

- 引入更多启发式或强化学习基线
- 尝试 Action Masking，降低无效动作比例
- 增加节点异构性、CPU 约束和任务优先级建模
- 做更系统的多 seed 统计与复杂流量场景分析

## 说明

本仓库当前更适合作为毕业设计工程实现与实验复现支撑仓库。若在答辩结束后继续公开完善，可进一步补充：

- 英文摘要
- 关键实验图表
- 更正式的结果分析
- License 与 Release 信息

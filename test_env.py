import pandas as pd
from utils.trace_parser import AzureTraceParser
from envs.serverless_env import ServerlessEdgeEnv

# 1. 加载数据 (只取前 1000 条做测试)
print("🔍 正在加载测试数据...")
parser = AzureTraceParser()
df = parser.load_day_data("d01").iloc[:1000]

# 2. 初始化环境
env = ServerlessEdgeEnv(trace_df=df, num_nodes=3)
obs, info = env.reset()

print("🚀 开始运行冒烟测试...")
for i in range(20):
    # 模拟 AI 随机选一个节点
    action = env.action_space.sample() 
    
    # 执行一步
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 观察输出：重点看 start_state (启动状态) 和 latency (延迟)
    print(f"Step {i:02d} | 分配节点: {action} | 状态: {info['start_state']:4s} | 延迟: {info['latency_ms']:.2f}ms | 奖励: {reward:.2f}")

    if terminated:
        break

print("\n📊 测试统计结果:")
print(f"总点击数: {info['hot_hits'] + info['warm_hits'] + info['cold_hits']}")
print(f"缓存命中率: {info['cache_hit_rate']:.2%}")
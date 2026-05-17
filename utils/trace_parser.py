from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AzureTraceParser:
    def __init__(self, data_dir: Optional[str] = None):
        # 默认相对项目根目录，换机器也能跑通
        self.data_dir = data_dir or os.path.join(
            _PROJECT_ROOT, "data", "azure_functions_trace"
        )

    def load_day_data(self, day_index="d01"):
        print(f"[AzureTraceParser] 正在解析 {day_index} 数据流...")

        # 使用 os.path.join 进行最稳妥的路径拼接，绝不飘红
        invocations = pd.read_csv(os.path.join(self.data_dir, f"invocations_per_function_md.anon.{day_index}.csv"))
        durations = pd.read_csv(
            os.path.join(self.data_dir, f"function_durations_percentiles.anon.{day_index}.csv"),
            usecols=["HashFunction", "percentile_Average_50"],
        )

        memory_path = os.path.join(self.data_dir, f"app_memory_percentiles.anon.{day_index}.csv")
        if os.path.isfile(memory_path):
            memory = pd.read_csv(
                memory_path,
                usecols=["HashApp", "AverageAllocatedMb_pct50"],
            )
            mem_map = memory.drop_duplicates(subset=["HashApp"]).set_index("HashApp")["AverageAllocatedMb_pct50"]
        else:
            print(
                f"[AzureTraceParser] 警告: 未找到 `{memory_path}`，"
                "`AverageAllocatedMb_pct50` 将按未知处理并在后续填充为 0（例如 d13/d14 占位评估）。"
            )
            mem_map = pd.Series(dtype="float32")

        # 处理调用量长表
        id_cols = ['HashOwner', 'HashApp', 'HashFunction', 'Trigger']
        inv_long = pd.melt(invocations, id_vars=id_cols, var_name='Minute', value_name='Invocations')
        inv_long['Minute'] = pd.to_numeric(inv_long['Minute'], downcast='integer')
        inv_long['Invocations'] = pd.to_numeric(inv_long['Invocations'], downcast='integer')
        inv_long = inv_long[inv_long['Invocations'] > 0].copy()

        # 对超大长表避免反复 merge，直接 map 到目标列，减少内存峰值。
        dur_map = durations.drop_duplicates(subset=["HashFunction"]).set_index("HashFunction")["percentile_Average_50"]
        inv_long["AverageAllocatedMb_pct50"] = inv_long["HashApp"].map(mem_map)
        inv_long["percentile_Average_50"] = inv_long["HashFunction"].map(dur_map)
        df = inv_long

        # V3 核心逻辑：截取 Layer_ID
        df['Layer_ID'] = df['HashApp'].str[:4]
        df.fillna(0, inplace=True)
        df = df[df['Invocations'] > 0].sort_values(by='Minute')

        print(f"[AzureTraceParser] 解析完成，当前 Task 流共计 {len(df)} 条有效请求。")
        return df


def load_concat_trace_days(
    days: Iterable[str],
    rows_per_day: int = 0,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Concatenate multiple trace days into a single chronological stream (within each day's rows),
    tagging each row with ``SourceDay`` for debugging / experiment manifests.
    ``rows_per_day`` matches ``run_ablation`` / training scripts: ``0`` loads all rows for that day.
    """
    day_list = [d.strip() for d in days if str(d).strip()]
    if not day_list:
        raise ValueError("days must contain at least one non-empty day id.")

    parser = AzureTraceParser(data_dir=data_dir)
    frames = []
    for day in day_list:
        frame = parser.load_day_data(day)
        if rows_per_day and rows_per_day > 0:
            frame = frame.iloc[: rows_per_day].copy()
        frame["SourceDay"] = day
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = AzureTraceParser()
    sample_df = parser.load_day_data("d01")
    print(sample_df.head())
    
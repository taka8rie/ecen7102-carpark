"""
01_prepare.py — 第 1 步：清洗 + 重采样
输入：../cp2023.parquet（每 5 分钟一条的原始记录，2023 全年，5 个停车场）
输出：data/series_15min.parquet  每 15 分钟一行，每个序列一列
      results/data_quality.csv   每个序列的数据质量统计
      results/target_jumps.csv   主目标（P5 员工车位）5 分钟内跳变 ≥ 40 的记录
运行：python 01_prepare.py
"""
from common import *

SPIKE_THRESHOLD = 60   # 与前后 5 点中位数相差超过 60 个车位 → 视为孤立尖峰
FFILL_LIMIT = 2        # 最多向前填补 2 格（30 分钟）的缺失
JUMP_THRESHOLD = 40    # 记录 5 分钟内变化 ≥ 40 的跳变（不删除，只统计）

raw = pd.read_parquet(RAW_PARQUET)
raw = raw[raw["time"] < "2024-01-01"]                       # 丢掉跨到 2024 年的最后几条

# 给每条序列起一个短名字，例如 P5_Staff_Car、P3_MonthlyPass_Moto
veh = {"Light Vehicle": "Car", "Motorcycle": "Moto"}
raw["series"] = raw["car_park"] + "_" + raw["ptype"].str.replace(" ", "") + "_" + raw["vehicle"].map(veh)

# 统一的 15 分钟时间网格：2023-01-01 00:00 到 2023-12-31 23:45
grid = pd.date_range("2023-01-01 00:00", "2023-12-31 23:45", freq="15min")

out, quality = {}, []
for name, g in raw.groupby("series"):
    s = g.set_index("time")["available"].sort_index().astype(float)

    # (1) 孤立尖峰：和前后共 5 个点的中位数差太多 → 设为缺失
    med = s.rolling(5, center=True, min_periods=3).median()
    spike = (s - med).abs() > SPIKE_THRESHOLD
    s[spike] = np.nan

    # (2) 负数没有物理意义 → 截到 0
    n_neg = int((s < 0).sum())
    s = s.clip(lower=0)

    # (3) 重采样：每 15 分钟一格，格内取平均（原理讲义 1.1）
    y = s.resample("15min").mean().reindex(grid)
    n_missing = int(y.isna().sum())

    # (4) 短缺口用“上一个已知值”补（只用过去，不偷看未来）；长缺口保留 NaN
    y = y.ffill(limit=FFILL_LIMIT)

    out[name] = y
    quality.append({"series": name, "raw_records": len(g), "spikes_removed": int(spike.sum()),
                    "negatives_clipped": n_neg, "missing_slots_before_fill": n_missing,
                    "missing_slots_after_fill": int(y.isna().sum()),
                    "min": y.min(), "median": y.median(), "max": y.max()})

wide = pd.DataFrame(out)
wide.index.name = "time"
wide.to_parquet(DATA / "series_15min.parquet")

q = pd.DataFrame(quality).set_index("series")
q.to_csv(RESULTS / "data_quality.csv")
print(q.loc[[TARGET]].T.to_string())

# 主目标的跳变：5 分钟内变化 ≥ 40（相邻两条原始记录间隔 ≤ 10 分钟才算）
t = raw[raw["series"] == TARGET].set_index("time")["available"].sort_index()
gap_min = t.index.to_series().diff().dt.total_seconds() / 60
jumps = pd.DataFrame({"available": t, "change": t.diff(), "gap_min": gap_min})
jumps = jumps[(jumps["change"].abs() >= JUMP_THRESHOLD) & (jumps["gap_min"] <= 10)]
jumps.to_csv(RESULTS / "target_jumps.csv")
print(f"\n{TARGET}: {len(jumps)} 次 5 分钟内跳变 ≥ {JUMP_THRESHOLD}")
print("按小时分布:", jumps.index.hour.value_counts().sort_index().to_dict())
print(f"\n输出：{DATA / 'series_15min.parquet'}  形状 {wide.shape}")

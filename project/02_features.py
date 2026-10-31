"""
02_features.py — 第 2 步：把时间序列变成监督学习的表（原理讲义第 02 章）
输入：data/series_15min.parquet、../um_calendar_2023.csv
输出：data/features_h4.parquet   预测 1 小时后（H = 4）
      data/features_h96.parquet  预测 24 小时后（H = 96）
每一行 = 一个预测起点 t；列 = 特征 + target（y_{t+H}）+ target_time + split
运行：python 02_features.py
"""
from common import *

series = pd.read_parquet(DATA / "series_15min.parquet")
y = series[TARGET]
cal = load_calendar()
OTHER = ["P2_Staff_Car", "P3_Staff_Car", "P6_Staff_Car"]   # 其他停车场的员工车位（反映全校今天人多不多）


def build(H: int) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)                  # 索引 = 预测起点 t

    # ---- (1) 最近的值：都在 t 或 t 之前，一定合法 ----
    X["lag_0"] = y                                   # y_t
    X["lag_1"] = y.shift(1)                          # y_{t-1}（15 分钟前）
    X["lag_2"] = y.shift(2)
    X["lag_4"] = y.shift(4)                          # 1 小时前
    X["diff_1"] = y - y.shift(1)                     # 最近 15 分钟的变化方向
    X["diff_4"] = y - y.shift(4)                     # 最近 1 小时的变化
    X["mean_1h"] = y.rolling(4).mean()               # 最近 1 小时平均（窗口截止在 t）
    X["min_today"] = y.groupby(y.index.date).cummin()  # 今天到目前为止的最低值（只看过去）

    # ---- (2) 季节性滞后：相对“目标时刻”往前 L 步，要求 L > H（L = H 时就是 lag_0）----
    for L, name in [(96, "yday"), (672, "lweek"), (1344, "lweek2")]:
        if L > H:
            X[name] = y.shift(L - H)                 # y_{t+H-L}

    # ---- (3) 其他停车场的当前值 ----
    for c in OTHER:
        X[c + "_now"] = series[c]

    # ---- (4) 日历特征：用目标时刻 t+H 的（日历是提前知道的）----
    tt = X.index + pd.Timedelta(minutes=15 * H)
    slot = tt.hour * 4 + tt.minute // 15
    X["slot"] = slot                                 # 0..95，给树模型用
    X["slot_sin"] = np.sin(2 * np.pi * slot / SLOTS_PER_DAY)
    X["slot_cos"] = np.cos(2 * np.pi * slot / SLOTS_PER_DAY)
    X["dow"] = tt.dayofweek                          # 0 = 周一 … 6 = 周日
    day = cal.reindex(tt.normalize())                # 目标时刻那一天的校历
    X["is_holiday"] = day["is_holiday"].fillna(0).to_numpy().astype(int)
    X["is_off"] = ((tt.dayofweek >= 5) | (X["is_holiday"] == 1)).astype(int)   # 周末或公众假期
    X["period_group"] = day["period_group"].to_numpy()   # teaching / exam / break

    # ---- 标签和元数据 ----
    X["target"] = y.shift(-H)                        # y_{t+H}：唯一允许出现“未来”的列
    X["target_time"] = tt
    X["split"] = split_of(tt)

    # 去掉目标落在 2024 年、或起点太早导致季节性滞后全空的行
    X = X[X["split"] != "none"]
    return X


for H in [4, 96]:
    X = build(H)
    X.to_parquet(DATA / f"features_h{H}.parquet")
    n = X.groupby("split").size().reindex(["train", "val", "cal", "test"])
    print(f"H = {H:>2}: {X.shape[1] - 3} 个特征，行数 {n.to_dict()}，target 缺失 {int(X['target'].isna().sum())}")
print("\n特征列：", [c for c in X.columns if c not in ("target", "target_time", "split")])

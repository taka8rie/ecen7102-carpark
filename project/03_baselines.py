"""
03_baselines.py — 第 3 步：三个基线（原理讲义第 04 章）
  persistence      ŷ_{t+H} = y_t                     （未来 = 现在）
  seasonal_naive   ŷ_{t+H} = y_{t+H-672}             （= 上周同一时刻）
  profile_mean     训练集里“同一日类型 × 同一时刻 × 同一学期阶段”的平均
在验证集（9/16–9/30）和测试集（11–12 月）上算 MAE / RMSE，
分“全天”和“白天 07:00–20:00”两种口径，测试集再按学期阶段拆开。
输出：results/baselines.csv、results/baselines_by_period.csv、
      results/fig_baselines_test_week.png、data/baseline_preds_h{H}.parquet
运行：python 03_baselines.py
"""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["persistence", "seasonal_naive", "profile_mean"]


def daytype(X):
    """日类型：周一到周五用 0–4，周末和公众假期统一记为 'off'。"""
    return np.where(X["is_off"] == 1, "off", X["dow"].astype(str))


def profile_table(train):
    """在训练集上按 (日类型, 时刻, 学期阶段) 求目标的平均值，并准备两级后备表。"""
    t = train.dropna(subset=["target"]).assign(daytype=daytype(train.dropna(subset=["target"])))
    full = t.groupby(["daytype", "slot", "period_group"])["target"].mean()
    fb1 = t.groupby(["daytype", "slot"])["target"].mean()        # 后备 1：不分学期阶段
    fb2 = t.groupby(["slot"])["target"].mean()                   # 后备 2：只看时刻
    return full, fb1, fb2


def predict_profile(X, tables):
    full, fb1, fb2 = tables
    k = pd.DataFrame({"daytype": daytype(X), "slot": X["slot"].to_numpy(), "period_group": X["period_group"].to_numpy()}, index=X.index)
    p = pd.Series(full.reindex(pd.MultiIndex.from_frame(k[["daytype", "slot", "period_group"]])).to_numpy(), index=X.index)
    p = p.fillna(pd.Series(fb1.reindex(pd.MultiIndex.from_frame(k[["daytype", "slot"]])).to_numpy(), index=X.index))
    p = p.fillna(pd.Series(fb2.reindex(k["slot"]).to_numpy(), index=X.index))
    return p


rows, by_period, preds_keep = [], [], {}
for H in [4, 96]:
    X = pd.read_parquet(DATA / f"features_h{H}.parquet")
    tables = profile_table(X[X["split"] == "train"])

    P = pd.DataFrame(index=X.index)
    P["persistence"] = X["lag_0"]
    P["seasonal_naive"] = X["lweek"]
    P["profile_mean"] = predict_profile(X, tables)
    P["target"] = X["target"]
    P["split"] = X["split"]
    P["target_time"] = X["target_time"]
    P["period_group"] = X["period_group"]
    P["is_off"] = X["is_off"]
    P.to_parquet(DATA / f"baseline_preds_h{H}.parquet")
    preds_keep[H] = P

    # 公平比较：所有方法都用同一批行（目标和三个预测都不缺失）
    ok = P[["target"] + METHODS].notna().all(axis=1)
    hour = P["target_time"].dt.hour
    for split in ["val", "test"]:
        for scope, m in [("all_day", ok), ("daytime_07_20", ok & (hour >= 7) & (hour < 20))]:
            d = P[m & (P["split"] == split)]
            for meth in METHODS:
                rows.append({"H": H, "split": split, "scope": scope, "method": meth, "n": len(d),
                             "MAE": mae(d["target"], d[meth]), "RMSE": rmse(d["target"], d[meth])})

    # 测试集按学期阶段 + 是否休息日拆开（白天口径）
    d = P[ok & (P["split"] == "test") & (hour >= 7) & (hour < 20)].copy()
    d["regime"] = np.where(d["is_off"] == 1, "off_day", d["period_group"])
    for reg, g in d.groupby("regime"):
        for meth in METHODS:
            by_period.append({"H": H, "regime": reg, "method": meth, "n": len(g), "MAE": mae(g["target"], g[meth])})

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "baselines.csv", index=False)
bp = pd.DataFrame(by_period)
bp.to_csv(RESULTS / "baselines_by_period.csv", index=False)

pd.set_option("display.width", 140)
print(res.pivot_table(index=["H", "scope", "method"], columns="split", values=["MAE", "RMSE"]).round(2).to_string())
print()
print(bp.pivot_table(index=["H", "method"], columns="regime", values="MAE").round(2).to_string())

# ---- 图：测试期中的一个教学周（英文标签，方便直接放进报告）----
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
week = slice("2023-11-13", "2023-11-19 23:45")
for ax, H in zip(axes, [4, 96]):
    P = preds_keep[H].set_index("target_time").loc[week]
    ax.plot(P.index, P["target"], color="black", lw=1.6, label="actual")
    ax.plot(P.index, P["persistence"], color="#2a78d6", lw=1.1, label="persistence")
    ax.plot(P.index, P["seasonal_naive"], color="#eb6834", lw=1.1, label="seasonal naive (last week)")
    ax.plot(P.index, P["profile_mean"], color="#1baf7a", lw=1.1, label="profile mean (train)")
    ax.set_title(f"P5 staff car spaces, 13–19 Nov 2023 — horizon {H * 15 // 60} h", fontsize=11, loc="left")
    ax.set_ylabel("available spaces")
    ax.grid(alpha=.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=9, loc="upper center", frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(RESULTS / "fig_baselines_test_week.png", dpi=150)
print("\n图已保存：results/fig_baselines_test_week.png")

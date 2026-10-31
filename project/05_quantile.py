"""
05_quantile.py — 第 5 步：分位数 XGBoost（原理讲义第 08、09 章）
一个模型同时输出 q0.1、q0.5、q0.9（objective = reg:quantileerror，即弹球损失），
标签仍用“变化量” Δ = y_{t+H} − y_t，预测时加回 y_t。
[q0.1, q0.9] 是一个“名义 80%”的预测区间，这一步先不做任何校准，看它实际能盖住多少。
输出：results/quantile_selection.csv   候选配置的验证集平均弹球损失
      results/quantile_metrics.csv     验证 / 校准 / 测试集：覆盖率、宽度、弹球损失、区间分数
      results/quantile_by_hour.csv     测试集每个小时的覆盖率和宽度
      results/quantile_by_period.csv   测试集按时段（白天）
      results/fig_quantile_test_week.png
      data/quantile_preds_h{H}.parquet  q10/q50/q90（第 6 步共形校准要用）
运行：python 05_quantile.py
"""
from common import *
from common import interval_metrics as interval_metrics_common
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAUS = np.array([0.1, 0.5, 0.9])
ALPHA = 0.2                      # 名义覆盖率 1 − α = 80%
PERIOD_CODE = {"teaching": 0, "exam": 1, "break": 2}


def design(X):
    drop = {"target", "target_time", "split", "period_group"}
    D = X[[c for c in X.columns if c not in drop]].copy()
    D["period_code"] = X["period_group"].map(PERIOD_CODE).astype(float)
    return D


def interval_metrics(y, lo, hi, q50=None):
    """在 common.interval_metrics 的基础上，再加上中位数的 MAE。"""
    out = interval_metrics_common(y, lo, hi, ALPHA)
    if q50 is not None:
        out["MAE_q50"] = mae(y, q50)
    return out


sel_rows, met_rows, hour_rows, per_rows, keep = [], [], [], [], {}
for H in [4, 96]:
    X = pd.read_parquet(DATA / f"features_h{H}.parquet")
    D = design(X)
    ok = X["target"].notna() & X["lag_0"].notna()
    tr, va = ok & (X["split"] == "train"), ok & (X["split"] == "val")
    delta = X["target"] - X["lag_0"]

    best = None
    for depth in [4, 6, 8]:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=TAUS,      # 一次学三个分位数
            n_estimators=3000, learning_rate=0.03, max_depth=depth, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            early_stopping_rounds=200, random_state=0, n_jobs=4)
        model.fit(D[tr], delta[tr], eval_set=[(D[va], delta[va])], verbose=False)
        Q = model.predict(D) + X["lag_0"].to_numpy()[:, None]       # 加回 y_t，形状 (n, 3)
        raw_cross = float(np.mean(Q[va.to_numpy(), 0] > Q[va.to_numpy(), 2]))
        Q = np.sort(Q, axis=1)                                       # 防止分位数交叉
        yv = X.loc[va, "target"].to_numpy()
        pb = np.mean([pinball(yv, Q[va.to_numpy(), k], t) for k, t in enumerate(TAUS)])
        sel_rows.append({"H": H, "depth": depth, "trees": model.best_iteration + 1,
                         "val_mean_pinball": pb, "val_crossing_rate": raw_cross})
        if best is None or pb < best[0]:
            best = (pb, depth, Q)
    Q = best[2]
    print(f"H={H}: 选 depth={best[1]}（验证集平均弹球损失 {best[0]:.3f}）")

    P = pd.DataFrame({"target": X["target"], "target_time": X["target_time"], "split": X["split"],
                      "q10": Q[:, 0], "q50": Q[:, 1], "q90": Q[:, 2],
                      "period_group": X["period_group"], "is_off": X["is_off"]}, index=X.index)
    P.loc[~X["lag_0"].notna(), ["q10", "q50", "q90"]] = np.nan
    P.to_parquet(DATA / f"quantile_preds_h{H}.parquet")
    keep[H] = P

    ev = P["target"].notna() & P["q10"].notna()
    hour = P["target_time"].dt.hour
    day = (hour >= 7) & (hour < 20)
    for split in ["val", "cal", "test"]:
        for scope, m in [("all_day", ev), ("daytime_07_20", ev & day)]:
            d = P[m & (P["split"] == split)]
            row = {"H": H, "split": split, "scope": scope, "n": len(d),
                   **interval_metrics(d["target"], d["q10"], d["q90"], d["q50"]),
                   "mean_pinball": np.mean([pinball(d["target"], d[c], t) for c, t in zip(["q10", "q50", "q90"], TAUS)])}
            met_rows.append(row)
    d = P[ev & (P["split"] == "test")]
    for h, g in d.groupby(d["target_time"].dt.hour):
        hour_rows.append({"H": H, "hour": h, **interval_metrics(g["target"], g["q10"], g["q90"])})
    d = P[ev & (P["split"] == "test") & day].copy()
    d["regime"] = np.where(d["is_off"] == 1, "off_day", d["period_group"])
    for reg, g in d.groupby("regime"):
        per_rows.append({"H": H, "regime": reg, "n": len(g), **interval_metrics(g["target"], g["q10"], g["q90"])})

pd.DataFrame(sel_rows).to_csv(RESULTS / "quantile_selection.csv", index=False)
met = pd.DataFrame(met_rows); met.to_csv(RESULTS / "quantile_metrics.csv", index=False)
hr = pd.DataFrame(hour_rows); hr.to_csv(RESULTS / "quantile_by_hour.csv", index=False)
pr = pd.DataFrame(per_rows); pr.to_csv(RESULTS / "quantile_by_period.csv", index=False)

pd.set_option("display.width", 170)
print(pd.DataFrame(sel_rows).round(3).to_string(index=False))
print("\n== 未校准的 80% 区间 ==")
print(met.round(3).to_string(index=False))
print("\n== 测试集按时段（白天）==")
print(pr.round(3).to_string(index=False))

# ---- 图：测试周 + 80% 区间 ----
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
week = slice("2023-11-13", "2023-11-19 23:45")
for ax, H in zip(axes, [4, 96]):
    P = keep[H].set_index("target_time").loc[week]
    ax.fill_between(P.index, P["q10"], P["q90"], color="#eb6834", alpha=.25, lw=0, label="80% interval (q0.1–q0.9, uncalibrated)")
    ax.plot(P.index, P["q50"], color="#eb6834", lw=1.2, label="median q0.5")
    ax.plot(P.index, P["target"], color="black", lw=1.4, label="actual")
    c = ((P["target"] >= P["q10"]) & (P["target"] <= P["q90"])).mean()
    ax.set_title(f"P5 staff car spaces, 13–19 Nov 2023 — horizon {H * 15 // 60} h (coverage this week {c:.0%})", fontsize=11, loc="left")
    ax.set_ylabel("available spaces"); ax.grid(alpha=.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, fontsize=9, loc="upper center", frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RESULTS / "fig_quantile_test_week.png", dpi=150)
print("\n图已保存：results/fig_quantile_test_week.png")

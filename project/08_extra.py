"""
08_extra.py — 第 8 步：补充分析（回应组员提问，报告附录 C）
  1. 训练 / 验证 / 测试三段的 MAE（看过拟合程度）
  2. 测试集每个小时的 MAE（误差图）
  3. 按天的块自助法（block bootstrap）：MAE 之差的 95% 置信区间
  4. 数据切分时间轴图
输出：results/extra/ 下的 mae_train_val_test.csv、mae_by_hour.csv、bootstrap_ci.csv、
      fig_mae_by_hour.png、fig_split_timeline.png
运行：python 08_extra.py（需要先跑完 01–04）
"""
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = RESULTS / "extra"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
best = {4: "persistence", 96: "seasonal_naive"}
rows, hour_rows, boot_rows = [], [], []
for H in [4, 96]:
    X = pd.read_parquet(DATA / f"features_h{H}.parquet")
    B = pd.read_parquet(DATA / f"baseline_preds_h{H}.parquet")
    M = pd.read_parquet(DATA / f"model_preds_h{H}.parquet")
    num = [c for c in X.columns if c not in {"target", "target_time", "split", "period_group"}]
    ev = X["target"].notna() & B[["persistence", "seasonal_naive", "profile_mean"]].notna().all(axis=1) & X[num].notna().all(axis=1)
    day = (X["target_time"].dt.hour >= 7) & (X["target_time"].dt.hour < 20)
    P = {"persistence": B["persistence"], "seasonal_naive": B["seasonal_naive"], "ridge": M["ridge"], "xgboost": M["xgboost"]}
    for split in ["train", "val", "test"]:
        m = ev & (X["split"] == split)
        for k, p in P.items():
            rows.append({"H": H, "split": split, "method": k, "n": int(m.sum()),
                         "MAE_day": mae(X.loc[m & day, "target"], p[m & day]), "MAE_all": mae(X.loc[m, "target"], p[m])})
    m = ev & (X["split"] == "test")
    hr = X["target_time"].dt.hour
    for h in range(24):
        mm = m & (hr == h)
        for k in [best[H], "xgboost"]:
            hour_rows.append({"H": H, "hour": h, "method": k, "MAE": mae(X.loc[mm, "target"], P[k][mm])})
    mm = m & day
    T = pd.DataFrame({"y": X.loc[mm, "target"], "d": X.loc[mm, "target_time"].dt.date,
                      **{k: P[k][mm] for k in P}})
    g = T.groupby("d")
    err = {k: g.apply(lambda d: (d["y"] - d[k]).abs().sum()) for k in P}
    cnt = g.size()
    days = cnt.index.to_numpy()
    for a, b in [("xgboost", best[H]), ("xgboost", "ridge")]:
        point = err[a].sum() / cnt.sum() - err[b].sum() / cnt.sum()
        bs = []
        for _ in range(4000):
            idx = rng.integers(0, len(days), len(days))
            bs.append(err[a].iloc[idx].sum() / cnt.iloc[idx].sum() - err[b].iloc[idx].sum() / cnt.iloc[idx].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boot_rows.append({"H": H, "compare": f"{a} - {b}", "diff_MAE_day": point, "ci_lo": lo, "ci_hi": hi, "n_days": len(days)})
R = pd.DataFrame(rows); R.to_csv(OUT / "mae_train_val_test.csv", index=False)
print(R.pivot_table(index=["H", "method"], columns="split", values="MAE_day").round(2).to_string())
print(R.pivot_table(index=["H", "method"], columns="split", values="MAE_all").round(2).to_string())
Bt = pd.DataFrame(boot_rows); Bt.to_csv(OUT / "bootstrap_ci.csv", index=False); print(Bt.round(2).to_string())
Hr = pd.DataFrame(hour_rows); Hr.to_csv(OUT / "mae_by_hour.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
names = {"persistence": "Persistence (best baseline)", "seasonal_naive": "Seasonal naive (best baseline)", "xgboost": "XGBoost"}
for ax, H in zip(axes, [4, 96]):
    d = Hr[Hr.H == H].pivot(index="hour", columns="method", values="MAE")
    w = 0.4
    ax.bar(d.index - w/2, d[best[H]], w, color="#b4b2a9", label=names[best[H]])
    ax.bar(d.index + w/2, d["xgboost"], w, color="#534ab7", label="XGBoost")
    ax.axvspan(6.5, 19.5, color="#f1efe8", zorder=0)
    ax.set_title(f"Test MAE by hour of day, {H*15//60} h ahead", loc="left", fontsize=11)
    ax.set_xlabel("hour of target time (shaded = daytime 07–20)"); ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("MAE (spaces)")
fig.tight_layout(); fig.savefig(OUT / "fig_mae_by_hour.png", dpi=150)

s = pd.read_parquet(DATA / "series_15min.parquet")[TARGET]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(s.index, s.values, lw=0.3, color="#444441")
cols = {"train": "#cecbf6", "val": "#fac775", "cal": "#9fe1cb", "test": "#f5c4b3"}
lab = {"train": "Train  1 Jan–15 Sep", "val": "Validation  16–30 Sep", "cal": "Calibration  Oct", "test": "Test  Nov–Dec (walk-forward)"}
for name, a, b in SPLITS:
    ax.axvspan(pd.Timestamp(a), min(pd.Timestamp(b), s.index[-1]), color=cols[name], alpha=.7, zorder=0, label=lab[name])
ax.set_xlim(s.index[0], s.index[-1]); ax.set_ylabel("available spaces")
ax.set_title("P5 staff car spaces in 2023 and the chronological split (by target time)", loc="left", fontsize=11)
ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(OUT / "fig_split_timeline.png", dpi=150)
print("ok")

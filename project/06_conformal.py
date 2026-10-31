"""
06_conformal.py — 第 6 步：共形校准（原理讲义第 10 章）
用 10 月（校准集，模型训练时没见过）的误差，把区间"量准"。比较五种 80% 区间：
  qr        第 5 步的分位数回归 [q0.1, q0.9]，不校准
  abs       固定宽度：第 4 步 XGBoost 点预测 ± q̂，q̂ = 校准集绝对残差的分位数
  cqr       CQR：[q0.1 − Q̂, q0.9 + Q̂]，Q̂ 由校准集分数 max(q0.1 − y, y − q0.9) 算出
  cqr_hour  分小时的 CQR：24 个小时各算一个 Q̂（分组共形 / Mondrian conformal）
  aci       在 cqr 外面再套自适应共形推断：测试期每走一步，按"之前有没有盖住"调整 α_t
输出：data/intervals_h{H}.parquet          每种方法的上下界（第 7 步用）
      results/conformal_metrics.csv         校准集与测试集的覆盖率、宽度、区间分数
      results/conformal_by_hour.csv、results/conformal_by_period.csv、results/conformal_weekly.csv
运行：python 06_conformal.py
"""
from common import *

ALPHA = 0.2          # 目标覆盖率 80%
GAMMA = 0.005        # ACI 的调整速度
METHODS = ["qr", "abs", "cqr", "cqr_hour", "aci"]


def aci_intervals(cal_scores, base_lo, base_hi, y, H, alpha=ALPHA, gamma=GAMMA):
    """自适应共形推断（Gibbs & Candès 2021）。
    行按目标时刻排好、每 15 分钟一行。第 i 行的结果要等到 H 步之后才知道，
    所以第 i 行用的 α_i 只能用到第 i − H 行及以前的"有没有盖住"。"""
    s = np.sort(cal_scores); n = len(s)
    lo, hi, alphas = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    err = np.zeros(len(y)); a = alpha
    for i in range(len(y)):
        if i - H >= 0:
            a += gamma * (alpha - err[i - H])          # α_{t+1} = α_t + γ(α − err_t)
        a_c = min(max(a, 0.001), 0.999)              # 防止 α ≤ 0（区间无限宽）或 ≥ 1
        k = min(int(np.ceil((n + 1) * (1 - a_c))), n)
        Q = s[k - 1]
        lo[i], hi[i], alphas[i] = base_lo[i] - Q, base_hi[i] + Q, a
        err[i] = float(not (lo[i] <= y[i] <= hi[i]))
    return lo, hi, alphas


rows, hour_rows, per_rows, week_rows = [], [], [], []
for H in [4, 96]:
    M = pd.read_parquet(DATA / f"model_preds_h{H}.parquet")
    Q = pd.read_parquet(DATA / f"quantile_preds_h{H}.parquet")
    P = Q.join(M[["xgboost"]])
    P = P[P["target"].notna() & P["q10"].notna() & P["xgboost"].notna()].sort_values("target_time")
    P["hour"] = P["target_time"].dt.hour
    cal = P[P["split"] == "cal"]

    # (1) 未校准
    P["qr_lo"], P["qr_hi"] = P["q10"], P["q90"]
    # (2) 固定宽度
    qhat = conformal_quantile(np.abs(cal["target"] - cal["xgboost"]), ALPHA)
    P["abs_lo"], P["abs_hi"] = P["xgboost"] - qhat, P["xgboost"] + qhat
    # (3) CQR
    E = np.maximum(cal["q10"] - cal["target"], cal["target"] - cal["q90"])
    Qhat = conformal_quantile(E, ALPHA)
    P["cqr_lo"], P["cqr_hi"] = P["q10"] - Qhat, P["q90"] + Qhat
    # (4) 分小时的 CQR：每个小时只用校准集里同一小时的分数
    Qh = {h: conformal_quantile(E[cal["hour"] == h], ALPHA) for h in range(24)}
    P["Qhat_hour"] = P["hour"].map(Qh)
    P["cqr_hour_lo"], P["cqr_hour_hi"] = P["q10"] - P["Qhat_hour"], P["q90"] + P["Qhat_hour"]
    # (5) ACI：只在测试期滚动
    T = P[P["split"] == "test"]
    assert (T["target_time"].diff().dropna() == pd.Timedelta("15min")).all(), "测试期需要连续的 15 分钟网格"
    lo, hi, alphas = aci_intervals(E.to_numpy(), T["q10"].to_numpy(), T["q90"].to_numpy(), T["target"].to_numpy(), H)
    P["aci_lo"], P["aci_hi"], P["aci_alpha"] = np.nan, np.nan, np.nan
    P.loc[T.index, "aci_lo"], P.loc[T.index, "aci_hi"], P.loc[T.index, "aci_alpha"] = lo, hi, alphas

    print(f"H={H}: 校准集 n={len(cal)}，q̂(固定宽度)={qhat:.2f}，Q̂(CQR)={Qhat:+.2f}，"
          f"分小时 Q̂ 范围 {min(Qh.values()):+.2f} ~ {max(Qh.values()):+.2f}，ACI 期末 α={alphas[-1]:.3f}")
    P.to_parquet(DATA / f"intervals_h{H}.parquet")

    day = (P["hour"] >= 7) & (P["hour"] < 20)
    for split in ["cal", "test"]:
        for scope, m in [("all_day", P["split"] == split), ("daytime_07_20", (P["split"] == split) & day)]:
            d = P[m]
            for meth in METHODS:
                if d[meth + "_lo"].isna().all():
                    continue
                rows.append({"H": H, "split": split, "scope": scope, "method": meth, "n": len(d),
                             **interval_metrics(d["target"], d[meth + "_lo"], d[meth + "_hi"], ALPHA)})
    T = P[P["split"] == "test"].copy()
    for (h, g) in T.groupby("hour"):
        for meth in METHODS:
            hour_rows.append({"H": H, "hour": h, "method": meth, **interval_metrics(g["target"], g[meth + "_lo"], g[meth + "_hi"], ALPHA)})
    Td = T[(T["hour"] >= 7) & (T["hour"] < 20)].copy()
    Td["regime"] = np.where(Td["is_off"] == 1, "off_day", Td["period_group"])
    for reg, g in Td.groupby("regime"):
        for meth in METHODS:
            per_rows.append({"H": H, "regime": reg, "method": meth, "n": len(g), **interval_metrics(g["target"], g[meth + "_lo"], g[meth + "_hi"], ALPHA)})
    Td["week"] = Td["target_time"].dt.to_period("W-SUN").dt.start_time
    for wk, g in Td.groupby("week"):
        for meth in METHODS:
            week_rows.append({"H": H, "week_start": wk.date(), "method": meth, "n": len(g),
                              "coverage": float(np.mean((g["target"] >= g[meth + "_lo"]) & (g["target"] <= g[meth + "_hi"]))),
                              "width": float(np.mean(g[meth + "_hi"] - g[meth + "_lo"]))})

res = pd.DataFrame(rows); res.to_csv(RESULTS / "conformal_metrics.csv", index=False)
pd.DataFrame(hour_rows).to_csv(RESULTS / "conformal_by_hour.csv", index=False)
per = pd.DataFrame(per_rows); per.to_csv(RESULTS / "conformal_by_period.csv", index=False)
pd.DataFrame(week_rows).to_csv(RESULTS / "conformal_weekly.csv", index=False)

pd.set_option("display.width", 170)
print(res.round(3).to_string(index=False))
print("\n== 测试集按时段（白天）覆盖率 ==")
print(per.pivot_table(index=["H", "method"], columns="regime", values="coverage").round(3).to_string())
print("\n== 测试集按时段（白天）宽度 ==")
print(per.pivot_table(index=["H", "method"], columns="regime", values="width").round(1).to_string())

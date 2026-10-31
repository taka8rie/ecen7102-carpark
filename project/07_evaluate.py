"""
07_evaluate.py — 第 7 步：11–12 月模拟的最终评估，并生成报告用的表和图（图用英文）
  1. 总表：点预测（第 4 步）和区间（第 6 步）在测试集上的成绩
  2. 可靠性：名义覆盖率 50%–95% 时，实际覆盖率是多少（越贴近对角线越好）
  3. 给司机看的单边保证："有 90% 把握至少还有 L 个空位"
  4. ACI 的调整速度 γ 敏感性分析（只汇报，不用来选参数）
输出：results/table_point.csv、results/table_interval.csv、results/reliability.csv、
      results/one_sided.csv、results/aci_gamma_sensitivity.csv、results/example_forecasts.csv
      results/fig_final_intervals_week.png、fig_reliability.png、fig_coverage_weekly.png
运行：python 07_evaluate.py（需要先跑完 01–06）
"""
from common import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.2
NAMES = {"qr": "Quantile XGB (uncalibrated)", "abs": "Split conformal, fixed width",
         "cqr": "CQR", "cqr_hour": "CQR by hour (Mondrian)", "aci": "CQR + ACI"}


def aci_intervals(cal_scores, base_lo, base_hi, y, H, alpha, gamma):
    s = np.sort(cal_scores); n = len(s); lo = np.empty(len(y)); hi = np.empty(len(y)); err = np.zeros(len(y)); a = alpha
    for i in range(len(y)):
        if i - H >= 0:
            a += gamma * (alpha - err[i - H])
        k = min(int(np.ceil((n + 1) * (1 - min(max(a, 0.001), 0.999)))), n)
        lo[i], hi[i] = base_lo[i] - s[k - 1], base_hi[i] + s[k - 1]
        err[i] = float(not (lo[i] <= y[i] <= hi[i]))
    return lo, hi


# ---------------- 1. 总表 ----------------
fin = pd.read_csv(RESULTS / "models.csv")
t = fin[fin["split"] == "test"].copy()
best_base = {4: "persistence", 96: "seasonal_naive"}
t["skill_vs_best_baseline_day"] = t.apply(lambda r: 1 - r["MAE_day"] / t[(t.H == r.H) & (t.method == best_base[r.H])]["MAE_day"].iloc[0], axis=1)
t = t[["H", "method", "MAE_day", "RMSE_day", "MAE_all", "RMSE_all", "skill_vs_best_baseline_day"]]
t.to_csv(RESULTS / "table_point.csv", index=False)

conf = pd.read_csv(RESULTS / "conformal_metrics.csv")
ti = conf[conf["split"] == "test"].pivot_table(index=["H", "method"], columns="scope",
                                               values=["coverage", "width", "interval_score"])
ti.columns = [f"{a}_{b}" for a, b in ti.columns]
ti = ti.reset_index()
ti.to_csv(RESULTS / "table_interval.csv", index=False)

# ---------------- 2. 可靠性 + 3. 单边保证 + 4. γ 敏感性 ----------------
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
rel_rows, one_rows, gam_rows, ex_rows, keep = [], [], [], [], {}
for H in [4, 96]:
    P = pd.read_parquet(DATA / f"intervals_h{H}.parquet")
    cal, T = P[P["split"] == "cal"], P[P["split"] == "test"].copy()
    day = (T["hour"] >= 7) & (T["hour"] < 20)
    E = np.maximum(cal["q10"] - cal["target"], cal["target"] - cal["q90"]).to_numpy()
    R = np.abs(cal["target"] - cal["xgboost"]).to_numpy()
    for lev in LEVELS:
        a = 1 - lev
        q = conformal_quantile(R, a)
        Q = conformal_quantile(E, a)
        Qh = T["hour"].map({h: conformal_quantile(E[cal["hour"].to_numpy() == h], a) for h in range(24)})
        ivs = {"abs": (T["xgboost"] - q, T["xgboost"] + q),
               "cqr": (T["q10"] - Q, T["q90"] + Q),
               "cqr_hour": (T["q10"] - Qh, T["q90"] + Qh),
               "aci": aci_intervals(E, T["q10"].to_numpy(), T["q90"].to_numpy(), T["target"].to_numpy(), H, a, 0.005)}
        for meth, (lo, hi) in ivs.items():
            for scope, m in [("all_day", np.ones(len(T), bool)), ("daytime_07_20", day.to_numpy())]:
                y = T["target"].to_numpy()[m]
                rel_rows.append({"H": H, "nominal": lev, "method": meth, "scope": scope,
                                 "coverage": float(np.mean((y >= np.asarray(lo)[m]) & (y <= np.asarray(hi)[m]))),
                                 "width": float(np.mean(np.asarray(hi)[m] - np.asarray(lo)[m]))})

    # 单边下界：分数 s = q0.1 − y（真实值比下界低多少），分小时各取 90% 分位
    s_low = (cal["q10"] - cal["target"]).to_numpy()
    Lh = {h: conformal_quantile(s_low[cal["hour"].to_numpy() == h], 0.10) for h in range(24)}
    T["L90"] = T["q10"] - T["hour"].map(Lh)
    for scope, m in [("all_day", np.ones(len(T), bool)), ("daytime_07_20", day.to_numpy())]:
        one_rows.append({"H": H, "scope": scope, "target": 0.90,
                         "achieved": float(np.mean(T["target"].to_numpy()[m] >= T["L90"].to_numpy()[m])),
                         "mean_gap_to_actual": float(np.mean(T["target"].to_numpy()[m] - T["L90"].to_numpy()[m]))})

    for g in [0.0, 0.001, 0.002, 0.005, 0.01]:
        lo, hi = aci_intervals(E, T["q10"].to_numpy(), T["q90"].to_numpy(), T["target"].to_numpy(), H, ALPHA, g)
        y = T["target"].to_numpy()
        gam_rows.append({"H": H, "gamma": g, **interval_metrics(y, lo, hi, ALPHA),
                         "coverage_day": float(np.mean(((y >= lo) & (y <= hi))[day.to_numpy()]))})

    # 举例：几个具体时刻的预测（给 slides 用）
    for ts in ["2023-11-14 09:00", "2023-11-14 10:00", "2023-11-14 18:00", "2023-12-20 10:00", "2023-12-27 10:00"]:
        r = T.set_index("target_time").loc[pd.Timestamp(ts)]
        ex_rows.append({"H": H, "target_time": ts, "actual": r["target"], "point_xgb": r["xgboost"], "median_q50": r["q50"],
                        "cqr_hour_lo": r["cqr_hour_lo"], "cqr_hour_hi": r["cqr_hour_hi"], "abs_lo": r["abs_lo"], "abs_hi": r["abs_hi"],
                        "at_least_L90": r["L90"]})
    keep[H] = T

rel = pd.DataFrame(rel_rows); rel.to_csv(RESULTS / "reliability.csv", index=False)
one = pd.DataFrame(one_rows); one.to_csv(RESULTS / "one_sided.csv", index=False)
gam = pd.DataFrame(gam_rows); gam.to_csv(RESULTS / "aci_gamma_sensitivity.csv", index=False)
ex = pd.DataFrame(ex_rows); ex.to_csv(RESULTS / "example_forecasts.csv", index=False)

pd.set_option("display.width", 170)
print("== 点预测（测试集）==\n", t.round(3).to_string(index=False))
print("\n== 区间（测试集）==\n", ti.round(3).to_string(index=False))
print("\n== 可靠性（测试集白天）==\n", rel[rel.scope == "daytime_07_20"].pivot_table(index=["H", "method"], columns="nominal", values="coverage").round(3).to_string())
print("\n== 单边 90% 下界 ==\n", one.round(3).to_string(index=False))
print("\n== ACI γ 敏感性 ==\n", gam.round(3).to_string(index=False))
print("\n== 举例 ==\n", ex.round(1).to_string(index=False))

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
week = slice("2023-11-13", "2023-11-19 23:45")
for ax, H, meth in zip(axes, [4, 96], ["cqr_hour", "abs"]):
    P = keep[H].set_index("target_time").loc[week]
    ax.fill_between(P.index, P[meth + "_lo"], P[meth + "_hi"], color="#2a78d6", alpha=.22, lw=0, label="80% interval (" + NAMES[meth] + ")")
    ax.plot(P.index, P["xgboost"] if meth == "abs" else P["q50"], color="#2a78d6", lw=1.1, label="point forecast")
    ax.plot(P.index, P["target"], color="black", lw=1.4, label="actual")
    c = ((P["target"] >= P[meth + "_lo"]) & (P["target"] <= P[meth + "_hi"])).mean()
    ax.set_title(f"Horizon {H * 15 // 60} h — {NAMES[meth]} (coverage this week {c:.0%})", fontsize=11, loc="left")
    ax.set_ylabel("available spaces"); ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="lower left", ncol=3)
fig.tight_layout(); fig.savefig(RESULTS / "fig_final_intervals_week.png", dpi=150)

colors = {"abs": "#9aa0a6", "cqr": "#eb6834", "cqr_hour": "#2a78d6", "aci": "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, H in zip(axes, [4, 96]):
    d = rel[(rel.H == H) & (rel.scope == "daytime_07_20")]
    ax.plot([0.45, 1], [0.45, 1], color="black", lw=.8, ls="--", label="ideal")
    for meth in ["abs", "cqr", "cqr_hour", "aci"]:
        g = d[d.method == meth]
        ax.plot(g["nominal"], g["coverage"], marker="o", ms=4, color=colors[meth], label=NAMES[meth])
    ax.set_title(f"Reliability, daytime 07–20, horizon {H * 15 // 60} h", fontsize=10, loc="left")
    ax.set_xlabel("nominal coverage 1 − α"); ax.set_ylabel("empirical coverage (test)")
    ax.set_xlim(.45, 1); ax.set_ylim(.35, 1); ax.grid(alpha=.3)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "fig_reliability.png", dpi=150)

wk = pd.read_csv(RESULTS / "conformal_weekly.csv", parse_dates=["week_start"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, H in zip(axes, [4, 96]):
    for meth in ["qr", "cqr", "cqr_hour", "aci"]:
        g = wk[(wk.H == H) & (wk.method == meth)]
        ax.plot(g["week_start"], g["coverage"], marker="o", ms=4, color=colors.get(meth, "#c4c7cc") if meth != "qr" else "#b07aa1", label=NAMES[meth])
    ax.axhline(0.8, color="black", lw=.8, ls="--")
    ax.axvspan(pd.Timestamp("2023-12-04"), pd.Timestamp("2023-12-17"), color="#eda100", alpha=.12, label="exam period")
    ax.axvspan(pd.Timestamp("2023-12-18"), pd.Timestamp("2023-12-31"), color="#1baf7a", alpha=.10, label="Christmas recess")
    ax.set_title(f"Weekly daytime coverage, horizon {H * 15 // 60} h", fontsize=10, loc="left")
    ax.set_ylim(.4, 1.02); ax.grid(alpha=.3); ax.tick_params(axis="x", labelrotation=30)
axes[0].set_ylabel("coverage of 80% interval"); axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(RESULTS / "fig_coverage_weekly.png", dpi=150)
print("\n图已保存：fig_final_intervals_week.png、fig_reliability.png、fig_coverage_weekly.png")

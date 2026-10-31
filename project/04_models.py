"""
04_models.py — 第 4 步：Ridge 与 XGBoost 点预测（原理讲义第 05、07 章）
对 H = 4（1 小时）和 H = 96（24 小时）分别：
  1. Ridge：标准化 + 独热编码 + 人工交互特征，λ 在验证集上选
  2. XGBoost：比较“预测绝对值”和“预测变化量 Δ = y_{t+H} − y_t”，树深在验证集上选，早停也用验证集
  3. 选出的配置在测试集上汇报一次（之后不再回头改）
输出：results/model_selection.csv  所有候选配置的验证集成绩
      results/models.csv           基线 + 选中模型，验证集和测试集成绩（同一批行）
      results/models_by_period.csv 测试集按时段拆开
      results/xgb_importance_h{H}.csv、results/fig_importance.png、results/fig_models_test_week.png
      data/model_preds_h{H}.parquet  选中模型在所有行上的预测（第 5、6 步要用）
运行：python 04_models.py
"""
from common import *
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINES = ["persistence", "seasonal_naive", "profile_mean"]
PERIOD_CODE = {"teaching": 0, "exam": 1, "break": 2}


def numeric_cols(X):
    """所有数值特征（去掉标签、元数据和类别列）。"""
    drop = {"target", "target_time", "split", "period_group"}
    return [c for c in X.columns if c not in drop]


def day_mask(P):
    h = P["target_time"].dt.hour
    return (h >= 7) & (h < 20)


def scores(y, yhat, daytime):
    return {"MAE_all": mae(y, yhat), "MAE_day": mae(y[daytime], yhat[daytime]),
            "RMSE_all": rmse(y, yhat), "RMSE_day": rmse(y[daytime], yhat[daytime])}


# ---------------- Ridge ----------------
def ridge_design(X):
    """Ridge 需要的输入：数值特征 + 几个类别列（后面独热编码）。"""
    D = X[numeric_cols(X)].copy()
    hour = X["target_time"].dt.hour
    D["dow_c"] = X["dow"].astype(str)
    D["period_c"] = X["period_group"].astype(str)
    D["hour_c"] = hour.astype(str)
    D["offhour_c"] = X["is_off"].astype(str) + "_" + hour.astype(str)   # 人工交互：休息日 × 小时
    return D


def fit_ridge(D_tr, y_tr, lam):
    cat = ["dow_c", "period_c", "hour_c", "offhour_c"]
    num = [c for c in D_tr.columns if c not in cat]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    model = make_pipeline(pre, Ridge(alpha=lam))      # sklearn 的 alpha 就是 λ
    return model.fit(D_tr, y_tr)


# ---------------- XGBoost ----------------
def xgb_design(X):
    D = X[numeric_cols(X)].copy()
    D["period_code"] = X["period_group"].map(PERIOD_CODE).astype(float)
    return D


def fit_xgb(D_tr, y_tr, D_va, y_va, depth):
    model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=depth, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        early_stopping_rounds=200, eval_metric="mae", random_state=0, n_jobs=4)
    model.fit(D_tr, y_tr, eval_set=[(D_va, y_va)], verbose=False)
    return model


sel_rows, final_rows, period_rows, importance = [], [], [], {}
week_preds = {}
for H in [4, 96]:
    X = pd.read_parquet(DATA / f"features_h{H}.parquet")
    B = pd.read_parquet(DATA / f"baseline_preds_h{H}.parquet")
    Dr, Dx = ridge_design(X), xgb_design(X)

    # 训练行：目标不缺失；Ridge 还要求特征完整（它不接受 NaN）
    has_y = X["target"].notna()
    ridge_ok = Dr[numeric_cols(X)].notna().all(axis=1)
    tr, va = X["split"] == "train", X["split"] == "val"

    # 统一的评价行：目标、三个基线、Ridge 特征都不缺失（和第 3 步一样讲究“同一批行”）
    ev = has_y & B[BASELINES].notna().all(axis=1) & ridge_ok
    dm = day_mask(X)

    def val_score(pred):
        m = ev & va
        return scores(X.loc[m, "target"], pred[m], dm[m])

    preds = {}
    # ---- Ridge：λ 和目标形式都在验证集上选 ----
    best = None
    for mode in ["abs", "delta"]:
        for lam in [0.1, 1, 10, 100, 1000]:
            m = has_y & ridge_ok & tr
            ytr = X.loc[m, "target"] - (X.loc[m, "lag_0"] if mode == "delta" else 0)
            model = fit_ridge(Dr[m], ytr, lam)
            p = pd.Series(np.nan, index=X.index)
            p[ridge_ok] = model.predict(Dr[ridge_ok]) + (X.loc[ridge_ok, "lag_0"] if mode == "delta" else 0)
            s = val_score(p)
            sel_rows.append({"H": H, "model": "ridge", "target_mode": mode, "param": f"lambda={lam}", **s})
            if best is None or s["MAE_day"] < best[0]:
                best = (s["MAE_day"], mode, lam, p)
    preds["ridge"] = best[3]
    ridge_choice = f"{best[1]}, λ={best[2]}"

    # ---- XGBoost：目标形式 × 树深，早停用验证集 ----
    best = None
    for mode in ["abs", "delta"]:
        for depth in [4, 6, 8]:
            ok = has_y & X["lag_0"].notna()          # 变化量模式需要 lag_0；两种模式用同样的训练行，比较才公平
            m_tr, m_va = ok & tr, ok & va
            off_tr = X.loc[m_tr, "lag_0"] if mode == "delta" else 0
            off_va = X.loc[m_va, "lag_0"] if mode == "delta" else 0
            model = fit_xgb(Dx[m_tr], X.loc[m_tr, "target"] - off_tr, Dx[m_va], X.loc[m_va, "target"] - off_va, depth)
            p = pd.Series(model.predict(Dx) + (X["lag_0"].to_numpy() if mode == "delta" else 0), index=X.index)
            s = val_score(p)
            sel_rows.append({"H": H, "model": "xgboost", "target_mode": mode, "param": f"depth={depth}, trees={model.best_iteration + 1}", **s})
            if best is None or s["MAE_day"] < best[0]:
                best = (s["MAE_day"], mode, depth, p, model)
    preds["xgboost"] = best[3]
    xgb_choice = f"{best[1]}, depth={best[2]}, trees={best[4].best_iteration + 1}"
    imp = pd.Series(best[4].get_booster().get_score(importance_type="gain"))
    importance[H] = (imp / imp.sum()).sort_values(ascending=False)
    importance[H].to_csv(RESULTS / f"xgb_importance_h{H}.csv", header=["gain_share"])
    print(f"H={H}: Ridge 选 {ridge_choice}；XGBoost 选 {xgb_choice}")

    # ---- 汇报：基线 + 选中的模型，验证集和测试集，同一批行 ----
    allp = {**{b: B[b] for b in BASELINES}, **preds}
    for split in ["val", "test"]:
        m = ev & (X["split"] == split)
        for name, p in allp.items():
            final_rows.append({"H": H, "split": split, "method": name, "n": int(m.sum()),
                               **scores(X.loc[m, "target"], p[m], dm[m])})
    # 测试集按时段拆开（白天口径）
    m = ev & (X["split"] == "test") & dm
    regime = np.where(X["is_off"] == 1, "off_day", X["period_group"])
    for reg in ["teaching", "exam", "break", "off_day"]:
        mm = m & (regime == reg)
        for name, p in allp.items():
            period_rows.append({"H": H, "regime": reg, "method": name, "n": int(mm.sum()),
                                "MAE": mae(X.loc[mm, "target"], p[mm])})

    out = pd.DataFrame({"target": X["target"], "target_time": X["target_time"], "split": X["split"],
                        "ridge": preds["ridge"], "xgboost": preds["xgboost"]})
    out.to_parquet(DATA / f"model_preds_h{H}.parquet")
    week_preds[H] = out.join(B[["persistence", "seasonal_naive"]])

sel = pd.DataFrame(sel_rows); sel.to_csv(RESULTS / "model_selection.csv", index=False)
fin = pd.DataFrame(final_rows); fin.to_csv(RESULTS / "models.csv", index=False)
per = pd.DataFrame(period_rows); per.to_csv(RESULTS / "models_by_period.csv", index=False)

pd.set_option("display.width", 160)
print("\n== 验证集：所有候选（白天 MAE）==")
print(sel.pivot_table(index=["H", "model", "target_mode"], columns="param", values="MAE_day").round(2).to_string())
print("\n== 基线 + 选中模型 ==")
print(fin.pivot_table(index=["H", "method"], columns="split", values=["MAE_day", "MAE_all", "RMSE_day"]).round(2).to_string())
print("\n== 测试集按时段（白天 MAE）==")
print(per.pivot_table(index=["H", "method"], columns="regime", values="MAE").round(2).to_string())

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, H in zip(axes, [4, 96]):
    top = importance[H].head(10)[::-1]
    ax.barh(top.index, top.values, color="#2a78d6")
    ax.set_title(f"XGBoost feature importance (gain), horizon {H * 15 // 60} h", fontsize=10, loc="left")
    ax.set_xlabel("share of total gain")
fig.tight_layout(); fig.savefig(RESULTS / "fig_importance.png", dpi=150)

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
week = slice("2023-11-13", "2023-11-19 23:45")
for ax, H in zip(axes, [4, 96]):
    P = week_preds[H].set_index("target_time").loc[week]
    ax.plot(P.index, P["target"], color="black", lw=1.6, label="actual")
    ax.plot(P.index, P["persistence" if H == 4 else "seasonal_naive"], color="#9aa0a6", lw=1.1,
            label="best baseline (persistence / seasonal naive)")
    ax.plot(P.index, P["ridge"], color="#1baf7a", lw=1.1, label="Ridge")
    ax.plot(P.index, P["xgboost"], color="#eb6834", lw=1.4, label="XGBoost")
    ax.set_title(f"P5 staff car spaces, 13–19 Nov 2023 — horizon {H * 15 // 60} h", fontsize=11, loc="left")
    ax.set_ylabel("available spaces"); ax.grid(alpha=.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=9, loc="upper center", frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RESULTS / "fig_models_test_week.png", dpi=150)
print("\n图已保存：results/fig_importance.png、results/fig_models_test_week.png")

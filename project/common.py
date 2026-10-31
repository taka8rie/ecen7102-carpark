"""
common.py — 所有步骤共用的路径、常量和小工具。
其他脚本用 `from common import *` 引入。
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent          # .../Claude_work/project
WORK = HERE.parent                              # .../Claude_work
DATA = HERE / "data"                            # 中间数据（脚本生成）
RESULTS = HERE / "results"                      # 结果表格和图
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

RAW_PARQUET = WORK / "cp2023.parquet"           # parse.py 从官方 JSON 解析出来的原始记录
CALENDAR_CSV = WORK / "um_calendar_2023.csv"    # build_calendar.py 生成的校历

# ---------- 常量 ----------
TARGET = "P5_Staff_Car"         # 主目标：P5 停车场员工汽车车位
SLOTS_PER_DAY = 96              # 15 分钟一格，一天 96 格
SLOTS_PER_WEEK = 672

# 按“目标时刻” t+h 划分（左闭右开）。见原理讲义第 03 章。
SPLITS = [
    ("train", "2023-01-01", "2023-09-16"),   # 训练
    ("val",   "2023-09-16", "2023-10-01"),   # 验证：调参、早停、比较模型
    ("cal",   "2023-10-01", "2023-11-01"),   # 校准：只用来算共形预测的 q̂
    ("test",  "2023-11-01", "2024-01-01"),   # 测试：11–12 月模拟，最后才看
]

# 学期阶段归成三组：教学 / 考试 / 假期
PERIOD_GROUP = {"teaching": "teaching", "exam": "exam", "exam_study": "exam",
                "recess": "break", "break": "break", "summer": "break"}


def split_of(target_time: pd.DatetimeIndex) -> np.ndarray:
    """给每个目标时刻贴上 train / val / cal / test 标签。"""
    lab = np.full(len(target_time), "none", dtype=object)
    for name, a, b in SPLITS:
        m = (target_time >= a) & (target_time < b)
        lab[m] = name
    return lab


def load_calendar() -> pd.DataFrame:
    """读校历，按日期索引。增加 period_group 列。"""
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"]).set_index("date")
    cal["period_group"] = cal["period"].map(PERIOD_GROUP)
    return cal


# ---------- 评价指标（原理讲义第 11 章） ----------
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

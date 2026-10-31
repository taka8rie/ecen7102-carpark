# UM 停车场车位预测（含不确定性）· ECEN 7102 课程项目

**Predicting campus car park availability with uncertainty** — University of Macau, ECEN 7102 (Probability & Statistics), Fall 2026 group project.

用 2023 年澳门大学停车场的开放数据，预测 P5 停车场员工汽车车位在未来 1 小时和 24 小时的可用数量，并给出带覆盖率保证的预测区间（分位数回归 + 共形预测）。用 2023 年 1–10 月学习，模拟预测 11–12 月。

## 目录结构

```
.
├── README.md                 本文件
├── cp2023.parquet            原始记录（由 parse.py 从官方 JSON 解析），2023 全年，5 个停车场
├── um_holidays_2023.csv      UM 公众假期
├── um_calendar_2023.csv      校历：每天属于教学 / 考试 / 假期（由 build_calendar.py 生成）
├── parse.py                  官方 JSON → cp2023.parquet
├── build_calendar.py         公众假期 + 校历 → um_calendar_2023.csv
├── qc.py                     原始数据质量检查
├── p5_staff_overview.png     早期探索图
├── 车位预测原理.html          原理讲义（下载后用浏览器打开，可离线阅读）
├── 车位预测项目实录.html       每一步做了什么、结果说明了什么（同上）
└── project/
    ├── README.md             运行说明
    ├── requirements.txt
    ├── common.py             路径、常量、数据切分、评价指标
    ├── 01_prepare.py         第 1 步：清洗 + 重采样到 15 分钟
    ├── 02_features.py        第 2 步：做特征（H = 4 即 1 小时；H = 96 即 24 小时）
    ├── 03_baselines.py       第 3 步：三个基线
    ├── 04_models.py          第 4 步：Ridge 与 XGBoost（预测变化量）
    ├── 05_quantile.py        第 5 步：分位数 XGBoost（未校准的 80% 区间）
    ├── data/                 中间数据（不进仓库，运行脚本自动生成）
    └── results/              结果表格和图
```

## 快速开始

```bash
cd project
pip install -r requirements.txt
python 01_prepare.py
python 02_features.py
python 03_baselines.py
python 04_models.py
python 05_quantile.py
```

如果要从官方原始 JSON 重新生成 `cp2023.parquet`（JSON 有 269 MB，不在仓库里）：

```bash
python parse.py car_park_availability_2023.json cp2023.parquet
python build_calendar.py um_holidays_2023.csv um_calendar_2023.csv
```

## 数据切分（按目标时刻）

| 时间段 | 用途 |
|---|---|
| 2023-01-01 – 09-15 | 训练 train |
| 09-16 – 09-30 | 验证 val：调参、早停、比较模型 |
| 10-01 – 10-31 | 校准 cal：只用来算共形预测的 q̂ |
| 11-01 – 12-31 | 测试 test：11–12 月模拟，只在最后汇报 |

## 进度

- [x] 第 1 步：清洗 + 重采样
- [x] 第 2 步：特征
- [x] 第 3 步：基线（persistence / seasonal naive / profile mean）
- [x] 第 4 步：Ridge、XGBoost 点预测
- [x] 第 5 步：分位数 XGBoost（未校准 80% 区间实际覆盖：1 小时 77%，24 小时 61%）
- [ ] 第 6 步：共形预测（固定宽度、CQR，可选 ACI）
- [ ] 第 7 步：11–12 月模拟评估与分时段分析
- [ ] 报告（11/16）与展示（11/25）

## 目前的结果（白天 07:00–20:00，MAE，单位：车位）

| 步长 | 最好的基线（验证 / 测试） | Ridge（验证 / 测试） | XGBoost（验证 / 测试） |
|---|---|---|---|
| 1 小时 | persistence 13.19 / 11.22 | 7.80 / 6.98 | **6.87 / 6.25** |
| 24 小时 | seasonal naive 13.20 / 21.08 | 15.00 / 16.22 | **13.31 / 16.13** |

所有选择只看验证集（9/16–9/30），测试集（11–12 月）只汇报一次。

## 数据来源

- University of Macau Data and API platform（data.um.edu.mo），Car Park Availability 2023
- UM Academic Calendar 2022/23、2023/24

## 关于生成式 AI 的使用

本仓库的代码、讲解文档和部分分析由 Claude（Anthropic）协助完成，组员负责审阅、理解和对结果负责。按课程要求，最终报告中会说明使用的工具和方式。

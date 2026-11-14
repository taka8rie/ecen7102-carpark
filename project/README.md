# ECEN7102 停车场项目代码（P5 员工车位预测）

## 文件夹结构
```
Claude_work/
├── cp2023.parquet            原始记录（parse.py 从官方 JSON 解析）
├── um_calendar_2023.csv      校历（build_calendar.py 生成）
└── project/
    ├── common.py             路径、常量、数据切分、评价指标
    ├── 01_prepare.py         第 1 步：清洗 + 重采样到 15 分钟
    ├── 02_features.py        第 2 步：做特征（H = 4 和 H = 96）
    ├── 03_baselines.py       第 3 步：三个基线
    ├── 04_models.py          第 4 步：Ridge 与 XGBoost
    ├── 05_quantile.py        第 5 步：分位数 XGBoost（未校准的 80% 区间）
    ├── 06_conformal.py       第 6 步：共形校准（固定宽度、CQR、分小时 CQR、ACI）
    ├── 07_evaluate.py        第 7 步：最终评估、可靠性、单边保证、报告用的表和图
    ├── 08_extra.py           第 8 步：补充分析，输出到 results/extra/
    ├── data/                 中间数据（脚本自动生成）
    └── results/              结果表格和图
```

## 怎么运行
```bash
cd project
pip install -r requirements.txt     # 第一次运行前装一次
python 01_prepare.py
python 02_features.py
python 03_baselines.py
python 04_models.py
python 05_quantile.py
python 06_conformal.py
python 07_evaluate.py
python 08_extra.py
```
每一步都会读上一步的输出，所以要按顺序跑。

## 数据切分（按目标时刻）
| 时间段 | 用途 |
|---|---|
| 2023-01-01 – 09-15 | 训练 train |
| 09-16 – 09-30 | 验证 val（调参、早停、比较模型） |
| 10-01 – 10-31 | 校准 cal（只用来算共形预测的 q̂） |
| 11-01 – 12-31 | 测试 test（11–12 月模拟，最后才看） |

讲解见上一级目录的 `车位预测原理.html`（原理）和 `车位预测项目实录.html`（每一步做了什么）。

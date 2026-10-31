import pandas as pd, sys
hol = set(pd.read_csv(sys.argv[1]).date)
days = pd.DataFrame({'date': pd.date_range('2023-01-01', '2023-12-31')})
# (start, end inclusive, period) from UM almanac 2022/23 and 2023/24
spans = [
 ('2023-01-01','2023-01-04','recess'),        # New Year recess + semester break
 ('2023-01-05','2023-01-14','teaching'),      # Sem 2 classes start 5 Jan
 ('2023-01-15','2023-01-29','recess'),        # Lunar New Year recess
 ('2023-01-30','2023-05-03','teaching'),      # last day of Sem 2 classes 3 May
 ('2023-04-07','2023-04-10','recess'),        # Easter recess
 ('2023-05-04','2023-05-07','exam_study'),
 ('2023-05-08','2023-05-20','exam'),          # incl. contingency day
 ('2023-05-21','2023-06-04','break'),         # after exams, before summer term
 ('2023-06-05','2023-08-16','summer'),        # summer term / summer recess (last day 3 Aug), pre-semester
 ('2023-08-17','2023-11-30','teaching'),      # Sem 1 classes 17 Aug - 30 Nov
 ('2023-12-01','2023-12-03','exam_study'),
 ('2023-12-04','2023-12-16','exam'),
 ('2023-12-17','2023-12-31','recess'),        # Christmas/New Year recess
]
days['period'] = None
for s,e,p in spans:  # later spans override earlier (Easter inside teaching)
    m = (days.date>=s)&(days.date<=e); days.loc[m,'period'] = p
days['weekday'] = days.date.dt.strftime('%a')
days['is_weekend'] = (days.date.dt.dayofweek>=5).astype(int)
days['is_holiday'] = days.date.dt.strftime('%Y-%m-%d').isin(hol).astype(int)
days['date'] = days.date.dt.strftime('%Y-%m-%d')
days.to_csv(sys.argv[2], index=False)
print(days.period.value_counts().to_string()); print('nulls', days.period.isna().sum())

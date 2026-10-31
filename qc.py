import pandas as pd, sys
df=pd.read_parquet(sys.argv[1])
print('range', df.time.min(), '->', df.time.max())
snap=df.drop_duplicates(['time','car_park'])
print('snapshots per car park:\n', snap.groupby('car_park').size())
for cp,g in snap.sort_values('time').groupby('car_park'):
    d=g.time.diff().dt.total_seconds()/60
    print(cp, 'median gap %.1f min'%d.median(), '| gaps>30min:', (d>30).sum(), '| max gap %.0f h'%(d.max()/60))
# biggest gaps for P3
g=snap[snap.car_park=='P3'].sort_values('time'); d=g.time.diff()
big=g.assign(gap=d)[d>pd.Timedelta('2h')].sort_values('gap',ascending=False).head(8)
print('largest P3 gaps:\n', big[['time','gap']].to_string())
s=df.groupby(['car_park','vehicle','ptype']).available.agg(['min','median','max','count'])
print(s.to_string())
# daily coverage
days=snap.assign(d=snap.time.dt.date).groupby('d').size()
print('days with data', len(days), 'days with < 50% expected (6*288):', (days<0.5*6*288).sum())

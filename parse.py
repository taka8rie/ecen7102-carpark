import json, sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
rows=[]; bad=0
with open(src, encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except Exception: bad+=1; continue
        t=r['recordDate']['$date']
        for x in r['records']:
            v=x['noOfAvailableSpace']
            v=int(v['$numberLong']) if isinstance(v,dict) else int(v)
            rows.append((t, r['carParkCode'], x['vehicleType'], x['parkingType'], v))
df=pd.DataFrame(rows, columns=['time_utc','car_park','vehicle','ptype','available'])
df['time']=pd.to_datetime(df.time_utc, utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)
df=df.drop(columns='time_utc')
df.to_parquet(out, index=False)
print('bad lines', bad, 'rows', len(df))

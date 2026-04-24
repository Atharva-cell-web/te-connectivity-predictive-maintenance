import pandas as pd
df = pd.read_parquet(r'd:\te-connectivity-3\new_processed_data\M607_TEST.parquet', engine='pyarrow')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
pivot = df.dropna(subset=['value']).pivot_table(index='timestamp', columns='variable_name', values='value').reset_index()
scrap = pivot[(pivot['timestamp'] >= '2026-03-14 06:40:00+00:00') & (pivot['timestamp'] <= '2026-03-14 07:30:00+00:00')][['timestamp', 'Scrap_counter']]
last_val = None
for i, row in scrap.iterrows():
    if last_val is not None and row['Scrap_counter'] > last_val:
        print(f"{row['timestamp']} : {last_val} -> {row['Scrap_counter']}")
    last_val = row['Scrap_counter']

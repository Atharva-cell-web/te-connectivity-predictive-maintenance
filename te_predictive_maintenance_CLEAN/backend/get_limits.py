import pandas as pd

# Load the February data
data_path = r'D:\te connectivity 3\new_processed_data\FEB_TEST_RESULTS.parquet'
df = pd.read_parquet(data_path)

# Filter for healthy parts
healthy_df = df[df['scrap_probability'] < 0.30]

print("\n--- FEBRUARY DATA DATES ---")
print("Start Date:", df['timestamp'].min())
print("End Date:", df['timestamp'].max())

print("\n--- COPY THIS INTO backend/config_limits.py ---")
print("SAFE_LIMITS = {")
sensors = [
    'Cushion', 'Cycle_time', 'Cyl_tmp_z1', 'Cyl_tmp_z3', 'Cyl_tmp_z4', 
    'Cyl_tmp_z5', 'Cyl_tmp_z8', 'Dosage_time', 'Injection_pressure', 
    'Injection_time', 'Peak_pressure_position', 'Peak_pressure_time', 
    'Switch_position', 'Switch_pressure'
]

for s in sensors:
    if s in healthy_df.columns:
        min_val = round(healthy_df[s].quantile(0.01), 2)
        max_val = round(healthy_df[s].quantile(0.99), 2)
        print(f'    "{s}": {{"min": {min_val}, "max": {max_val}, "unit": ""}},')
print("}")
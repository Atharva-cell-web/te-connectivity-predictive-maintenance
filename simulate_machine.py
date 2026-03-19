import pandas as pd
import requests
import time
import json

# Configuration
CSV_FILE = "new_raw_data/M231-11.csv"
API_ENDPOINT = "http://localhost:8080/api/predict/live"
DELAY_SECONDS = 1  # Simulates the machine cycle time

print(f"🚀 Starting Live Simulator for {CSV_FILE}...")

# Load the raw data
df = pd.read_csv(CSV_FILE)

# Clean it slightly just like the pipeline does
df['timestamp'] = pd.to_datetime(df['timestamp']).astype(str)
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df.dropna(subset=['value'])

print("📡 Connected to machine. Beginning live transmission...")

# Send data row by row to the backend
for index, row in df.iterrows():
    # Package the row as a JSON payload
    payload = {
        "machine_id": "M231",
        "variable_name": row['variable_name'],
        "value": row['value'],
        "timestamp": row['timestamp']
    }
    
    try:
        # Fire the data to your FastAPI backend
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Sent {row['variable_name']} = {row['value']} at {row['timestamp']}")
        else:
            print(f"❌ Error from server: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend server is not running! Please start start_server.py")
        break
        
    # Wait before sending the next reading
    time.sleep(DELAY_SECONDS)
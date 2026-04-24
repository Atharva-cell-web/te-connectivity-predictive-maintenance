import time
import traceback
import sys
sys.path.insert(0, r"d:\te-connectivity-3")

t0 = time.time()
print("Importing backend...")
from backend.data_access import build_control_room_payload
print(f"Import took {time.time()-t0:.1f}s")

t1 = time.time()
print("\nBuilding M607 payload for anchor_time=2026-03-14T07:00:00.000Z ...")
try:
    result = build_control_room_payload(
        machine_id="M-607",
        time_window=60,
        future_window=30,
        anchor_time="2026-03-14T07:00:00.000Z"
    )
    print(f"SUCCESS in {time.time()-t1:.1f}s")
    print(f"Timeline points: {len(result.get('timeline', []))}")
    print(f"Telemetry grid: {len(result.get('telemetry_grid', []))}")
    print(f"Current health: {result.get('current_health', {}).get('status')}")
    print(f"Risk: {result.get('current_health', {}).get('risk_score')}")
    
    # Check sensor values in telemetry
    for sensor in result.get('telemetry_grid', [])[:5]:
        print(f"  {sensor['sensor']}: {sensor['value']} ({sensor['status']})")
except Exception as e:
    print(f"FAILED in {time.time()-t1:.1f}s")
    print(f"Error: {e}")
    traceback.print_exc()

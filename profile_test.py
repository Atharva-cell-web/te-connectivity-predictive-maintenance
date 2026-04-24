import time
import sys

# add backend to path
sys.path.append(r"d:\te-connectivity-3")

from backend.data_access import _build_machine_feb_history
from backend.data_access import build_control_room_payload

def run_test():
    print("Testing build_control_room_payload parts...")
    t0 = time.time()
    build_control_room_payload(
        machine_id="M-356",
        time_window=60,
        future_window=30,
        anchor_time="2026-03-30T18:00:00.000Z"
    )
    t1 = time.time()
    print(f"\nTotal: {t1-t0} sec")

if __name__ == "__main__":
    run_test()

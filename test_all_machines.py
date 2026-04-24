import time
import sys
sys.path.insert(0, r"d:\te-connectivity-3")
from backend.data_access import build_control_room_payload

machines = [
    ("M-356", "2026-04-16T00:00:00Z"),
    ("M-607", "2026-03-14T07:00:00Z"),
    ("M-612", "2026-03-15T00:00:00Z")
]

for m, anchor in machines:
    t1 = time.time()
    try:
        res = build_control_room_payload(m, 60, 30, anchor)
        t2 = time.time()
        print(f"{m} ({anchor}): {len(res.get('timeline', []))} points in {t2-t1:.2f}s")
    except Exception as e:
        print(f"{m} FAILED: {e}")

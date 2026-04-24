import cProfile
import pstats
import sys
sys.path.insert(0, r"d:\te-connectivity-3")

from backend.data_access import build_control_room_payload

def profile_it():
    build_control_room_payload(
        machine_id="M-607",
        time_window=60,
        future_window=30,
        anchor_time="2026-03-14T07:00:00.000Z"
    )

print("Starting profiler...")
profiler = cProfile.Profile()
profiler.enable()
profile_it()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)

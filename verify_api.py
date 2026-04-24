import urllib.request
import json
import time

t0 = time.time()
print('Querying API (this may take a few minutes for cold start)...')
try:
    req = urllib.request.urlopen('http://127.0.0.1:8000/api/control-room/M-356?time_window=60&future_window=30&anchor_time=2026-03-30T18:00:00.000Z', timeout=600)
    data = json.loads(req.read())
    print(f'API returned in {time.time()-t0:.1f}s.')
    print(f'Timeline length: {len(data.get("timeline", []))}') 
    print('First 2 points:', json.dumps(data.get('timeline', [])[:2], indent=2))
    print('Last 2 points:', json.dumps(data.get('timeline', [])[-2:], indent=2))
except Exception as e:
    print(f"Error: {e}")

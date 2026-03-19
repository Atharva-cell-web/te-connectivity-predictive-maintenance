from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from run_realtime_check import run
from data_access import build_control_room_payload, get_recent_window
from forecasting import generate_forecast
from config_limits import SAFE_LIMITS
import pandas as pd
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status/{machine_id}")
def get_machine_status(machine_id: str):
    try:
        decision = run(machine_id)
        return decision
    except Exception as e:
        print(f"Error in Status API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/control-room/{machine_id}")
def get_control_room_data(
    machine_id: str,
    time_window: int = Query(default=240, ge=30, le=1440),
    future_window: int = Query(default=35, ge=5, le=60),
):
    try:
        print(f"[REQUEST] Machine: {machine_id} | Time: {datetime.now().strftime('%H:%M:%S')} | Future: {future_window}m", flush=True)
        result = build_control_room_payload(machine_id=machine_id, time_window=time_window, future_window=future_window)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in Control Room API: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trend/{machine_id}/{parameter}")
def get_trend_data(machine_id: str, parameter: str):
    try:
        history_df = get_recent_window(machine_id, minutes=30)
        
        target_col = parameter
        if parameter not in history_df.columns:
            possible_name = f"{parameter}__last_5m"
            if possible_name in history_df.columns:
                target_col = possible_name
            else:
                print(f"❌ Column not found: {parameter}. Available: {list(history_df.columns)[:5]}...")
                raise HTTPException(status_code=404, detail=f"Parameter '{parameter}' not found in dataset")

        if len(history_df) > 1000:
            history_df = history_df.iloc[::30, :].copy()

        forecast_df = generate_forecast(history_df, target_col)
        
        if 'timestamp' in forecast_df.columns:
            forecast_df['timestamp'] = forecast_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        limits = SAFE_LIMITS.get(parameter, {})
        
        return {
            "data": forecast_df.to_dict(orient="records"),
            "limits": limits
        }
    except Exception as e:
        print(f"Error in Trend API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

from typing import Dict, Any, Optional

@app.post("/api/predict/live")
async def predict_live(data: Dict[str, Any]):
    from backend.live_predictor import predict_from_raw
    machine_id = data.pop("machine_id", "UNKNOWN")
    return predict_from_raw(machine_id, data)

@app.get("/api/predict/buffer-status")
async def buffer_status(machine_id: Optional[str] = None):
    from backend.live_predictor import get_buffer_status
    return get_buffer_status(machine_id)

@app.post("/api/predict/clear-buffer")
async def clear_buf(machine_id: Optional[str] = None):
    from backend.live_predictor import clear_buffer
    clear_buffer(machine_id)
    return {"status": "cleared"}

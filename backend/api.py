from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from run_realtime_check import run
from data_access import build_control_room_payload, get_recent_window
from forecasting import generate_forecast
from config_limits import SAFE_LIMITS
import pandas as pd

app = FastAPI()

# Allow React to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status/{machine_id}")
def get_machine_status(machine_id: str):
    """
    Returns the exact JSON object needed for Zone A & Zone C
    """
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
):
    """
    Returns a unified payload for Control Room sections:
    machine info, summary stats, health state, timeline, and safe limits.
    """
    try:
        return build_control_room_payload(machine_id=machine_id, time_window=time_window)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in Control Room API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trend/{machine_id}/{parameter}")
def get_trend_data(machine_id: str, parameter: str):
    """
    Returns the exact JSON needed for Zone B (History + Prediction + Limits)
    """
    try:
        # 1. Get Real History (Last 30 mins)
        history_df = get_recent_window(machine_id, minutes=30)
        
        # --- FIX: Column Name Mapping ---
        # The frontend asks for "Injection_pressure", but the file might have "Injection_pressure__last_5m"
        target_col = parameter
        if parameter not in history_df.columns:
            possible_name = f"{parameter}__last_5m"
            if possible_name in history_df.columns:
                target_col = possible_name
            else:
                print(f"❌ Column not found: {parameter}. Available: {list(history_df.columns)[:5]}...")
                raise HTTPException(status_code=404, detail=f"Parameter '{parameter}' not found in dataset")

        # --- OPTIMIZATION: Downsample to prevent browser crash ---
        # Take 1 row every 30 rows (approx every 30s)
        if len(history_df) > 1000:
            history_df = history_df.iloc[::30, :].copy()

       # 2. Generate Forecast
        # We pass the CORRECT column name found above
        forecast_df = generate_forecast(history_df, target_col)
        
        # --- FIX: Lock Timestamp format to prevent browser IST conversion ---
        if 'timestamp' in forecast_df.columns:
            # Convert datetime to a strict string format: YYYY-MM-DD HH:MM:SS
            forecast_df['timestamp'] = forecast_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 3. Get Limits
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

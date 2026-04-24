import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import asyncio
import jwt
import os
import json
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, WebSocketException, File, UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.run_realtime_check import run
from backend.data_access import (
    build_control_room_payload, 
    get_recent_window, 
    get_available_machines,
    get_audit_validation_results,
    save_audit_cases,
    normalize_machine_id,
    unified_predict_scrap # Now using V9 directly
)
from backend.ml_inference_v9 import get_production_features
from backend.forecasting import generate_forecast
from backend.config_limits import SAFE_LIMITS
from backend.ingestion_service import (
    handle_upload, 
    get_ingestion_history, 
    run_conversion_pipeline,
    get_pipeline_status,
    should_refresh_latest_pipeline,
)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from fastapi import Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern Lifespan handler replacing deprecated on_event("startup").
    Keeps the dashboard pointed at the newest processed data.
    """
    def _run_sync():
        try:
            should_refresh, reason = should_refresh_latest_pipeline()
            if not should_refresh:
                print(f"[PIPELINE] Startup sync skipped: {reason}", flush=True)
                return

            print(f"[PIPELINE] Startup sync starting in background: {reason}", flush=True)
            run_conversion_pipeline()
            print(f"[PIPELINE] Background startup sync completed.", flush=True)
        except Exception as exc:
            print(f"[PIPELINE] Background startup sync failed: {exc}", flush=True)

    # Offload to background thread so FastAPI can start serving immediately
    import threading
    threading.Thread(target=_run_sync, daemon=True).start()
    print(f"[SERVER] Startup initialization complete. API READY.", flush=True)
    yield
    print(f"[SERVER] Shutting down...", flush=True)

app = FastAPI(lifespan=lifespan)

# Production Fix: High-Visibility Connection Logger
@app.middleware("http")
async def audit_logger(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    print(f"[API] {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration:.3f}s", flush=True)
    return response

# Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "te-connectivity-dev-secret-key-2026-rotate-me")
JWT_ALGORITHM = "HS256"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Allow all origins for local dev (JWT Bearer tokens don't need allow_credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Security Definition
security = HTTPBearer(auto_error=False)


def _decode_jwt_token(token: str):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Production JWT Token Verification Dependency.
    In local dev, we gracefully bypass to preserve UI functionality if no token is provided.
    If a token IS provided, we always verify it.
    """
    LOCAL_DEV = os.environ.get("LOCAL_DEV", "false").lower() == "true"
    
    if credentials is None:
        if LOCAL_DEV:
            return None
        raise HTTPException(status_code=401, detail="Unauthorized: Missing JWT")
    
    try:
        return _decode_jwt_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


def verify_websocket_token(token: Optional[str]):
    LOCAL_DEV = os.environ.get("LOCAL_DEV", "false").lower() == "true"

    if not token:
        if LOCAL_DEV:
            return None
        raise WebSocketException(code=1008, reason="Missing JWT")

    try:
        return _decode_jwt_token(token)
    except Exception as e:
        if LOCAL_DEV:
            print(f"[WS WARN] Invalid token in LOCAL_DEV, allowing connection: {e}", flush=True)
            return None
        raise WebSocketException(code=1008, reason=f"Authentication failed: {str(e)}")

@app.post("/api/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Mock authentication endpoint to issue JWTs for the frontend."""
    # In a real app, verify against a DB. Here we use a safe mock 'admin'/'ai-cup'
    if form_data.username != "admin" or form_data.password != "ai-cup":
        raise HTTPException(status_code=401, detail="Incorrect username or password. Check Configuration Key.")
    
    # Create the token
    payload = {
        "sub": form_data.username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24)
    }
    access_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/status/{machine_id}", dependencies=[Depends(verify_token)])
def get_machine_status(machine_id: str):
    try:
        decision = run(machine_id)
        return decision
    except Exception as e:
        print(f"Error in Status API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/machines", dependencies=[Depends(verify_token)])
def get_machines():
    return get_available_machines()

@app.get("/api/audit/validation", dependencies=[Depends(verify_token)])
def get_audit_validation():
    """
    Returns the cross-referenced results of the model predictions 
    vs. the high-fidelity scrap log transcribe from user.
    """
    return get_audit_validation_results()

@app.post("/api/audit/case", dependencies=[Depends(verify_token)]) # Standardized to singular to match frontend
async def add_audit_case(case: Dict[str, Any]):
    """Add a new ground-truth record."""
    print(f"DEBUG: Received audit case: {case}")
    # CRITICAL FIX: Read the RAW audit_cases.json directly instead of calling
    # get_audit_validation_results(), which can crash if the parquet file is
    # missing expected columns (scrap_5m, etc.), returning an empty list and
    # causing previously saved cases to disappear.
    audit_path = Path(__file__).resolve().parent / "audit_cases.json"
    try:
        current = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    except Exception:
        current = []
    current.append(case)
    if save_audit_cases(current):
        print("DEBUG: Audit case saved successfully")
        return {"status": "success", "message": "Case added successfully"}
    print("DEBUG: Failed to save audit case")
    raise HTTPException(status_code=500, detail="Failed to persist audit case")

@app.put("/api/audit/case/{index}", dependencies=[Depends(verify_token)]) # Standardized to singular to match frontend
async def update_audit_case(index: int, case: Dict[str, Any]):
    """Update an existing ground-truth record."""
    audit_path = Path(__file__).resolve().parent / "audit_cases.json"
    try:
        results = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    except Exception:
        results = []
    if index < 0 or index >= len(results):
        raise HTTPException(status_code=404, detail="Audit case index out of range")
    
    # Standardize the case input to match storage format
    results[index] = case
    if save_audit_cases(results):
        return {"status": "success", "message": "Case updated successfully"}
    raise HTTPException(status_code=500, detail="Failed to persist audit case")

@app.delete("/api/audit/case/{index}", dependencies=[Depends(verify_token)])
async def delete_audit_case(index: int):
    """Remove a ground-truth record."""
    audit_path = Path(__file__).resolve().parent / "audit_cases.json"
    try:
        results = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    except Exception:
        results = []
    if index < 0 or index >= len(results):
        raise HTTPException(status_code=404, detail="Audit case index out of range")
    
    results.pop(index)
    if save_audit_cases(results):
        return {"status": "success", "message": "Case deleted successfully"}
    raise HTTPException(status_code=500, detail="Failed to persist audit case")

@app.get("/api/machines/status", dependencies=[Depends(verify_token)])
def get_all_machine_statuses():
    """Returns a high-speed summary of all machines using the unified V9 inference point."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    machines = get_available_machines()
    results = []

    def _safe_run(machine):
        try:
            # Senior Pro: Use the same unified check as the dashboard to ensure consistency
            decision = run(machine["id"])
            return {
                "id": machine["id"],
                "status": decision.get("alert_level", "UNKNOWN"),
                "risk": decision.get("ml_risk_probability", 0.0)
            }
        except Exception:
            return {"id": machine["id"], "status": "OFFLINE", "risk": 0.0}
            
    with ThreadPoolExecutor(max_workers=max(1, min(len(machines), 5))) as pool:
        futures = {pool.submit(_safe_run, m): m for m in machines}
        for future, machine in futures.items():
            try:
                results.append(future.result(timeout=10))
            except (FuturesTimeout, Exception):
                results.append({"id": machine["id"], "status": "OFFLINE", "risk": 0.0})

    return results

@app.websocket("/ws/machines/status")
async def websocket_machines_status(websocket: WebSocket):
    """Real-time broadcast of fleet machine statuses."""
    await websocket.accept()
    token = websocket.query_params.get("token")
    try:
        verify_websocket_token(token)
    except Exception as e:
        await websocket.close(code=1008)
        return

    print("[WS CONNECTED] Fleet Status Stream", flush=True)
    try:
        while True:
            # Fetch all statuses in parallel
            results = await run_in_threadpool(get_all_machine_statuses)
            await websocket.send_json(results)
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        print("[WS DISCONNECTED] Fleet Status Stream", flush=True)
    except Exception as e:
        import traceback
        print(f"[WS ERROR - FLEET] {e}\n{traceback.format_exc()}", flush=True)
        try:
            await websocket.close()
        except:
            pass

@app.get("/api/control-room/{machine_id}", dependencies=[Depends(verify_token)])
def get_control_room_data(
    machine_id: str,
    time_window: int = Query(default=60, ge=30, le=1440),
    future_window: int = Query(default=30, ge=5, le=60),
    anchor_time: str = Query(default=None)
):
    try:
        # Senior Pro Fix: Normalize machine ID for hyphenated frontend requests (M-231 -> M231)
        machine_norm = normalize_machine_id(machine_id)
        print(f"[REQUEST] Machine: {machine_norm} (Orig: {machine_id}) | Time: {datetime.now().strftime('%H:%M:%S')} | Future: {future_window}m | Anchor: {anchor_time}", flush=True)
        result = build_control_room_payload(machine_id=machine_norm, time_window=time_window, future_window=future_window, anchor_time=anchor_time)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in Control Room API: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trend/{machine_id}/{parameter}", dependencies=[Depends(verify_token)])
def get_trend_data(machine_id: str, parameter: str):
    try:
        history_df = get_recent_window(machine_id, minutes=60)
        
        target_col = parameter
        if parameter not in history_df.columns:
            possible_name = f"{parameter}__last_5m"
            if possible_name in history_df.columns:
                target_col = possible_name
            else:
                print(f"❌ Column not found: {parameter}. Available: {list(history_df.columns)[:5]}...")
                raise HTTPException(status_code=404, detail=f"Parameter '{parameter}' not found in dataset")

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


@app.websocket("/ws/control-room/{machine_id}")
async def websocket_control_room(
    websocket: WebSocket,
    machine_id: str,
    time_window: int = 60,
    future_window: int = 30,
):
    await websocket.accept()
    token = websocket.query_params.get("token")
    try:
        verify_websocket_token(token)
    except Exception as e:
        await websocket.close(code=1008)
        return

    # Production Fix: Normalize machine ID early to prevent NameError in logging
    machine_norm = normalize_machine_id(machine_id)
    anchor_time = websocket.query_params.get("anchor_time")
    
    print(f"[WS CONNECTED] Machine: {machine_norm} | Anchor: {anchor_time}", flush=True)
    try:
        while True:
            # Offload heavy pandas processing to a background thread
            payload = await run_in_threadpool(
                build_control_room_payload,
                machine_norm,
                time_window,
                future_window,
                anchor_time
            )
            await websocket.send_json(payload)
            
            # If we have an anchor time, we are in "Replay Mode"
            # We serve the static slice once and then wait, or serve periodically if windows change
            # But usually for history, live streaming is disabled. 
            # We'll sleep longer or just serve once and keep connection open.
            if anchor_time:
                # In Replay mode, we don't need high frequency updates
                await asyncio.sleep(30) 
            else:
                await asyncio.sleep(5)  # Live mode: 5s streaming
    except WebSocketDisconnect:
        print(f"[WS DISCONNECTED] Machine: {machine_id}", flush=True)
    except Exception as e:
        # Senior Pro Fix: machine_norm is now always defined at start of block
        print(f"[WS ERROR] {machine_id}: {e}", flush=True)
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/predict/live", dependencies=[Depends(verify_token)])
async def predict_live(data: Dict[str, Any]):
    from backend.live_predictor import predict_from_raw
    machine_id = data.pop("machine_id", "UNKNOWN")
    return predict_from_raw(machine_id, data)

@app.get("/api/predict/buffer-status", dependencies=[Depends(verify_token)])
async def buffer_status(machine_id: Optional[str] = None):
    from backend.live_predictor import get_buffer_status
    return get_buffer_status(machine_id)

@app.post("/api/predict/clear-buffer", dependencies=[Depends(verify_token)])
async def clear_buf(machine_id: Optional[str] = None):
    from backend.live_predictor import clear_buffer
    clear_buffer(machine_id)
    return {"status": "cleared"}

# --- INGESTION & PIPELINE ROUTES ---

@app.get("/api/ingest/history", dependencies=[Depends(verify_token)])
async def ingest_history():
    return get_ingestion_history()

@app.post("/api/ingest/upload", dependencies=[Depends(verify_token)])
async def upload_dataset(
    mes_files: List[UploadFile] = File(...),
    hydra_file: UploadFile = File(...)
):
    mes_payloads = []
    for mes_file in mes_files:
        mes_content = await mes_file.read()
        mes_payloads.append((mes_content, mes_file.filename))
    hydra_content = await hydra_file.read()
    
    result = handle_upload(
        mes_payloads,
        hydra_content, hydra_file.filename
    )
    return result

@app.post("/api/ingest/process", dependencies=[Depends(verify_token)])
async def process_pipeline(background_tasks: BackgroundTasks):
    # Trigger the real conversion and merge pipeline in the background
    background_tasks.add_task(run_conversion_pipeline)
    return {"status": "success", "message": "Pipeline processing started in background"}

@app.get("/api/ingest/status", dependencies=[Depends(verify_token)])
async def ingest_status():
    return get_pipeline_status()

# --- MAINTENANCE LOG ROUTES ---

_MAINTENANCE_LOG_PATH = Path(__file__).resolve().parent.parent / "metrics" / "maintenance_log.json"

def _read_maintenance_log() -> list:
    try:
        if _MAINTENANCE_LOG_PATH.exists():
            return json.loads(_MAINTENANCE_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _write_maintenance_log(entries: list):
    _MAINTENANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MAINTENANCE_LOG_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")

@app.post("/api/maintenance/acknowledge", dependencies=[Depends(verify_token)])
async def acknowledge_warning(data: Dict[str, Any]):
    """Mark a sensor warning as acknowledged for a machine."""
    entries = _read_maintenance_log()
    entry = {
        "type": "acknowledge",
        "machine_id": data.get("machine_id", "UNKNOWN"),
        "sensor": data.get("sensor", "UNKNOWN"),
        "status": data.get("status", "UNKNOWN"),
        "value": data.get("value"),
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        "operator": data.get("operator", "dashboard-user"),
    }
    entries.insert(0, entry)
    entries = entries[:500]  # Keep last 500 entries
    _write_maintenance_log(entries)
    print(f"[MAINTENANCE ACK] {entry['machine_id']} / {entry['sensor']} @ {entry['acknowledged_at']}", flush=True)
    return {"status": "success", "entry": entry}

@app.post("/api/maintenance/log", dependencies=[Depends(verify_token)])
async def log_maintenance(data: Dict[str, Any]):
    """Record a maintenance event for a machine sensor."""
    entries = _read_maintenance_log()
    entry = {
        "type": "maintenance",
        "machine_id": data.get("machine_id", "UNKNOWN"),
        "sensor": data.get("sensor", "UNKNOWN"),
        "action": data.get("action", "Manual check"),
        "note": data.get("note", ""),
        "priority": data.get("priority", "medium"),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "operator": data.get("operator", "dashboard-user"),
    }
    entries.insert(0, entry)
    entries = entries[:500]
    _write_maintenance_log(entries)
    print(f"[MAINTENANCE LOG] {entry['machine_id']} / {entry['sensor']} — {entry['action']}", flush=True)
    return {"status": "success", "entry": entry}

@app.get("/api/maintenance/log", dependencies=[Depends(verify_token)])
async def get_maintenance_log(machine_id: Optional[str] = None, limit: int = 50):
    """Retrieve maintenance log, optionally filtered by machine."""
    entries = _read_maintenance_log()
    if machine_id:
        entries = [e for e in entries if e.get("machine_id") == machine_id]
    return entries[:limit]


@app.get("/api/analytics/fleet", dependencies=[Depends(verify_token)])
def get_fleet_analytics():
    """
    Returns a comprehensive fleet-wide analytics summary:
    - Per-machine risk scores, status, alert counts
    - Model accuracy from audit validation
    - Scrap event distribution across machines
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    from backend.data_access import normalize_machine_id

    machines = get_available_machines()

    def _safe_payload(machine):
        mid = machine.get("id", "")
        try:
            m_norm = normalize_machine_id(mid)
            payload = build_control_room_payload(machine_id=m_norm, time_window=240, future_window=30)
            health = payload.get("current_health", {})
            stats = payload.get("summary_stats", {})
            tg = payload.get("telemetry_grid", [])
            warnings = sum(1 for r in tg if r.get("status") in ("WARNING", "HIGH", "CRITICAL"))
            return {
                "id": mid,
                "display_id": payload.get("machine_info", {}).get("display_id", mid),
                "machine_type": payload.get("machine_info", {}).get("machine_type", "UNKNOWN"),
                "status": health.get("status", "UNKNOWN"),
                "risk_score": round(float(health.get("risk_score", 0.0)), 3),
                "active_alerts": warnings,
                "past_scrap_detected": stats.get("past_scrap_detected", 0),
                "future_scrap_predicted": stats.get("future_scrap_predicted", 0),
                "root_causes": health.get("root_causes", [])[:3],
            }
        except Exception as e:
            print(f"[Analytics] Skipped {mid}: {e}")
            return {
                "id": mid,
                "display_id": mid,
                "machine_type": "UNKNOWN",
                "status": "OFFLINE",
                "risk_score": 0.0,
                "active_alerts": 0,
                "past_scrap_detected": 0,
                "future_scrap_predicted": 0,
                "root_causes": [],
            }

    machine_stats = []
    with ThreadPoolExecutor(max_workers=min(len(machines), 4)) as pool:
        futures = {pool.submit(_safe_payload, m): m for m in machines}
        for future in futures:
            try:
                machine_stats.append(future.result(timeout=20))
            except (FuturesTimeout, Exception) as e:
                m = futures[future]
                machine_stats.append({
                    "id": m.get("id", ""), "display_id": m.get("id", ""),
                    "machine_type": "UNKNOWN", "status": "OFFLINE",
                    "risk_score": 0.0, "active_alerts": 0,
                    "past_scrap_detected": 0, "future_scrap_predicted": 0, "root_causes": [],
                })

    # Audit model accuracy
    audit = get_audit_validation_results()
    accuracy = audit.get("accuracy", 0.0)
    total_cases = audit.get("total_cases", 0)
    matched = audit.get("matched_count", 0)

    # Fleet-wide aggregation
    total_alerts = sum(m["active_alerts"] for m in machine_stats)
    total_scrap = sum(m["past_scrap_detected"] for m in machine_stats)
    critical_machines = [m for m in machine_stats if m["status"] in ("CRITICAL", "HIGH")]
    avg_risk = round(sum(m["risk_score"] for m in machine_stats) / max(len(machine_stats), 1), 3)

    return {
        "machines": machine_stats,
        "fleet_summary": {
            "total_machines": len(machine_stats),
            "critical_count": len(critical_machines),
            "average_risk": avg_risk,
            "total_active_alerts": total_alerts,
            "total_scrap_events": total_scrap,
            "model_accuracy": round(accuracy * 100, 1),
            "audit_total_cases": total_cases,
            "audit_matched": matched,
        }
    }


if __name__ == "__main__":
    import uvicorn
    runtime_port = int(os.environ.get("BACKEND_PORT", os.environ.get("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=runtime_port)


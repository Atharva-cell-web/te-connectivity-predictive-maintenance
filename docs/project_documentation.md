# Predictive Maintenance Pipeline Complete Documentation

Welcome to the Predictive Maintenance Predictive System documentation! This document provides an end-to-end overview of how the entire project functions, starting from raw data ingestion and cleaning, through model training, and all the way to real-time dashboard visualizations.

This guide is designed in simple, easy-to-understand language so that any new team member can quickly grasp the architecture, codebase, and data flow.

---

## 1. Step 1: Data Ingestion & Cleaning
**Where it happens in the code**: `backend/ingestion_service.py`

The system starts when raw industrial data is uploaded (sensors and machine execution logs). The data mainly comes in two formats:
- **MES (Sensor Data):** Sent as `.csv` files (e.g., `M356.csv`). These contain continuous reading from machine sensors like temperature, pressure, etc.
- **Hydra Data (Machine Logs):** Sent as `.xlsx` or `.xls` files. These contain machine execution logs, cycle times, and scrap records.
- **Param Data:** Optional parameter thresholds with minimum and maximum values.

### The Cleaning Process
When data is handed to the ingestion service, it goes through several cleaning steps:
1. **Schema Validation & Sniffing:** The script checks every file to understand if it's Sensor Data or Hydra Data. It skips files missing critical timestamps or values.
2. **Timestamp Coercion:** All complex date representations are standardized into Universal Time Coordinated (UTC).
3. **Filtering non-numeric noise:** Many raw industrial files contain status codes (like '0A000'). The pipeline converts values into strict numbers. Any row that cannot be parsed as a number is tagged as `NaN` (Not a Number) and subsequently dropped.
4. **Timestamp Pivoting:** Sensor data is loaded as a "long" list and pivoted into a "wide" structure, so that every timestamp has columns for every unique `variable_name` (e.g. `Cushion`, `Cycle_time`, `Shot_size`).
5. **Merging datasets:** Hydra (Logs) and MES (Sensors) correspond based on their timestamps and machine IDs. 

### What is Dropped?
- Rows containing invalid timestamp structures.
- Rows containing string-type values where numbers are expected. 
- Empty CSV or Excel files. 
- Specific target columns like `future_scrap`, `scrap_weight`, and `scrap_quantity` are dropped when building evaluation data because they directly give away the answer (doing so prevents "Data Leakage" where the model perfectly memorizes the dataset but fails in real life).

**Output Feature Count:** The final processed array creates a massive unified matrix (`FINAL_TRAINING_MASTER_V3.parquet` and Machine Specific `{MACHINE_ID}_TEST.parquet` files). Fully aggregated models train on up to **275 features**.

---

## 2. Step 2: AI Model Training
**Where it happens in the code**: `scripts/train_forecaster_v3.py` & `scripts/train_risk_forecaster.py`

Once we have perfectly aligned and clean historical data, we need to train AI to predict future machine failures and future sensor trends. The system uses two types of powerful Artificial Intelligence algorithms combined together.

### Model 1: The Sensor Forecaster (`train_forecaster_v3.py`)
- **What it does:** Predicts what the sensors will look like in the immediate future.
- **How many models:** 1 massive `LightGBM Regressor` model configured to output multi-target arrays (MultiOutputRegressor).
- **Features Used:** Operates on 24 targeted core sensors (e.g., `Cycle_time`, `Cyl_tmp_z1..z8`, `Injection_pressure`). 
- **Lags Mechanism:** Uses 30 steps of historical "look-back lag" to predict the next 10 steps into the future. 

### Model 2: The Stacked Risk Forecaster (`train_risk_forecaster.py`)
- **What it does:** Predicts the actual probability of generating scrap items from 5 minutes up to 30 minutes in the future.
- **How many models:** 6 separate regressors, individually trained for 5m, 10m, 15m, 20m, 25m, and 30m horizons.
- **How training is done:** 
  1. It uses a strong Base Production model to score all training rows with a base "Risk Score".
  2. It calculates how this risk score has behaved dynamically (Calculating Risk Means, Standard Deviations, Maximums in a 15-minute window, and Scrap Velocity).
  3. Uses **20 precise velocity/lag features** to map these dynamic historical curves to future scrap generation. By making it a regression problem (predicting a 0-1 continuous score) instead of a binary classification (0 or 1), the AI avoids issues with rare class imbalances (where scrap is naturally rare in manufacturing).

**Output:** Stores heavily optimized pickel (`.pkl`) artifacts in `models/` directory.

---

## 3. Step 3: Real-Time Flow & The Backend API
**Where it happens in the code:** `backend/api.py` & `backend/ml_inference_v9.py`

When the system runs, it does not retrain models. It uses the pre-trained `.pkl` files and streams the Parquet datasets as if they were live telemetry over the network!

### Data Flow Execution:
1. **API Initialization:** The FastAPI server spins up. It detects the latest `{MACHINE_ID}_TEST.parquet` files which act as the database buffer.
2. **Incoming User Requests:** The user opens the Dashboard, which establishes a **WebSocket**. WebSockets act as an open pipeline (instead of regular HTTP where connections open and close).
3. **ML Inference execution:** For every machine currently being monitored (e.g. `ws/control-room/M356`):
   - The backend runs `build_control_room_payload()`. 
   - It scoops up a rolling 60-minute window of recent reading for the given machine. 
   - Uses the MultiOutputRegressor to forecast the future 30 minutes of sensor behavior.
   - Pushes the 60-minute historical + 30-minute projected telemetry to the `RiskForecasterModel`.
   - Generates the High / Medium / Low Alert warnings based on dynamic threshold limits.
4. **Data Dispatching:** The Backend packages all risks, limit violations, trends, and future trajectories into a clean JSON array and broadcasts it down the WebSocket to the frontend every 5 seconds.

---

## 4. Step 4: The Dashboard (Frontend)
**Where it happens in the code:** `frontend/src/`

This is what the final user interact with. 
- **React/Vite Infrastructure:** The frontend listens specifically to the APIs and WebSockets.
- **Data Rendering:** 
   - When a WebSocket JSON payload is received, the frontend plots it onto smooth dynamic charts.
   - **Historical line** is painted solid indicating what *did* happen.
   - **Future line** is painted dashed/orange showing what the *AI Model* expects to happen.
- **Alert Triage:** Machine tiles change color automatically (Green -> Yellow -> Red) when the Backend sends a "CRITICAL" status, preventing bad quality outputs by visually alerting operators immediately before it happens.
- **Audit Logging:** Allows users to input "ground-truth exceptions" via `/api/audit/case` dynamically.

---

## The Complete Sequence Map
For a quick mental recap, if a new packet of Sensor data comes into our system, here is how it walks:

1. `M-XY.csv` uploaded in the browser ->  
2. `backend.api.handle_upload()` saves it locally ->  
3. `backend.ingestion_service.run_conversion_pipeline()` cleans strings into numbers, aligns time structures, and stores it in high-speed `.parquet` formats ->  
4. The Backend reads the `.parquet` file in sliding intervals imitating live time ->  
5. **AI Inference (`ml_inference_v9.py`)** swallows the recent window, calculates mathematical Risk Lags/Velocity, queries its loaded models, and outputs a 1-to-30 minute prophecy line ->  
6. **FastAPI WebSockets (`api.py`)** packages this array and shoots it precisely to the correct connected machine tab ->  
7.  **Frontend Charting System** catches the array and visualizes the future line. Operator avoids a scrap event!

End of flow.

# HOW WE BUILT THIS

## 1. The Problem & Our Approach

We built this project to monitor Injection Molding machine behavior and predict scrap (defective parts) before defects happen.  
Our primary focus machine is **M-231**.

Instead of reacting after scrap is produced, we designed a predictive system that continuously reads process signals, estimates risk, and flags unsafe trends early enough for operators to intervene.

## 2. The Data Flow (Start to End)

### Data Scope

We trained and validated this workflow using factory sensor data from **January 12, 2026** through **February 17, 2026**.

### Data Preprocessing

We prepared the raw machine logs in a strict pipeline:

1. We cleaned noisy/incomplete records.
2. We standardized timestamp handling in **UTC** to avoid browser-side timezone drift (especially IST conversion issues).
3. We merged process data with quality outcome labels so each record is tied to a target outcome (`scrap` vs `yield`).

This gave us a reliable timeline of machine behavior with trustworthy labels.

### Feature Engineering with Rolling Windows

We did not model isolated sensor points.  
We engineered **rolling-window features** to capture machine health trends over time.

Examples include:

- 5-minute rolling mean
- 15-minute rolling mean
- rolling max/min and short-term trend behavior

This matters because scrap risk usually builds up gradually. Rolling features let the model "see" momentum and instability before failure.

## 3. The AI Model (How It Works)

We selected **LightGBM** because it is fast, robust, and well-suited for high-dimensional tabular sensor data.

Key model decisions:

- We treated this as a binary classification problem (scrap risk vs normal operation).
- We handled class imbalance carefully because true scrap events are rare.
- We tuned weighting so the model does not ignore minority scrap cases.

On unseen February data, this approach delivered **87% precision**, meaning when we flag high risk, we are usually right.

## 4. Backend Architecture (Where Data Comes From Now)

Our backend is built with **FastAPI** in `backend/api.py`, which serves dashboard-ready endpoints for status and trends.

For this lightweight GitHub delivery, `backend/run_realtime_check.py` reads pre-calculated, real AI prediction outputs from `FEB_TEST_RESULTS.parquet`.  
This keeps setup fast and practical on any local PC, without requiring full training workflows or heavy runtime inference setup.

We also use `backend/config_limits.py` to define dashboard operating bounds.  
These limits are not guessed manually. We derived them from historically healthy production behavior using dynamic percentile boundaries:

- lower safe boundary: **1st percentile**
- upper safe boundary: **99th percentile**

This makes alert thresholds data-driven and consistent with real factory behavior.

## 5. How to Read the Dashboard

### Overall Machine Status

This is the top-level AI signal for current risk:

- **Optimal**: stable operation
- **AI Warning**: elevated risk trend
- **AI Risk Prediction**: high confidence risk pattern

It is tied directly to the model's current **ML Risk Probability**.

### Telemetry Gauges

These show live process values such as:

- Injection Pressure
- key cylinder temperature zones
- other critical molding parameters

Operators can quickly compare current values against expected safe behavior.

### System Health Monitor (Chart)

This chart displays historical machine behavior on exact factory timestamps, locked to the canonical UTC timeline.  
Safe operating boundaries are shown so drift, excursions, or unstable trends are easy to spot before quality loss occurs.

## Final Summary

We built an end-to-end predictive maintenance pipeline that converts raw molding telemetry into actionable risk intelligence.  
The result is a practical system that is accurate, explainable, and deployable on standard local machines for fast team adoption.

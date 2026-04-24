# 🏭 TE Connectivity: AI-Powered Predictive Maintenance
### *The "V5" Fleet Monitoring Solution*

[![Version](https://img.shields.io/badge/Release-v5.1.0--V5--Full-7916ff?style=for-the-badge&logo=rocket)](https://github.com/Vishh70/te-connectivity-3)
[![Status](https://img.shields.io/badge/Status-Deployment--Ready-2ea44f?style=for-the-badge&logo=checkmarx)](https://github.com/Vishh70/te-connectivity-3)
[![FastAPI](https://img.shields.io/badge/FastAPI-005863?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Collaboration](https://img.shields.io/badge/Team-Atharva%20&%20Vishnu-orange?style=for-the-badge&logo=github)](./CONTRIBUTING.md)

The TE Connectivity Predictive Maintenance system is a state-of-the-art solution designed to monitor, predict, and explain scrap risks across mechanical production fleets. By leveraging the **V9 Universal Oracle** and the new **Time Machine Replay Engine**, we provide operators with high-precision forecasting and the ability to audit historical scrap events with surgical precision.

### 🌟 New in V5.5: "The Time Machine"
- **Historical Replay Mode**: Anchor the dashboard to any past timestamp to verify AI predictions against actual scrap outcomes.
- **Unified V9 Oracle**: One high-fidelity model serving the entire fleet with 75% precision alignment.
- **Seamless WebSockets**: Stabilized real-time streaming with intelligent HTTP fallback.

---

## 🏛️ Advanced Architecture

Our architecture is optimized for high-performance telemetry processing and explainable AI insights.

```mermaid
graph LR
    classDef machine fill:#ff99cc,stroke:#333,stroke-width:2px,color:#000;
    classDef backend fill:#99ccff,stroke:#333,stroke-width:2px,color:#000;
    classDef frontend fill:#99ffcc,stroke:#333,stroke-width:2px,color:#000;
    classDef data fill:#ffffff,stroke:#333,stroke-width:2px,color:#000,stroke-dasharray: 5 5;

    subgraph "Production Fleet"
        M1["🏭 Machine 231"]:::machine
        M2["🏭 Machine 607"]:::machine
        Mn["🏭 Machine N"]:::machine
    end

    subgraph "Data Storage"
        PQT[(".parquet Telemetry")]:::data
        PKL[(".pkl Model Weights")]:::data
        JSON[("Config & Limits")]:::data
    end

    subgraph "Production Backend"
        DA["💾 Data Access Layer"]:::backend
        NORM["⚖️ Z-Score Normalization"]:::backend
        INF["🧠 V5 Inference Engine"]:::backend
        SHAP["🔍 SHAP Processor"]:::backend
        API["📡 REST / WebSocket API"]:::backend
    end

    subgraph "Digital Twin Dashboard"
        DR["🕹️ Control Room"]:::frontend
        AH["📊 Analytics"]:::frontend
        XAI["🔍 Explainability Tool"]:::frontend
    end

    M1 & M2 & Mn --> DA
    PQT --> DA
    DA --> NORM
    NORM --> INF
    INF --> SHAP
    SHAP --> API
    PKL & JSON -.-> INF
    API --> DR
    API --> AH
    API --> XAI
```

---

## 📡 API Specification (V5)

| Endpoint | Method | Description | Payload / Query |
| :--- | :--- | :--- | :--- |
| `/api/login` | `POST` | Authenticates user and issues JWT. | `{username, password}` |
| `/api/machines` | `GET` | Lists all available 200+ fleet assets. | - |
| `/api/status/{id}`| `GET` | Fetches real-time alert level and probability. | `machine_id` |
| `/api/control-room/{id}`| `GET/WS` | High-fidelity digital twin payload. | `time_window`, `future_window` |
| `/api/trend/{id}/{p}`| `GET` | Historic sensor trend + safety limits. | `machine_id`, `parameter` |
| `/api/audit/validation`| `GET` | Ground-truth verification results. | - |
| `/api/analytics/fleet`| `GET` | Aggregated fleet performance metrics. | - |
| `/ws/control-room/{id}`| `WS` | Stable real-time telemetry stream. | `anchor_time` (optional) |

---

## 🤝 Collaborative Workflow (Atharva & Vishnu)

> [!TIP]
> This project is designed for high-end collaboration. 
> See the [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide for details on:
> *   Branching strategy and merge protocols.
> *   Synchronizing local environments (Venv, Npm).
> *   Manual data handoff (Mergedata control).

---

## 🐋 Production Deployment (Docker)

> [!TIP]
> This project is containerized for professional, one-click deployment.

### 🚀 Quickstart: Run Development Environment
To launch the entire suite (Backend + Frontend) simultaneously, open PowerShell in the project root and run:
```powershell
.\run-dev.ps1
```
This script will start the FastAPI backend on port 8000 and the Vite React frontend concurrently.

---

## 📂 Project Structure

- `backend/` - Core API, Normalization Engine, and Inference.
- `frontend/` - React/Vite Source, Fleet Analytics Dashboard, and XAI Tools.
- `models/` - Production Weights for V9 model.
- `metrics/` - Fleet-wide thresholds and normalization calibration.

---
**Status:** ✅ 100% Finalized & Perfected | **Designed For:** TE Connectivity AI Cup 🏆

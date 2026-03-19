# TE Connectivity Predictive Monitoring (Inference-Ready)

This repository is prepared so anyone can run the model locally without retraining.

## What is included for direct run

- Pre-trained model: `models/lightgbm_scrap_risk_wide.pkl`
- Inference feature map: `processed/features/rolling_feature_columns.txt`
- Demo feature data: `processed/features/rolling_features_demo.parquet`
- Backend API: `backend/api.py`
- Frontend dashboard: `frontend/`

Training scripts are kept in `scripts/` but are not required to run the app.

## Local setup (for your friends)

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ and npm

### 2. Clone repository

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd "<YOUR_REPO_FOLDER>"
```

### 3. Run backend (Terminal 1)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
uvicorn api:app --host 0.0.0.0 --port 8080
```

Backend URL: `http://127.0.0.1:8080`

### 4. Run frontend (Terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in terminal (usually `http://127.0.0.1:5173`).

## Quick API check

```powershell
curl "http://127.0.0.1:8080/api/status/M-231"
curl "http://127.0.0.1:8080/api/trend/M-231/Injection_pressure"
```

## Push this project to GitHub

Run these commands from project root (`d:\te connectivity 3`):

```powershell
git init
git add .
git commit -m "Inference-ready dashboard: model + demo data + run docs"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Because `.gitignore` is configured, large training/raw data and `node_modules` are excluded.

## Automatic project sync

If you want one command to stage changes, create a timestamped commit, pull, and push:

```powershell
.\sync_project.cmd
```

If there are no new file changes, the same command still performs sync-only behavior and will pull or push any pending branch commits so your local checkout and GitHub stay aligned.

By default, this command uses one shared GitHub remote:

- `origin` -> `Atharva-cell-web/te-connectivity-predictive-maintenance`

If Windows cached the wrong GitHub account and push fails with `permission denied`, reset the repo auth first:

```powershell
.\scripts\fix_github_push_auth.ps1
```

During the next push, sign in with any GitHub account that has collaborator access to the shared repo.

This repo also has local Git aliases configured:

```powershell
git autosync
git autosync-dry
git autosync-staged
```

You can also preview what will be committed without changing Git state:

```powershell
.\sync_project.cmd -DryRun
```

If you want a custom commit message:

```powershell
.\sync_project.cmd -Message "cleanup old scrap_prediction_v1 files"
```

If you only want to commit the files that are already staged in Source Control:

```powershell
.\sync_project.cmd -UseCurrentStaging
```

The same command is available through npm:

```powershell
npm run sync:auto
npm run sync:auto:dry
npm run sync:auto:auth
```

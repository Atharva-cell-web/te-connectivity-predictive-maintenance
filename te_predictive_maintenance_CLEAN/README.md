# TE Connectivity Predictive Maintenance

## Setup

### Backend
pip install fastapi uvicorn pandas numpy lightgbm joblib pyarrow
cd backend
python -m uvicorn api:app --host 0.0.0.0 --port 8080

### Frontend  
cd frontend
npm install
npm run dev

### Open Dashboard
http://localhost:5173

## Model Info
- AUC: 97.4%
- Recall: 98.1%
- Features: 378 (see models/feature_list.txt)

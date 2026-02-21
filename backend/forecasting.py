import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def generate_forecast(history_df, parameter_name, future_minutes=60):
    """
    Takes REAL historical data and projects it forward using Linear Regression.
    Returns: DataFrame with 'timestamp', 'value', 'type' (history/prediction)
    """
    # 1. Prepare Data
    df = history_df[["event_timestamp", parameter_name]].copy()
    df.columns = ["timestamp", "value"]
    df["type"] = "history"
    
    # Convert timestamps to numeric for regression (seconds since start)
    df["time_sec"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
    
    # 2. Fit Linear Regression on the last 15 minutes only (Current Trend)
    # We don't want the trend from an hour ago, we want the *current* momentum.
    latest_time = df["time_sec"].max()
    recent_mask = df["time_sec"] >= (latest_time - 15*60)
    
    X = df.loc[recent_mask, "time_sec"].values.reshape(-1, 1)
    y = df.loc[recent_mask, "value"].values
    
    if len(X) < 10: # Not enough data to forecast
        return df
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Generate Future Points (Next 60 mins)
    future_steps = np.linspace(latest_time, latest_time + (future_minutes * 60), num=10)
    future_vals = model.predict(future_steps.reshape(-1, 1))
    
    # 4. Create Future DataFrame
    last_timestamp = df["timestamp"].max()
    future_timestamps = [last_timestamp + pd.Timedelta(seconds=int(t - latest_time)) for t in future_steps]
    
    future_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "value": future_vals,
        "type": "prediction" # This flag tells React to make the line DOTTED
    })
    
    # Combine
    full_df = pd.concat([df.drop(columns=["time_sec"]), future_df])
    return full_df
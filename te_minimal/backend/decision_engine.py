from safety_rules import check_safety_limits
from ml_inference import predict_risk

def make_decision(snapshot, model, X):
    ml_risk = predict_risk(model, X)
    violations = check_safety_limits(snapshot)

    if violations:
        alert = "CRITICAL"
        reason = "SAFE_LIMIT_VIOLATION"
    elif ml_risk >= 0.7:
        alert = "HIGH"
        reason = "ML_RISK"
    elif ml_risk >= 0.4:
        alert = "MEDIUM"
        reason = "ML_RISK"
    else:
        alert = "LOW"
        reason = "NORMAL"

    return {
        "machine_id": snapshot["machine_id_normalized"],
        "timestamp": snapshot["event_timestamp"],
        "ml_risk_probability": round(ml_risk, 3),
        "alert_level": alert,
        "decision_reason": reason,
        "violations": violations
    }

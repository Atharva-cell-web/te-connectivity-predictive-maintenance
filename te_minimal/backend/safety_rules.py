from config_limits import SAFE_LIMITS

def check_safety_limits(snapshot):
    """
    Checks hard safe operating limits.
    Returns list of violations with exact values and units.
    """

    violations = []

    for param, limits in SAFE_LIMITS.items():
        if param not in snapshot:
            continue  # parameter not present in this machine

        value = snapshot[param]

        if value is None:
            continue

        # Upper limit violation
        if limits.get("max") is not None and value > limits["max"]:
            violations.append({
                "parameter": param,
                "current_value": float(value),
                "safe_limit": f"≤ {limits['max']}",
                "deviation": round(value - limits["max"], 3),
                "unit": limits["unit"],
                "direction": "ABOVE_LIMIT"
            })

        # Lower limit violation
        if limits.get("min") is not None and value < limits["min"]:
            violations.append({
                "parameter": param,
                "current_value": float(value),
                "safe_limit": f"≥ {limits['min']}",
                "deviation": round(limits["min"] - value, 3),
                "unit": limits["unit"],
                "direction": "BELOW_LIMIT"
            })

    return violations

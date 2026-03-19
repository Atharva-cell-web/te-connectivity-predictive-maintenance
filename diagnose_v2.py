from backend.data_access import _load_machine_pivot, _load_control_model_and_features
import numpy as np

machine_id = "M471"
print(f"--- Diagnosing Data for {machine_id} ---")
pivot, _ = _load_machine_pivot(machine_id)
model, model_features = _load_control_model_and_features()

missing_count = 0
for feature in model_features:
    if feature not in pivot.columns:
        print(f"MISSING (Defaulting to 0.0): {feature}")
        missing_count += 1

print(f"\nTOTAL MISSING FEATURES: {missing_count} out of {len(model_features)}")
if missing_count > 50:
    print("ROOT CAUSE FOUND: The rolling features are not being calculated because of column name mismatches!")

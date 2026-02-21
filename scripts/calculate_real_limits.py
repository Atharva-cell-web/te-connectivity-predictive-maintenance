# backend/config_limits.py

# ML thresholds (business-owned)
ML_THRESHOLDS = {
    "LOW": 0.30,
    "MEDIUM": 0.60
}

# Hard safety limits (Derived from Data + Engineering Judgment)
SAFE_LIMITS = {
    "Cushion": {
        "min": 1.68,
        "max": 10.72,
        "unit": "mm"
    },
    "Cycle_time": {
        "min": 0,
        "max": 26.12,
        "unit": "s"
    },
    "Cyl_tmp_z1": {
        "min": 190.0,   # Tightened from 203 for startup allowance
        "max": 350.0,   # Manually capped from 734 (sensor glitch)
        "unit": "°C"
    },
    "Cyl_tmp_z3": {
        "min": 196.44,
        "max": 350.0,   # Capped for realism
        "unit": "°C"
    },
    "Cyl_tmp_z4": {
        "min": 192.24,
        "max": 350.0,
        "unit": "°C"
    },
    "Cyl_tmp_z5": {
        "min": 188.02,
        "max": 350.0,
        "unit": "°C"
    },
    "Cyl_tmp_z8": {
        "min": 28.96,
        "max": 236.59,
        "unit": "°C"
    },
    "Dosage_time": {
        "min": 0,
        "max": 4.92,
        "unit": "s"
    },
    "Injection_pressure": {
        "min": 87.54,
        "max": 1991.97,
        "unit": "bar"
    },
    "Injection_time": {
        "min": 0.02,
        "max": 0.86,
        "unit": "s"
    },
    "Peak_pressure_position": {
        "min": 3.61,
        "max": 12.08,
        "unit": "bar" # Data labeled this 'bar', confirming if position (mm) or pressure
    },
    "Peak_pressure_time": {
        "min": 0.02,
        "max": 0.86,
        "unit": "s"   # Fixed unit from 'bar' to 's'
    },
    "Switch_position": {
        "min": 4.49,
        "max": 11.34,
        "unit": "mm"
    },
    "Switch_pressure": {
        "min": 88.59,
        "max": 1975.61,
        "unit": "bar"
    }
}

# Human-Readable Labels for Dashboard
PARAMETER_LABELS = {
    "Cushion": "Cushion Position",
    "Cycle_time": "Cycle Time",
    "Cyl_tmp_z1": "Cylinder Temp Zone 1",
    "Cyl_tmp_z3": "Cylinder Temp Zone 3",
    "Cyl_tmp_z4": "Cylinder Temp Zone 4",
    "Cyl_tmp_z5": "Cylinder Temp Zone 5",
    "Cyl_tmp_z8": "Cylinder Temp Zone 8",
    "Dosage_time": "Dosage Time",
    "Ejector_fix_deviation_torque": "Ejector Torque Deviation",
    "Extruder_start_position": "Extruder Start Pos",
    "Extruder_torque": "Extruder Torque",
    "Injection_pressure": "Injection Pressure",
    "Injection_time": "Injection Time",
    "Peak_pressure_position": "Peak Pressure Position",
    "Peak_pressure_time": "Peak Pressure Time",
    "Scrap_counter": "Scrap Counter",
    "Shot_counter": "Shot Counter",
    "Shot_size": "Shot Size",
    "Switch_position": "Switch Over Position",
    "Switch_pressure": "Switch Over Pressure"
}
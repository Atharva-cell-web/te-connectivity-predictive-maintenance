# backend/config_limits.py

# ML thresholds (business-owned)
ML_THRESHOLDS = {
    "LOW": 0.30,
    "MEDIUM": 0.60
}

# Hard safety limits (Calculated from 1st-99th Percentile of Jan-Feb 2026 Data)
SAFE_LIMITS = {
    "Cushion": {
        "min": 0.0,
        "max": 8.27,
        "unit": "mm"
    },
    "Cycle_time": {
        "min": 0.0,
        "max": 30.68,
        "unit": "s"
    },
    "Cyl_tmp_z1": {
        "min": 0.0,
        "max": 572.1,
        "unit": "°C"
    },
    "Cyl_tmp_z3": {
        "min": 0.0,
        "max": 563.1,
        "unit": "°C"
    },
    "Cyl_tmp_z4": {
        "min": 0.0,
        "max": 554.3,
        "unit": "°C"
    },
    "Cyl_tmp_z5": {
        "min": 0.0,
        "max": 545.2,
        "unit": "°C"
    },
    "Cyl_tmp_z8": {
        "min": 0.0,
        "max": 175.3,
        "unit": "°C"
    },
    "Dosage_time": {
        "min": 0.0,
        "max": 4.22,
        "unit": "s"
    },
    "Injection_pressure": {
        "min": 0.0,
        "max": 1919.0,
        "unit": "bar"
    },
    "Injection_time": {
        "min": 0.0,
        "max": 0.62,
        "unit": "s"
    },
    "Peak_pressure_position": {
        "min": 0.0,
        "max": 16.48,
        "unit": "mm" 
    },
    "Peak_pressure_time": {
        "min": 0.0,
        "max": 0.58,
        "unit": "s"  
    },
    "Switch_position": {
        "min": 0.0,
        "max": 10.0,
        "unit": "mm"
    },
    "Switch_pressure": {
        "min": 0.0,
        "max": 1905.0,
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
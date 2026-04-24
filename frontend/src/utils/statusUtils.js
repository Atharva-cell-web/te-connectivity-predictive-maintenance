/**
 * Standard utility for mapping backend status values to consistent UI labels and styles.
 */

export const UI_STATUS = {
  NORMAL: "NORMAL",
  WARNING: "WARNING",
  CRITICAL: "CRITICAL",
  WATCH: "WATCH",
  HIGH: "HIGH",
};

/**
 * Maps raw backend status to standardized UI status.
 * @param {string} rawStatus - Status from API (e.g., "EXCEEDED", "NORMAL", "WARNING")
 * @param {number} riskScore - Optional risk probability to derive status if missing
 */
export const mapStatus = (rawStatus, riskScore = null) => {
  let effectiveStatus = rawStatus;
  let effectiveRisk = riskScore;

  // Polymorphic fix: If first arg is a number, treat as riskScore
  if (typeof rawStatus === 'number' && riskScore === null) {
    effectiveStatus = null;
    effectiveRisk = rawStatus;
  }

  if (!effectiveStatus && effectiveRisk === null) return UI_STATUS.NORMAL;

  const status = String(effectiveStatus || "").toUpperCase().trim();

  if (status === "EXCEEDED" || status === "CRITICAL" || status === "ERROR") return UI_STATUS.CRITICAL;
  if (status === "WARNING" || status === "ALERT") return UI_STATUS.WARNING;
  if (status === "WATCH") return UI_STATUS.WATCH;
  if (status === "HIGH") return UI_STATUS.HIGH;

  // Fallback to risk score if status is normal, missing, or unrecognized
  const numericRisk = Number(effectiveRisk);
  if (effectiveRisk !== null && !Number.isNaN(numericRisk)) {
    if (numericRisk >= 0.8) return UI_STATUS.CRITICAL;
    if (numericRisk >= 0.6) return UI_STATUS.HIGH;
    if (numericRisk >= 0.35) return UI_STATUS.WATCH;
  }

  return UI_STATUS.NORMAL;
};

/**
 * Returns Tailwind/CSS classes for a given UI status.
 */
export const getStatusColorClass = (uiStatus) => {
  switch (uiStatus) {
    case UI_STATUS.CRITICAL:
      return "text-red-600";
    case UI_STATUS.HIGH:
      return "text-orange-600";
    case UI_STATUS.WARNING:
    case UI_STATUS.WATCH:
      return "text-amber-600";
    default:
      return "text-emerald-600";
  }
};

/**
 * Returns the badge class defined in index.css.
 */
export const getStatusBadgeClass = (uiStatus) => {
  switch (uiStatus) {
    case UI_STATUS.CRITICAL:
      return "badge-critical";
    case UI_STATUS.HIGH:
      return "badge-high";
    case UI_STATUS.WARNING:
      return "badge-warning";
    case UI_STATUS.WATCH:
      return "badge-watch";
    default:
      return "badge-normal";
  }
};

/**
 * TE Project Specific: Maps sensor names to their official source and description
 * Based on TE Parameter Documentation.
 */
export const SENSOR_METADATA = {
  // MES / Process Parameters
  cushion: { source: "MES", label: "Cushion", unit: "mm", description: "Material left in front of screw after injection" },
  cycle_time: { source: "MES", label: "Cycle Time", unit: "s", description: "Full injection molding cycle duration" },
  cylinder_temp: { source: "MES", label: "Cylinder Temperature", unit: "°C", description: "Heating zone temperature(s)" },
  dosage_time: { source: "MES", label: "Dosage Time", unit: "s", description: "Time to fill screw/barrel to set position" },
  ejector_torque: { source: "MES", label: "Ejector Fix Deviation Torque", unit: "Nm", description: "Ejection force / torque safety metric" },
  extruder_start_pos: { source: "MES", label: "Extruder Start Position", unit: "mm", description: "Screw position at cycle start" },
  extruder_torque: { source: "MES", label: "Extruder Torque", unit: "Nm", description: "Extruder load / torque safety metric" },
  injection_pressure: { source: "MES", label: "Injection Pressure", unit: "bar", description: "Pressure during injection" },
  injection_time: { source: "MES", label: "Injection Time", unit: "s", description: "Time to inject to switchover position" },
  machine_status: { source: "MES", label: "Machine Status", unit: "state", description: "Running / Error / Idle state" },
  peak_pressure_pos: { source: "MES", label: "Peak Pressure Position", unit: "mm", description: "Screw position when peak pressure reached" },
  peak_pressure_time: { source: "MES", label: "Peak Pressure Time", unit: "s", description: "Time when peak pressure occurs" },
  scrap_counter: { source: "MES", label: "Scrap Counter", unit: "shots", description: "Rejected part count / scrap shots" },
  switch_position: { source: "MES", label: "Switch Position", unit: "mm", description: "Switchover screw position" },
  switch_pressure: { source: "MES", label: "Switch Pressure", unit: "bar", description: "Pressure at switchover point" },
};

export const formatSensorName = (sensor) => {
  const normKey = String(sensor || "").toLowerCase().replace(/ /g, "_");

  // Also try exact match for edge cases
  if (SENSOR_METADATA[normKey]) return SENSOR_METADATA[normKey].label;

  // Try to find matching key even if slightly different
  const matchedKey = Object.keys(SENSOR_METADATA).find(k => k === normKey || normKey.includes(k) || k.includes(normKey));
  if (matchedKey) return SENSOR_METADATA[matchedKey].label;

  return String(sensor || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

/**
 * Categorizes a sensor for UI grouping.
 */
export const getSensorCategory = (key) => {
  const normKey = String(key || "").toLowerCase();

  if (SENSOR_METADATA[normKey]) return SENSOR_METADATA[normKey].source;

  if (normKey.includes("temp") || normKey.includes("tmp")) return "TEMPERATURE";
  if (normKey.includes("press")) return "PRESSURE";
  if (normKey.includes("time")) return "TIMING";

  return "PROCESS";
};

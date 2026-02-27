import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, AlertTriangle, CheckCircle2, Gauge, Settings2 } from "lucide-react";

const API_BASE = "http://127.0.0.1:8080";
const FUTURE_RISK_THRESHOLD = 0.6;

const MACHINE_OPTIONS = [
  { value: "M-231", label: "M-231" },
  { value: "M-471", label: "M-471" },
  { value: "M-607", label: "M-607" },
  { value: "M-612", label: "M-612" },
];

const TIME_WINDOW_OPTIONS = [
  { value: 120, futureMinutes: 35, label: "2H Past / 35M Future" },
  { value: 60, futureMinutes: 20, label: "1H Past / 20M Future" },
];

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const mergeLimits = (safeLimits, overrides) => {
  const merged = { ...(safeLimits || {}) };
  Object.entries(overrides || {}).forEach(([sensor, override]) => {
    merged[sensor] = {
      ...(merged[sensor] || {}),
      ...override,
    };
  });
  return merged;
};

const detectBreaches = (latestSensors, limits) => {
  const breaches = [];
  Object.entries(latestSensors || {}).forEach(([sensor, rawValue]) => {
    const value = toNumber(rawValue);
    const limit = limits?.[sensor];
    if (value === null || !limit) {
      return;
    }
    const minValue = toNumber(limit.min);
    const maxValue = toNumber(limit.max);
    const span = Math.max(
      (maxValue ?? value) - (minValue ?? value),
      Math.abs(maxValue ?? 1),
      1,
    );

    if (maxValue !== null && value > maxValue) {
      breaches.push({
        sensor,
        direction: "above",
        value,
        threshold: maxValue,
        severity: (value - maxValue) / span,
      });
      return;
    }
    if (minValue !== null && value < minValue) {
      breaches.push({
        sensor,
        direction: "below",
        value,
        threshold: minValue,
        severity: (minValue - value) / span,
      });
    }
  });

  return breaches.sort((a, b) => b.severity - a.severity);
};

const formatClock = (timestamp) => {
  if (!timestamp || typeof timestamp !== "string") {
    return "";
  }
  return timestamp.slice(11, 16);
};

function GlobalHeader({
  machineId,
  timeWindowMinutes,
  onMachineChange,
  onTimeWindowChange,
  healthStatus,
}) {
  const statusClass =
    healthStatus === "HIGH"
      ? "border-red-500 bg-red-500/15 text-red-300"
      : healthStatus === "MEDIUM"
        ? "border-amber-500 bg-amber-500/15 text-amber-300"
        : "border-emerald-500 bg-emerald-500/15 text-emerald-300";

  return (
    <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-700 bg-slate-900/70 px-5 py-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Predictive Maintenance Control Room</h1>
        <p className="text-xs text-slate-400">Unified machine health, root cause, and telemetry view</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={machineId}
          onChange={(event) => onMachineChange(event.target.value)}
          className="rounded-md border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 outline-none"
        >
          {MACHINE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          value={timeWindowMinutes}
          onChange={(event) => onTimeWindowChange(Number(event.target.value))}
          className="rounded-md border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 outline-none"
        >
          {TIME_WINDOW_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <span className={`rounded-md border px-3 py-2 text-xs font-semibold ${statusClass}`}>
          STATUS: {healthStatus}
        </span>
      </div>
    </header>
  );
}

function SystemHealthMonitor({ timeline, riskScore }) {
  const chartData = useMemo(
    () =>
      (timeline || []).map((point) => {
        const risk = toNumber(point.risk_score) ?? 0;
        const alertDot = point.is_scrap_actual === 1 || (point.is_future && risk > FUTURE_RISK_THRESHOLD);
        return {
          ...point,
          pastRisk: point.is_future ? null : risk,
          futureRisk: point.is_future ? risk : null,
          alertDot,
        };
      }),
    [timeline],
  );

  if (!chartData.length) {
    return <div className="flex h-64 items-center justify-center text-sm text-slate-400">No timeline data available.</div>;
  }



  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
          <Activity size={16} className="text-cyan-300" />
          Section A: System Health Monitor
        </h2>
        <div className="text-xs text-slate-400">Current risk score: {(riskScore * 100).toFixed(1)}%</div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="timestamp" tickFormatter={formatClock} tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: 'Time', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: 'Risk Probability', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}
              formatter={(value) => (toNumber(value) ?? 0).toFixed(3)}
              labelFormatter={(value) => value}
            />
            <Legend verticalAlign="top" height={36} iconType="line" />
            <Line type="monotone" dataKey="pastRisk" name="Past (Actual)" stroke="#22d3ee" strokeWidth={2.4} dot={false} />
            <Line
              type="monotone"
              dataKey="futureRisk"
              name="Future (Predicted)"
              stroke="#f59e0b"
              strokeWidth={2.2}
              strokeDasharray="7 5"
              dot={false}
            />
            <ReferenceLine y={FUTURE_RISK_THRESHOLD} stroke="#ef4444" strokeDasharray="5 5" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function RootCauseAnalyzer({ timeline, sensor, safeLimits }) {
  const chartData = useMemo(() => {
    if (!sensor || !timeline || !timeline.length) return [];
    let lastPastIndex = -1;
    timeline.forEach((point, index) => { if (!point.is_future) lastPastIndex = index; });

    return timeline.map((point, index) => {
      const val = toNumber(point.sensors?.[sensor]);
      const isTransition = index === lastPastIndex;
      return {
        timestamp: point.timestamp,
        pastValue: (!point.is_future || isTransition) ? val : null,
        futureValue: (point.is_future || isTransition) ? val : null,
        rawValue: val
      };
    }).filter((point) => point.pastValue !== null || point.futureValue !== null);
  }, [timeline, sensor]);

  const limits = safeLimits?.[sensor] || {};
  const minLimit = toNumber(limits.min);
  const maxLimit = toNumber(limits.max);

  const yDomain = useMemo(() => {
    const values = chartData.map(d => d.rawValue).filter(v => v !== null);
    if (values.length === 0) return ['auto', 'auto'];

    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);

    const finalMin = minLimit !== null ? Math.min(dataMin, minLimit) : dataMin;
    const finalMax = maxLimit !== null ? Math.max(dataMax, maxLimit) : dataMax;

    const padding = (finalMax - finalMin) * 0.05;
    if (padding === 0) return [finalMin - 1, finalMax + 1];

    return [finalMin - padding, finalMax + padding];
  }, [chartData, minLimit, maxLimit]);

  // All-systems-normal fallback when no sensor is selected
  if (!sensor) {
    return (
      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
          <AlertTriangle size={16} className="text-amber-300" />
          Section B: Root Cause Analyzer
        </h2>
        <div className="flex h-72 flex-col items-center justify-center gap-3">
          <CheckCircle2 size={48} className="text-emerald-400" />
          <p className="text-lg font-semibold text-emerald-300">All Systems Normal</p>
          <p className="text-sm text-slate-400">No Root Cause to Display — click a Warning or Exceeded parameter in the telemetry grid</p>
        </div>
      </section>
    );
  }

  const displaySensorName = sensor.replace(/_/g, " ");

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
        <AlertTriangle size={16} className="text-amber-300" />
        Section B: Root Cause Analyzer ({displaySensorName})
      </h2>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="timestamp" tickFormatter={formatClock} tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: 'Time', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 12 }} />

            <YAxis domain={yDomain} tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: 'Sensor Value', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} />

            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}
              formatter={(value) => (toNumber(value) ?? 0).toFixed(3)}
              labelFormatter={(value) => value}
            />
            <Legend verticalAlign="top" height={36} iconType="line" />
            <ReferenceArea
              y1={minLimit !== null ? minLimit : undefined}
              y2={maxLimit !== null ? maxLimit : undefined}
              fillOpacity={0.1}
              fill="#10b981"
              label={{ value: 'Safe Zone', fill: '#10b981', fontSize: 10, position: 'insideTopRight' }}
            />

            <Line type="monotone" dataKey="pastValue" name="Past (Actual)" stroke="#60a5fa" strokeWidth={2.4} dot={false} connectNulls={true} />
            <Line type="monotone" dataKey="futureValue" name="Future (Predicted)" stroke="#f59e0b" strokeWidth={2.4} strokeDasharray="7 5" dot={false} connectNulls={true} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function TelemetryPanel({ timeline, latestSensors, safeLimits, selectedSensor, onSelectSensor, currentTimeWindow }) {
  const futurePoints = timeline?.filter(p => p.is_future) || [];
  const lastFuturePoint = futurePoints[futurePoints.length - 1] || {};
  let formattedFutureTime = `+${currentTimeWindow?.futureMinutes || 35} mins`;
  if (lastFuturePoint.timestamp) {
    formattedFutureTime = `at ${lastFuturePoint.timestamp.slice(11, 16)}`;
  }

  const tableData = useMemo(() => {
    if (!latestSensors || !safeLimits) return [];

    return Object.keys(safeLimits).map(sensor => {
      const current = toNumber(latestSensors[sensor]);
      const min = toNumber(safeLimits[sensor]?.min);
      const max = toNumber(safeLimits[sensor]?.max);
      const forecast = toNumber(lastFuturePoint.sensors?.[sensor]);

      let status = "good";
      let statusText = "Normal";

      if (current !== null) {
        let span = 100;
        if (min !== null && max !== null) span = max - min;
        else if (max !== null) span = max;
        else if (min !== null) span = min;

        if ((min !== null && current < min) || (max !== null && current > max)) {
          status = "critical";
          statusText = "Exceeded";
        } else if (
          (min !== null && current - min < span * 0.1) ||
          (max !== null && max - current < span * 0.1)
        ) {
          status = "warning";
          statusText = "Warning";
        }
      }

      return {
        sensorKey: sensor,
        sensor: sensor.replace(/_/g, " "),
        current,
        min,
        max,
        forecast,
        status,
        statusText
      };
    }).sort((a, b) => {
      const weights = { critical: 3, warning: 2, good: 1 };
      return weights[b.status] - weights[a.status];
    });
  }, [timeline, latestSensors, safeLimits, lastFuturePoint]);

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 flex flex-col h-full">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200 shrink-0">
        <Activity size={16} className="text-emerald-400" />
        Section C: Real-Time Telemetry Grid
      </h2>

      <div className="flex-1 overflow-auto rounded-lg border border-slate-700/50 bg-slate-900 shadow-inner">
        <table className="w-full text-left text-sm whitespace-nowrap">
          <thead className="sticky top-0 bg-slate-800 text-xs uppercase text-slate-400 z-10 shadow-md">
            <tr>
              <th className="px-4 py-3 font-medium">Parameter</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium">Current</th>
              <th className="px-4 py-3 font-medium">Safe Range</th>
              <th className="px-4 py-3 font-medium">
                Prediction
                <span className="block text-[10px] text-slate-500 font-normal normal-case mt-0.5 tracking-normal">
                  ({formattedFutureTime})
                </span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {tableData.map((row, idx) => {
              const isClickable = row.status === "critical" || row.status === "warning";
              const isSelected = selectedSensor === row.sensorKey;

              const rowClassName = [
                "transition-colors",
                isSelected
                  ? "bg-cyan-900/25 border-l-2 border-l-cyan-400"
                  : "",
                isClickable
                  ? "cursor-pointer hover:bg-slate-800/60"
                  : "cursor-default",
              ].filter(Boolean).join(" ");

              return (
                <tr
                  key={idx}
                  className={rowClassName}
                  onClick={() => { if (isClickable) onSelectSensor(row.sensorKey); }}
                >
                  <td className="px-4 py-3 font-medium text-slate-300 capitalize">{row.sensor.toLowerCase()}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${row.status === 'critical' ? 'bg-red-500 animate-pulse' :
                        row.status === 'warning' ? 'bg-amber-400' : 'bg-emerald-500'
                        }`}></span>
                      <span className={`text-xs font-semibold ${row.status === 'critical' ? 'text-red-400' :
                        row.status === 'warning' ? 'text-amber-300' : 'text-emerald-400'
                        }`}>{row.statusText}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-200">
                    {row.current !== null ? row.current.toFixed(2) : "--"}
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400">
                    {row.min !== null ? row.min : "0"} <span className="text-slate-600 px-1">~</span> {row.max !== null ? row.max : "∞"}
                  </td>
                  <td className="px-4 py-3 font-mono text-amber-400/90">
                    {row.forecast !== null ? row.forecast.toFixed(2) : "--"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PredictionSummary({ summaryStats, timeline }) {
  const stats = useMemo(() => {
    const pastPoints = timeline?.filter(p => !p.is_future) || [];
    const futurePoints = timeline?.filter(p => p.is_future) || [];

    const avgPastRisk = pastPoints.length ? (pastPoints.reduce((acc, p) => acc + p.risk_score, 0) / pastPoints.length) * 100 : 0;
    const avgFutureRisk = futurePoints.length ? (futurePoints.reduce((acc, p) => acc + p.risk_score, 0) / futurePoints.length) * 100 : 0;

    const pastScrapCount = summaryStats?.past_scrap_detected || 0;
    const futureScrapCount = summaryStats?.future_scrap_predicted || 0;

    const riskTrend = avgFutureRisk - avgPastRisk;

    return {
      pastScrapCount,
      futureScrapCount,
      avgPastRisk,
      riskTrend,
      trendText: riskTrend > 0 ? `Upward +${riskTrend.toFixed(1)}%` : riskTrend < 0 ? `Downward ${riskTrend.toFixed(1)}%` : "Stable 0.0%",
      trendColor: riskTrend > 0 ? "text-red-400" : riskTrend < 0 ? "text-emerald-400" : "text-slate-400",
      trendBorder: riskTrend > 0 ? "border-red-900/50 bg-red-900/10" : "border-slate-700/50 bg-slate-800/50"
    };
  }, [summaryStats, timeline]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Past Scrap Analysis Card */}
      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-semibold text-slate-200">Past Scrap Analysis</h3>
          <span className="px-2 py-1 text-[10px] uppercase font-bold tracking-wider text-emerald-400 bg-emerald-400/10 rounded">Actual</span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg border border-slate-700/50 bg-slate-800/50">
            <p className="text-xs text-slate-400 mb-1 font-medium">Total Past Scrap</p>
            <p className="text-2xl font-bold text-slate-100">{stats.pastScrapCount} <span className="text-sm font-normal text-slate-500">units</span></p>
            <p className="text-xs text-slate-500 mt-1">Last 4 hours</p>
          </div>
          <div className="p-4 rounded-lg border border-cyan-900/50 bg-cyan-900/10">
            <p className="text-xs text-cyan-400/70 mb-1 font-medium">Average Past Risk</p>
            <p className="text-2xl font-bold text-cyan-400">{stats.avgPastRisk.toFixed(1)}%</p>
            <p className="text-xs text-cyan-500/50 mt-1">Historical baseline</p>
          </div>
        </div>
      </section>

      {/* Future Scrap Forecast Card */}
      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-semibold text-slate-200">Future Scrap Forecast</h3>
          <span className="px-2 py-1 text-[10px] uppercase font-bold tracking-wider text-amber-400 bg-amber-400/10 rounded">Predicted</span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg border border-amber-900/50 bg-amber-900/10">
            <p className="text-xs text-amber-400/70 mb-1 font-medium">Predicted Scrap</p>
            <p className="text-2xl font-bold text-amber-400">{stats.futureScrapCount} <span className="text-sm font-normal text-amber-700/50">units</span></p>
            <p className="text-xs text-amber-500/50 mt-1">Next 60 min</p>
          </div>
          <div className={`p-4 rounded-lg border ${stats.trendBorder}`}>
            <p className="text-xs text-slate-400 mb-1 font-medium">Trend Comparison</p>
            <p className={`text-2xl font-bold ${stats.trendColor}`}>{stats.trendText}</p>
            <p className="text-xs text-slate-500 mt-1">Future vs Past</p>
          </div>
        </div>
      </section>
    </div>
  );
}

function App() {
  const [machineId, setMachineId] = useState("M-231");
  const [timeWindowMinutes, setTimeWindowMinutes] = useState(120);
  const [controlRoomData, setControlRoomData] = useState(null);
  const [limitOverrides, setLimitOverrides] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState(null);

  const selectedOption = TIME_WINDOW_OPTIONS.find(o => o.value === timeWindowMinutes) || TIME_WINDOW_OPTIONS[0];

  const fetchControlRoom = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/api/control-room/${machineId}`, {
        params: { time_window: timeWindowMinutes, future_window: selectedOption.futureMinutes },
      });
      setControlRoomData(response.data);
      setError(null);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [machineId, timeWindowMinutes, selectedOption.futureMinutes]);

  useEffect(() => {
    fetchControlRoom();
    const intervalId = setInterval(fetchControlRoom, 15000);
    return () => clearInterval(intervalId);
  }, [fetchControlRoom]);

  useEffect(() => {
    setLimitOverrides({});
    setSelectedSensor(null);
  }, [machineId, timeWindowMinutes]);

  const safeLimits = controlRoomData?.safe_limits || {};
  const effectiveLimits = useMemo(() => mergeLimits(safeLimits, limitOverrides), [safeLimits, limitOverrides]);
  const timeline = controlRoomData?.timeline || [];

  const latestPastPoint = useMemo(() => {
    const past = timeline.filter((point) => !point.is_future);
    return past.length ? past[past.length - 1] : null;
  }, [timeline]);

  const latestSensors = latestPastPoint?.sensors || {};
  const breaches = useMemo(() => detectBreaches(latestSensors, effectiveLimits), [latestSensors, effectiveLimits]);

  const activeRootCauses = useMemo(() => {
    const apiRootCauses = controlRoomData?.current_health?.root_causes || [];
    if (breaches.length) {
      return breaches.map((entry) => entry.sensor);
    }
    return apiRootCauses.length ? apiRootCauses : ["Injection_pressure"];
  }, [controlRoomData, breaches]);

  const adjustedRiskScore = useMemo(() => {
    const base = toNumber(controlRoomData?.current_health?.risk_score) ?? 0;
    if (!breaches.length) {
      return base;
    }
    const penalty = Math.min(0.35, breaches.length * 0.12);
    return Math.min(1, base + penalty);
  }, [controlRoomData, breaches]);

  const displayTimeline = useMemo(() => {
    if (!timeline.length) {
      return [];
    }
    const futurePenalty = breaches.length ? Math.min(0.2, breaches.length * 0.05) : 0;
    let lastPastIndex = -1;
    timeline.forEach((point, index) => {
      if (!point.is_future) {
        lastPastIndex = index;
      }
    });

    return timeline.map((point, index) => {
      let risk = toNumber(point.risk_score) ?? 0;
      if (index === lastPastIndex) {
        risk = Math.max(risk, adjustedRiskScore);
      }
      if (point.is_future) {
        risk = Math.min(1, risk + futurePenalty);
      }
      return {
        ...point,
        risk_score: Number(risk.toFixed(4)),
      };
    });
  }, [timeline, breaches.length, adjustedRiskScore]);

  const summaryStats = useMemo(() => {
    const base = controlRoomData?.summary_stats || { past_scrap_detected: 0, future_scrap_predicted: 0 };
    const timelineFutureCount = displayTimeline.filter(
      (point) => point.is_future && (toNumber(point.risk_score) ?? 0) > FUTURE_RISK_THRESHOLD,
    ).length;
    return {
      past_scrap_detected: base.past_scrap_detected ?? 0,
      future_scrap_predicted: Math.max(base.future_scrap_predicted ?? 0, timelineFutureCount),
    };
  }, [controlRoomData, displayTimeline]);

  const currentHealth = useMemo(() => {
    let status = "LOW";
    if (breaches.length || adjustedRiskScore >= FUTURE_RISK_THRESHOLD) {
      status = "HIGH";
    } else if (adjustedRiskScore >= 0.3) {
      status = "MEDIUM";
    }
    return {
      status,
      risk_score: adjustedRiskScore,
      root_causes: activeRootCauses,
    };
  }, [breaches.length, adjustedRiskScore, activeRootCauses]);

  // --- Telemetry status map for auto-switching logic ---
  const telemetryStatuses = useMemo(() => {
    const statuses = {};
    const limits = effectiveLimits;
    Object.keys(limits).forEach((sensor) => {
      const current = toNumber(latestSensors[sensor]);
      const min = toNumber(limits[sensor]?.min);
      const max = toNumber(limits[sensor]?.max);
      let status = "good";
      if (current !== null) {
        let span = 100;
        if (min !== null && max !== null) span = max - min;
        else if (max !== null) span = max;
        else if (min !== null) span = min;
        if ((min !== null && current < min) || (max !== null && current > max)) {
          status = "critical";
        } else if (
          (min !== null && current - min < span * 0.1) ||
          (max !== null && max - current < span * 0.1)
        ) {
          status = "warning";
        }
      }
      statuses[sensor] = status;
    });
    return statuses;
  }, [latestSensors, effectiveLimits]);

  // Find the first abnormal sensor (critical first, then warning)
  const firstAbnormalSensor = useMemo(() => {
    const critical = Object.keys(telemetryStatuses).find(s => telemetryStatuses[s] === "critical");
    if (critical) return critical;
    const warning = Object.keys(telemetryStatuses).find(s => telemetryStatuses[s] === "warning");
    return warning || null;
  }, [telemetryStatuses]);

  // Smart auto-switch: clear selection if sensor returned to normal
  useEffect(() => {
    if (selectedSensor && telemetryStatuses[selectedSensor] === "good") {
      setSelectedSensor(firstAbnormalSensor);
    }
  }, [telemetryStatuses, selectedSensor, firstAbnormalSensor]);

  // Derive activeSensor: user selection > first abnormal > null (all normal)
  const activeSensor = selectedSensor || firstAbnormalSensor;

  if (loading && !controlRoomData) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-200">
        Loading Control Room data...
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 p-4 text-slate-100 md:p-6">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-4">
        <GlobalHeader
          machineId={machineId}
          timeWindowMinutes={timeWindowMinutes}
          onMachineChange={setMachineId}
          onTimeWindowChange={setTimeWindowMinutes}
          healthStatus={currentHealth.status}
        />

        {error && <div className="rounded-md border border-red-700 bg-red-950/50 px-4 py-2 text-sm text-red-200">{error}</div>}

        {/* MAIN DASHBOARD GRID */}
        <div className="grid grid-cols-12 gap-4">

          {/* Section A: Full Width Top */}
          <div className="col-span-12">
            <SystemHealthMonitor timeline={displayTimeline} riskScore={currentHealth.risk_score} />
          </div>

          {/* Section B & C: Stacked vertically full width */}
          <div className="col-span-12 flex flex-col">
            <RootCauseAnalyzer timeline={displayTimeline} sensor={activeSensor} safeLimits={effectiveLimits} />
          </div>
          <div className="col-span-12 flex flex-col h-[500px]">
            <TelemetryPanel
              timeline={displayTimeline}
              latestSensors={latestSensors}
              safeLimits={effectiveLimits}
              selectedSensor={selectedSensor}
              onSelectSensor={setSelectedSensor}
              timeWindowOptions={TIME_WINDOW_OPTIONS}
              currentTimeWindow={selectedOption}
            />
          </div>

          {/* Section D: Full Width Bottom */}
          <div className="col-span-12 mt-2">
            <PredictionSummary summaryStats={summaryStats} timeline={displayTimeline} />
          </div>

        </div>
      </div>
    </main>
  );
}

export default App;
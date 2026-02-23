import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, AlertTriangle, Gauge, Settings2 } from "lucide-react";

const API_BASE = "http://127.0.0.1:8080";
const FUTURE_RISK_THRESHOLD = 0.6;

const MACHINE_OPTIONS = [
  { value: "M-231", label: "M-231" },
  { value: "M-471", label: "M-471" },
  { value: "M-607", label: "M-607" },
  { value: "M-612", label: "M-612" },
];

const TIME_WINDOW_OPTIONS = [
  { value: 60, label: "Past 1h + Future 15m" },
  { value: 120, label: "Past 2h + Future 30m" },
  { value: 240, label: "Past 4h + Future 1h" },
  { value: 360, label: "Past 6h + Future 1h30m" },
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

  const renderAlertDot = (props) => {
    const { cx, cy, payload } = props;
    if (!payload?.alertDot) {
      return null;
    }
    return <circle cx={cx} cy={cy} r={4} fill="#ef4444" stroke="#fee2e2" strokeWidth={1} />;
  };

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
            <XAxis dataKey="timestamp" tickFormatter={formatClock} tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}
              formatter={(value) => (toNumber(value) ?? 0).toFixed(3)}
              labelFormatter={(value) => value}
            />
            <Line type="monotone" dataKey="pastRisk" stroke="#22d3ee" strokeWidth={2.4} dot={renderAlertDot} />
            <Line
              type="monotone"
              dataKey="futureRisk"
              stroke="#f59e0b"
              strokeWidth={2.2}
              strokeDasharray="7 5"
              dot={renderAlertDot}
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
    if (!timeline || !timeline.length) return [];
    
    let lastPastIndex = -1;
    timeline.forEach((point, index) => {
      if (!point.is_future) lastPastIndex = index;
    });

    return timeline.map((point, index) => {
      const val = toNumber(point.sensors?.[sensor]);
      const isTransition = index === lastPastIndex;
      
      return {
        timestamp: point.timestamp,
        pastValue: (!point.is_future || isTransition) ? val : null,
        futureValue: (point.is_future || isTransition) ? val : null,
      };
    }).filter((point) => point.pastValue !== null || point.futureValue !== null); 
    // ^ The filter is back, but it won't break the graph because the backend is now sending real future data!
  }, [timeline, sensor]);

  const limits = safeLimits?.[sensor] || {};
  const minLimit = toNumber(limits.min);
  const maxLimit = toNumber(limits.max);

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
        <AlertTriangle size={16} className="text-amber-300" />
        Section B: Root Cause Analyzer ({sensor})
      </h2>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="timestamp" tickFormatter={formatClock} tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}
              formatter={(value) => (toNumber(value) ?? 0).toFixed(3)}
              labelFormatter={(value) => value}
            />
            {minLimit !== null && <ReferenceLine y={minLimit} stroke="#f97316" strokeDasharray="4 4" />}
            {maxLimit !== null && <ReferenceLine y={maxLimit} stroke="#ef4444" strokeDasharray="4 4" />}
            
            <Line type="monotone" dataKey="pastValue" stroke="#60a5fa" strokeWidth={2.4} dot={false} connectNulls={true} />
            <Line type="monotone" dataKey="futureValue" stroke="#f59e0b" strokeWidth={2.4} strokeDasharray="7 5" dot={false} connectNulls={true} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function TelemetryPanel({ latestSensors, effectiveLimits, breaches, onLimitOverride }) {
  const breachMap = useMemo(
    () =>
      (breaches || []).reduce((acc, entry) => {
        acc[entry.sensor] = entry;
        return acc;
      }, {}),
    [breaches],
  );

  const cards = Object.entries(latestSensors || {}).sort((a, b) => a[0].localeCompare(b[0]));

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
        <Gauge size={16} className="text-violet-300" />
        Section C: Telemetry Panel
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {cards.map(([sensor, rawValue]) => {
          const current = toNumber(rawValue);
          const limits = effectiveLimits?.[sensor] || {};
          const minValue = limits.min ?? "";
          const maxValue = limits.max ?? "";
          const breach = breachMap[sensor];
          const warningClass = breach
            ? "border-red-500 bg-red-950/60 shadow-[0_0_0_1px_rgba(239,68,68,0.3)] animate-pulse"
            : "border-slate-700 bg-slate-950/60";

          return (
            <article key={sensor} className={`rounded-lg border p-3 ${warningClass}`}>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-200">{sensor}</h3>
                {breach && <span className="text-[10px] font-semibold text-red-300">BREACH</span>}
              </div>
              <p className="font-mono text-lg text-slate-100">{current !== null ? current.toFixed(3) : "N/A"}</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <label className="text-[11px] text-slate-400">
                  Min
                  <input
                    type="number"
                    value={minValue}
                    onChange={(event) => onLimitOverride(sensor, "min", event.target.value)}
                    className="mt-1 w-full rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-200 outline-none"
                  />
                </label>
                <label className="text-[11px] text-slate-400">
                  Max
                  <input
                    type="number"
                    value={maxValue}
                    onChange={(event) => onLimitOverride(sensor, "max", event.target.value)}
                    className="mt-1 w-full rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-200 outline-none"
                  />
                </label>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function PredictionSummary({ machineInfo, summaryStats, health, activeSensor }) {
  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-200">
        <Settings2 size={16} className="text-sky-300" />
        Section D: Prediction Summary
      </h2>
      <div className="space-y-2 text-sm">
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Machine:</span> {machineInfo?.id}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Tool ID:</span> {machineInfo?.tool_id}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Part Number:</span> {machineInfo?.part_number}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Past Scrap Detected:</span> {summaryStats?.past_scrap_detected ?? 0}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Future Scrap Predicted:</span> {summaryStats?.future_scrap_predicted ?? 0}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Health Status:</span> {health?.status}
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Health Risk Score:</span> {((health?.risk_score ?? 0) * 100).toFixed(1)}%
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950/60 p-2 text-slate-200">
          <span className="text-slate-400">Active Root Cause:</span> {activeSensor}
        </div>
      </div>
    </section>
  );
}

function App() {
  const [machineId, setMachineId] = useState("M-231");
  const [timeWindowMinutes, setTimeWindowMinutes] = useState(240);
  const [controlRoomData, setControlRoomData] = useState(null);
  const [limitOverrides, setLimitOverrides] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchControlRoom = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/api/control-room/${machineId}`, {
        params: { time_window: timeWindowMinutes },
      });
      setControlRoomData(response.data);
      setError(null);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [machineId, timeWindowMinutes]);

  useEffect(() => {
    fetchControlRoom();
    const intervalId = setInterval(fetchControlRoom, 15000);
    return () => clearInterval(intervalId);
  }, [fetchControlRoom]);

  useEffect(() => {
    setLimitOverrides({});
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

  const activeSensor = activeRootCauses[0] || "Injection_pressure";

  const handleLimitOverride = useCallback((sensor, bound, rawValue) => {
    setLimitOverrides((previous) => {
      const next = { ...previous };
      const parsed = rawValue === "" ? null : Number(rawValue);
      const validValue = parsed !== null && Number.isFinite(parsed) ? parsed : null;

      const sensorOverrides = { ...(next[sensor] || {}) };
      if (validValue === null) {
        delete sensorOverrides[bound];
      } else {
        sensorOverrides[bound] = validValue;
      }

      if (Object.keys(sensorOverrides).length === 0) {
        delete next[sensor];
      } else {
        next[sensor] = sensorOverrides;
      }
      return next;
    });
  }, []);

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

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12">
            <SystemHealthMonitor timeline={displayTimeline} riskScore={currentHealth.risk_score} />
          </div>

          <div className="col-span-12 xl:col-span-8">
            <RootCauseAnalyzer timeline={displayTimeline} sensor={activeSensor} safeLimits={effectiveLimits} />
          </div>

          <div className="col-span-12 grid gap-4 xl:col-span-4">
            <TelemetryPanel
              latestSensors={latestSensors}
              effectiveLimits={effectiveLimits}
              breaches={breaches}
              onLimitOverride={handleLimitOverride}
            />
            <PredictionSummary
              machineInfo={controlRoomData?.machine_info}
              summaryStats={summaryStats}
              health={currentHealth}
              activeSensor={activeSensor}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

export default App;

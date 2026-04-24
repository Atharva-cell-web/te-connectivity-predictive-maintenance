import React, { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart,
} from "recharts";
import { Activity } from "lucide-react";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import { mapStatus, UI_STATUS, getStatusBadgeClass, getStatusColorClass } from "../utils/statusUtils";

dayjs.extend(utc);
dayjs.extend(timezone);

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

export default function HealthMonitor({ timeline, riskScore, auditAreas = [] }) {
  const chartData = useMemo(
    () =>
      (timeline || [])
        .map((point) => {
          const risk = toNumber(point.risk_score) ?? 0;
          const isFuture = point.type === "future" || point.is_future;
          const isBridge = point.type === "bridge";

          let time = null;
          if (typeof point.timestamp === "number" && Number.isFinite(point.timestamp)) {
            time = point.timestamp;
          } else if (typeof point.timestamp === "string") {
            const rawTs = point.timestamp.replace(" ", "T");
            time = Date.parse(rawTs);
          }
          if (!Number.isFinite(time)) return null;

          if (isBridge) {
            return {
              time,
              pastRisk: risk,
              futureRisk: toNumber(point.bridge_future_risk) ?? risk,
              isFuture: false,
              isBridge: true,
            };
          }

          return {
            time,
            pastRisk: isFuture ? null : risk,
            futureRisk: isFuture ? risk : null,
            isFuture,
            isBridge: false,
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.time - b.time),
    [timeline],
  );

  const predictionStartTime = useMemo(() => {
    const bridge = chartData.find((p) => p.isBridge);
    if (bridge) return bridge.time;
    const firstFuture = chartData.find((p) => p.isFuture);
    return firstFuture ? firstFuture.time : null;
  }, [chartData]);

  const riskBand = useMemo(() => {
    // Explicitly pass riskScore to the second argument for robust mapping
    const status = mapStatus(null, riskScore);
    return {
      label: status,
      tone: getStatusBadgeClass(status)
    };
  }, [riskScore]);

  if (!chartData.length) {
    return (
      <div className="glass-card flex h-64 items-center justify-center text-sm text-slate-400">
        No timeline data available.
      </div>
    );
  }

  const tooltipLabelFormatter = (value) => {
    if (value === null || value === undefined) return "";
    // Senior Pro Fix: Use .utc() to ensure "1 Means 1" timeline integrity, 
    // eliminating regional shifts (like IST +5.5h) in tooltips.
    return dayjs(value).utc().format("MMM DD YYYY HH:mm:ss");
  };

  const riskPercent = (riskScore * 100).toFixed(1);

  return (
    <section className="glass-card p-5 animate-slide-up" style={{ animationDelay: "0.15s" }}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-700">
            <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center">
              <Activity size={16} className="text-brand-500" />
            </div>
            System Health Monitor
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Past actual risk is connected to the forecast horizon at the prediction boundary.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-[0.2em] ${riskBand.tone}`}
          >
            {riskBand.label}
          </span>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2 text-right">
            <span className="block text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Current Risk
            </span>
            <span
              className={`text-2xl font-extrabold ${getStatusColorClass(mapStatus(null, riskScore))}`}
            >
              {riskPercent}%
            </span>
          </div>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
        <span className="rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1 text-emerald-700">
          Safe 0.00 - 0.35
        </span>
        <span className="rounded-full border border-amber-100 bg-amber-50 px-3 py-1 text-amber-700">
          Watch 0.35 - 0.60
        </span>
        <span className="rounded-full border border-orange-100 bg-orange-50 px-3 py-1 text-orange-700">
          High 0.60 - 0.80
        </span>
        <span className="rounded-full border border-red-100 bg-red-50 px-3 py-1 text-red-700">
          Critical 0.80 - 1.00
        </span>
      </div>

      <div className="h-[22rem] rounded-2xl border border-slate-100 bg-slate-50/50 px-3 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 16, right: 35, bottom: 0, left: 8 }}>
            <defs>
              <linearGradient id="pastGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="futureGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f97316" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <ReferenceArea y1={0} y2={0.35} fill="#10b981" fillOpacity={0.05} strokeOpacity={0} />
            <ReferenceArea y1={0.35} y2={0.6} fill="#f59e0b" fillOpacity={0.05} strokeOpacity={0} />
            <ReferenceArea y1={0.6} y2={0.8} fill="#f97316" fillOpacity={0.05} strokeOpacity={0} />
            <ReferenceArea y1={0.8} y2={1} fill="#ef4444" fillOpacity={0.05} strokeOpacity={0} />

            {/* Senior Feature: Ground-Truth Scrap Events Overlays */}
            {auditAreas.map((area, idx) => (
              <ReferenceArea
                key={`audit-${idx}`}
                x1={area.start}
                x2={area.end}
                fill="#ec4899"
                fillOpacity={0.12}
                strokeOpacity={0}
                label={{
                  value: area.id,
                  position: 'insideTopLeft',
                  fill: '#be185d',
                  fontSize: 10,
                  fontWeight: '900',
                  opacity: 0.8,
                  offset: 15
                }}
              />
            ))}
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="time"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => dayjs(t).utc().format("HH:mm")}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={{ stroke: "#cbd5e1" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={{ stroke: "#cbd5e1" }}
              label={{
                value: "Risk Probability",
                angle: -90,
                position: "insideLeft",
                offset: -10,
                fill: "#64748b",
                fontSize: 12,
                fontWeight: 600
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(255,255,255,0.75)",
                WebkitBackdropFilter: "blur(16px)",
                backdropFilter: "blur(16px)",
                border: "1px solid rgba(255,255,255,0.8)",
                borderRadius: "16px",
                color: "#1e293b",
                boxShadow: "0 10px 40px -10px rgba(0,0,0,0.15)",
              }}
              formatter={(value, name) => [(toNumber(value) ?? 0).toFixed(4), name]}
              labelFormatter={tooltipLabelFormatter}
            />
            <Legend
              verticalAlign="top"
              height={36}
              iconType="circle"
              wrapperStyle={{ fontSize: "12px", fontWeight: 600, color: "#475569" }}
            />
            <Area
              type="monotone"
              dataKey="pastRisk"
              name="Past (Actual)"
              stroke="#3b82f6"
              strokeWidth={2.5}
              fill="url(#pastGradient)"
              dot={false}
              connectNulls={false}
              activeDot={{ r: 5 }}
            />
            <Area
              type="monotone"
              dataKey="futureRisk"
              name="Future (Predicted)"
              stroke="#f97316"
              strokeWidth={2.5}
              strokeDasharray="6 4"
              fill="url(#futureGradient)"
              dot={{ r: 4, fill: "#f97316", stroke: "#fff", strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              connectNulls={false}
            />
            {predictionStartTime && (
              <ReferenceLine
                x={predictionStartTime}
                stroke="#94a3b8"
                strokeWidth={2}
                strokeDasharray="4 4"
                label={{
                  value: "Prediction boundary",
                  fill: "#475569",
                  fontSize: 11,
                  fontWeight: 800,
                  position: "top",
                  offset: 15,
                  background: { fill: '#f1f5f9', opacity: 0.8 }
                }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

import React, { useCallback, useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import { Activity, AlertTriangle, BarChart3, Clock3, ShieldAlert } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import apiClient, { getWsUrl } from "./utils/apiClient";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import {
  HealthMonitorSkeleton,
  TelemetryGridSkeleton,
  CardSkeleton
} from "./components/DashboardSkeletons";
import { mapStatus, UI_STATUS, getStatusColorClass } from "./utils/statusUtils";

const HealthMonitor = React.lazy(() => import("./components/HealthMonitor"));
const RootCause = React.lazy(() => import("./components/RootCause"));
const TelemetryGrid = React.lazy(() => import("./components/TelemetryGrid"));
const PredictionStats = React.lazy(() => import("./components/PredictionStats"));
const SensorDrawer = React.lazy(() => import("./components/SensorDrawer"));
const IngestionHub = React.lazy(() => import("./components/ingestion/IngestionHub"));
const AnalyticsHub = React.lazy(() => import("./components/AnalyticsHub"));
const AuditHub = React.lazy(() => import("./components/AuditHub"));

dayjs.extend(utc);
dayjs.extend(timezone);

const FUTURE_RISK_THRESHOLD = 0.6;

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

const fetchWithRetry = async (url, options = {}, retries = 3, delay = 1000) => {
  try {
    const response = await apiClient.get(url, options);
    return response.data;
  } catch (err) {
    if (retries > 0) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries - 1, delay);
    }
    throw err;
  }
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('jwt_token'));
  const [currentView, setCurrentView] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [machineId, setMachineId] = useState("M-356");
  const [anchorTime, setAnchorTime] = useState(null);
  const [draftPastWindowMinutes, setDraftPastWindowMinutes] = useState(60);
  const [draftFutureWindowMinutes, setDraftFutureWindowMinutes] = useState(30);
  const [pastWindowMinutes, setPastWindowMinutes] = useState(60);
  const [futureWindowMinutes, setFutureWindowMinutes] = useState(30);
  const [controlRoomData, setControlRoomData] = useState(null);
  const [limitOverrides, setLimitOverrides] = useState({});
  const [loading, setLoading] = useState(true);
  const [backgroundLoading, setBackgroundLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendConnecting, setBackendConnecting] = useState(false);
  const [fleet, setFleet] = useState([]); // Canonical machine registry for ID resolution
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const hasResolvedDefaultMachine = React.useRef(false);
  const wsRetryDelay = React.useRef(1000); // exponential backoff for WS reconnects
  const controlRoomRequestRef = React.useRef(0);

  const gridRef = React.useRef(null);

  const hasPendingWindowChanges =
    draftPastWindowMinutes !== pastWindowMinutes || draftFutureWindowMinutes !== futureWindowMinutes;

  const fetchControlRoom = useCallback(async ({
    showFullLoading = false,
    requestId = controlRoomRequestRef.current,
    targetMachineId = machineId,
    targetPastWindowMinutes = pastWindowMinutes,
    targetFutureWindowMinutes = futureWindowMinutes,
    targetAnchorTime = anchorTime,
  } = {}) => {
    try {
      if (showFullLoading) setLoading(true);
      else setBackgroundLoading(true);

      const params = { time_window: targetPastWindowMinutes, future_window: targetFutureWindowMinutes };
      if (targetAnchorTime) {
          params.anchor_time = targetAnchorTime;
      }

      const data = await fetchWithRetry(`/api/control-room/${targetMachineId}`, { params });
      if (controlRoomRequestRef.current !== requestId) return;
      setControlRoomData(data);
      setError(null);
    } catch (requestError) {
      if (controlRoomRequestRef.current !== requestId) return;
      setError(requestError.message);
    } finally {
      if (controlRoomRequestRef.current === requestId) {
        setLoading(false);
        setBackgroundLoading(false);
      }
    }
  }, [machineId, pastWindowMinutes, futureWindowMinutes, anchorTime]);

  useEffect(() => {
    const handleUnauthorized = () => setIsAuthenticated(false);
    window.addEventListener('auth-unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth-unauthorized', handleUnauthorized);
  }, []);

  useEffect(() => {
    if (!isAuthenticated || hasResolvedDefaultMachine.current) return;

    let cancelled = false;

    const ensureDefaultMachine = async () => {
      try {
        const response = await apiClient.get("/api/machines");
        if (cancelled) return;

        const machines = Array.isArray(response.data) ? response.data : [];
        setFleet(machines); // Store for canonical mapping
        if (machines.length === 0) return;

        const preferredMachine = machines.find((machine) => machine?.id === "M-356");
        const fallbackMachine = machines[0];
        const nextMachineId = preferredMachine?.id || fallbackMachine?.id || "M-356";

        hasResolvedDefaultMachine.current = true;
        if (nextMachineId && nextMachineId !== machineId) {
          setMachineId(nextMachineId);
        }
      } catch (error) {
        console.error("Failed to resolve default machine:", error);
        hasResolvedDefaultMachine.current = true;
      }
    };

    ensureDefaultMachine();

    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, machineId]);

  // Handle Manual Historical Replay Trigger
  useEffect(() => {
    if (isAuthenticated && machineId && anchorTime) {
      const requestId = controlRoomRequestRef.current + 1;
      controlRoomRequestRef.current = requestId;
      
      // Clear the previous machine's data immediately so we never render stale values.
      setControlRoomData(null);
      
      fetchControlRoom({
        showFullLoading: true,
        requestId,
        targetMachineId: machineId,
        targetPastWindowMinutes: pastWindowMinutes,
        targetAnchorTime: anchorTime,
      });
    }
  }, [anchorTime, machineId, isAuthenticated, fetchControlRoom, pastWindowMinutes]);

  useEffect(() => {
    if (!isAuthenticated || anchorTime) return;
    const requestId = controlRoomRequestRef.current + 1;
    controlRoomRequestRef.current = requestId;

    // Clear the previous machine's data immediately so we never render stale values.
    setControlRoomData(null);
    setError(null);
    setLoading(true);
    setBackgroundLoading(false);
    setSelectedSensor(null);
    setIsDrawerOpen(false);
    setLimitOverrides({});

    fetchControlRoom({
      showFullLoading: true,
      requestId,
      targetMachineId: machineId,
      targetPastWindowMinutes: pastWindowMinutes,
      targetFutureWindowMinutes: futureWindowMinutes,
    });

    // Normalize machineId (remove dashes) to ensure 100% backend route matching
    const normalizedMachineId = machineId.replace(/[^A-Z0-9]/gi, "");
    let wsPath = `/ws/control-room/${normalizedMachineId}?time_window=${pastWindowMinutes}&future_window=${futureWindowMinutes}`;
    if (anchorTime) wsPath += `&anchor_time=${anchorTime}`;
    const wsUrl = getWsUrl(wsPath);

    let socket = null;
    let destroyed = false;
    let fallbackInterval = null;
    let retryTimeout = null;

    const connect = () => {
      if (destroyed) return;
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        wsRetryDelay.current = 1000; // reset backoff on successful connect
        setBackendConnecting(false);
      };

      socket.onmessage = (event) => {
        try {
          if (controlRoomRequestRef.current !== requestId) return;
          const data = JSON.parse(event.data);
          setControlRoomData(data);
          setError(null);
          setLoading(false);
          setBackgroundLoading(false);
          setBackendConnecting(false);
          if (fallbackInterval) {
            clearInterval(fallbackInterval);
            fallbackInterval = null;
          }
        } catch (e) {
          console.error("WS Parse Error:", e);
        }
      };

      socket.onerror = () => {
        // WS errors are noisy — let onclose handle the reconnect
      };

      socket.onclose = (ev) => {
        if (destroyed) return;
        if (controlRoomRequestRef.current !== requestId) return;
        // If we never got data, start HTTP fallback polling
        if (!fallbackInterval) {
          fetchControlRoom({
            showFullLoading: false,
            requestId,
            targetMachineId: machineId,
            targetPastWindowMinutes: pastWindowMinutes,
            targetFutureWindowMinutes: futureWindowMinutes,
          });
          fallbackInterval = setInterval(
            () => fetchControlRoom({
              showFullLoading: false,
              requestId,
              targetMachineId: machineId,
              targetPastWindowMinutes: pastWindowMinutes,
              targetFutureWindowMinutes: futureWindowMinutes,
            }),
            30000
          );
        }
        // Reconnect with exponential backoff (cap at 30s)
        const delay = Math.min(wsRetryDelay.current, 30000);
        wsRetryDelay.current = Math.min(delay * 2, 30000);
        if (delay > 3000) setBackendConnecting(true);
        retryTimeout = setTimeout(() => {
          if (!destroyed) connect();
        }, delay);
      };
    };

    connect();

    return () => {
      destroyed = true;
      wsRetryDelay.current = 1000;
      if (retryTimeout) clearTimeout(retryTimeout);
      if (fallbackInterval) clearInterval(fallbackInterval);
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close();
      }
    };
  }, [machineId, pastWindowMinutes, futureWindowMinutes, fetchControlRoom, isAuthenticated, anchorTime]);

  const handleApplyWindows = useCallback(() => {
    setPastWindowMinutes(draftPastWindowMinutes);
    setFutureWindowMinutes(draftFutureWindowMinutes);
    setAnchorTime(null);
  }, [draftPastWindowMinutes, draftFutureWindowMinutes]);

  const handleReplayAnomaly = useCallback((machineRef, startTs) => {
    // Senior Sync Logic: Search fleet for canonical ID (e.g. M356 -> M-356)
    // This ensures Sidebar/Header selection remains active after the jump.
    const normalizedInput = String(machineRef || "").replace(/[^A-Z0-9]/gi, "").toUpperCase();
    const canonical = fleet.find(m => 
      m.id?.replace(/[^A-Z0-9]/gi, "").toUpperCase() === normalizedInput ||
      m.machine_id_normalized?.toUpperCase() === normalizedInput
    );
    
    const targetId = canonical?.id || machineRef;
    
    // Set anchor time (the split between past/future).
    // If the user's starting point is e.g. 5:00 PM: 
    // Past window: 5:00 PM to 6:00 PM (60 mins)
    // Future window: 6:00 PM to 6:30 PM (30 mins)
    // Therefore, the anchor is start time + 60 minutes.
    const anchorTimeMs = startTs + (60 * 60 * 1000);
    const targetAnchor = new Date(anchorTimeMs).toISOString();
    
    setMachineId(targetId);
    setAnchorTime(targetAnchor);
    setPastWindowMinutes(60);
    setDraftPastWindowMinutes(60);
    setFutureWindowMinutes(30);
    setDraftFutureWindowMinutes(30);
    
    // Clear stale data immediately to prevent "ghost" machine displays
    setControlRoomData(null);
    setCurrentView('dashboard');
    
    // Senior Note: We do NOT call fetchControlRoom manually here anymore.
    // The state changes to machineId and anchorTime will trigger the 
    // useEffect at line 162 automatically, preventing race conditions.
  }, [fleet]);

  useEffect(() => {
    setLimitOverrides({});
    setSelectedSensor(null);
  }, [machineId, pastWindowMinutes, futureWindowMinutes]);

  const safeLimits = controlRoomData?.safe_limits || {};
  const effectiveLimits = useMemo(
    () => mergeLimits(safeLimits, limitOverrides),
    [safeLimits, limitOverrides],
  );
  const timeline = controlRoomData?.timeline || [];
  const telemetryRows = Array.isArray(controlRoomData?.telemetry_grid)
    ? controlRoomData.telemetry_grid
    : [];

  const latestPastPoint = useMemo(() => {
    const past = timeline.filter((point) => !point.is_future);
    return past.length ? past[past.length - 1] : null;
  }, [timeline]);

  const isSensorFrozen = useMemo(() => {
    if (!latestPastPoint || !timeline.length) return false;
    const risk = toNumber(controlRoomData?.current_health?.risk_score) ?? 0;
    if (risk <= 0.8) return false;
    const trackedKeys = Object.keys(effectiveLimits);
    if (trackedKeys.length === 0) return false;
    const futurePoints = timeline.filter((p) => p.is_future);
    if (futurePoints.length === 0) return false;
    const firstFuturePoint = futurePoints[0];
    const pastSensors = latestPastPoint.sensors || {};
    const futureSensors = firstFuturePoint.sensors || {};
    return trackedKeys.every((k) => {
      const pastVal = toNumber(pastSensors[k]);
      const futureVal = toNumber(futureSensors[k]);
      return pastVal === futureVal;
    });
  }, [latestPastPoint, timeline, effectiveLimits, controlRoomData]);

  const rootCauses = useMemo(() => {
    const topLevel = controlRoomData?.root_causes;
    if (Array.isArray(topLevel)) {
      return topLevel
        .map((entry) => ({
          cause: typeof entry?.cause === "string" ? entry.cause : null,
          impact: toNumber(entry?.impact),
          risk_increasing: toNumber(entry?.risk_increasing),
          risk_decreasing: toNumber(entry?.risk_decreasing),
          top_parameters: Array.isArray(entry?.top_parameters) ? entry.top_parameters : [],
        }))
        .filter((entry) => Boolean(entry.cause))
        .slice(0, 3);
    }
    const fallback = controlRoomData?.current_health?.root_causes;
    if (Array.isArray(fallback)) {
      return fallback
        .filter((item) => typeof item === "string")
        .map((cause) => ({ cause, impact: null, top_parameters: [] }))
        .slice(0, 3);
    }
    return [];
  }, [controlRoomData]);

  const displayTimeline = useMemo(() => {
    if (!timeline.length) return [];
    return timeline.map((point) => ({
      ...point,
      risk_score: Number((toNumber(point.risk_score) ?? 0).toFixed(4)),
    }));
  }, [timeline]);

  const summaryStats = useMemo(() => {
    const base = controlRoomData?.summary_stats || {
      past_scrap_detected: 0,
      future_scrap_predicted: 0,
    };
    const timelineFutureCount = displayTimeline.filter(
      (point) => point.is_future && (toNumber(point.risk_score) ?? 0) > FUTURE_RISK_THRESHOLD,
    ).length;
    return {
      past_scrap_detected: base.past_scrap_detected ?? 0,
      future_scrap_predicted: Math.max(base.future_scrap_predicted ?? 0, timelineFutureCount),
      past_window_minutes: pastWindowMinutes,
      future_window_minutes: futureWindowMinutes,
    };
  }, [controlRoomData, displayTimeline, pastWindowMinutes, futureWindowMinutes]);

  const currentHealth = useMemo(() => {
    const apiHealth = controlRoomData?.current_health || {};
    const risk = toNumber(apiHealth.risk_score) ?? 0;
    const status = mapStatus(apiHealth.status, risk);
    return { status, risk_score: risk };
  }, [controlRoomData]);

  const telemetrySummary = useMemo(() => {
    return telemetryRows.reduce(
      (acc, row) => {
        const status = mapStatus(row?.status);
        if (status === UI_STATUS.CRITICAL) acc.critical += 1;
        else if (status === UI_STATUS.WARNING) acc.warning += 1;
        else acc.normal += 1;
        return acc;
      },
      { warning: 0, critical: 0, normal: 0 },
    );
  }, [telemetryRows]);

  const latestTimestampLabel = useMemo(() => {
    const latestPoint = latestPastPoint;
    if (!latestPoint?.timestamp) return "Searching...";
    // Senior Pro Fix: Use .utc() to ensure "1 Means 1" timeline integrity in the Header.
    const parsed = dayjs(latestPoint.timestamp).utc();
    if (!parsed.isValid()) return "Live sync";
    
    // Maintain absolute formatting for "Time Machine" historic context
    return parsed.format("HH:mm");
  }, [latestPastPoint, backgroundLoading]);

  const dashboardCards = useMemo(() => {
    const activeAlerts = telemetrySummary.warning + telemetrySummary.critical;
    const healthPercent = `${(currentHealth.risk_score * 100).toFixed(1)}%`;
    const alertTone =
      telemetrySummary.critical > 0
        ? "text-red-600"
        : telemetrySummary.warning > 0
          ? "text-amber-600"
          : "text-emerald-600";

    return [
      {
        title: "Current Risk",
        value: healthPercent,
        note: currentHealth.status,
        icon: Activity,
        iconClass: "bg-brand-50 text-brand-600",
        valueClass:
          currentHealth.risk_score >= 0.8
            ? "text-red-600"
            : currentHealth.risk_score >= 0.6
              ? "text-orange-600"
              : "text-emerald-600",
      },
      {
        title: "Active Alerts",
        value: activeAlerts,
        note: `${telemetrySummary.warning} warning${telemetrySummary.warning !== 1 ? 's' : ''} · ${telemetrySummary.critical} critical`,
        icon: AlertTriangle,
        iconClass: "bg-amber-50 text-amber-600",
        valueClass: alertTone,
        onClick: () => gridRef.current?.scrollIntoView({ behavior: "smooth" }),
        clickable: true,
      },
      {
        title: "Root Causes",
        value: rootCauses.length,
        note: rootCauses.length ? "Ranked drivers available" : "No active causes",
        icon: BarChart3,
        iconClass: "bg-emerald-50 text-emerald-600",
        valueClass: rootCauses.length ? "text-slate-800" : "text-emerald-600",
      },
      {
        title: "Total Scrap",
        value: summaryStats.past_scrap_detected,
        note: "Cumulative shots",
        icon: ShieldAlert,
        iconClass: "bg-red-50 text-red-600",
        valueClass: summaryStats.past_scrap_detected > 0 ? "text-red-600" : "text-emerald-600",
      },
      {
        title: "Live Sync",
        value: latestTimestampLabel,
        note: backgroundLoading ? "Refreshing..." : "Auto-refresh: 15s",
        icon: Clock3,
        iconClass: "bg-slate-100 text-slate-600",
        valueClass: "text-slate-800 text-2xl",
      },
    ];
  }, [
    currentHealth.risk_score,
    currentHealth.status,
    latestTimestampLabel,
    rootCauses.length,
    telemetrySummary.critical,
    telemetrySummary.warning,
    summaryStats.past_scrap_detected,
    backgroundLoading,
  ]);

  const telemetryStatuses = useMemo(() => {
    const statuses = {};
    telemetryRows.forEach((row) => {
      const sensor = row?.sensor;
      if (!sensor) return;
      const status = mapStatus(row?.status);
      if (status === UI_STATUS.CRITICAL) statuses[sensor] = "critical";
      else if (status === UI_STATUS.WARNING) statuses[sensor] = "warning";
      else statuses[sensor] = "good";
    });
    return statuses;
  }, [telemetryRows]);

  const firstAbnormalSensor = useMemo(() => {
    const critical = Object.keys(telemetryStatuses).find((s) => telemetryStatuses[s] === "critical");
    if (critical) return critical;
    const warning = Object.keys(telemetryStatuses).find((s) => telemetryStatuses[s] === "warning");
    return warning || null;
  }, [telemetryStatuses]);

  useEffect(() => {
    if (selectedSensor && telemetryStatuses[selectedSensor] === "good") {
      setSelectedSensor(firstAbnormalSensor);
    }
  }, [telemetryStatuses, selectedSensor, firstAbnormalSensor]);

  useEffect(() => {
    if (!selectedSensor && firstAbnormalSensor) {
      setSelectedSensor(firstAbnormalSensor);
    }
  }, [selectedSensor, firstAbnormalSensor]);

  const handleSelectSensor = useCallback((sensor) => {
    setSelectedSensor(sensor);
    setIsDrawerOpen(true);
  }, []);

  const selectedSensorData = useMemo(() => {
    if (!selectedSensor) return null;
    return telemetryRows.find((r) => r.sensor === selectedSensor) || null;
  }, [selectedSensor, telemetryRows]);

  const loadingMachineInfo = {
    display_id: machineId,
    machine_number: String(machineId).match(/\d+/)?.[0] || machineId,
  };

  if (!isAuthenticated) {
    return <Login onLoginSuccess={() => setIsAuthenticated(true)} />;
  }

  if (loading && !controlRoomData) {
    return (
      <div className="relative flex min-h-screen overflow-hidden bg-slate-50 font-sans">
        <Sidebar
          activeMachine={machineId}
          onSelectMachine={setMachineId}
          activeView={currentView}
          onViewChange={setCurrentView}
        />
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6 lg:p-8">
          <Header
            machineId={machineId}
            hasPendingWindowChanges={false}
            draftPastWindowMinutes={draftPastWindowMinutes}
            draftFutureWindowMinutes={draftFutureWindowMinutes}
            pastWindowMinutes={pastWindowMinutes}
            futureWindowMinutes={futureWindowMinutes}
            onMachineChange={setMachineId}
            onPastWindowChange={setDraftPastWindowMinutes}
            onFutureWindowChange={setDraftFutureWindowMinutes}
            onApplyWindows={() => {}}
            healthStatus="NORMAL"
            isSensorFrozen={false}
            machineInfo={loadingMachineInfo}
            lastUpdatedLabel="Checking signals..."
          />
          <div className="mx-auto max-w-[1400px] flex flex-col gap-6 space-y-6">
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
              {[1, 2, 3, 4].map((i) => <CardSkeleton key={i} />)}
            </div>
            <HealthMonitorSkeleton />
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2"><TelemetryGridSkeleton /></div>
              <div className="lg:col-span-1"><CardSkeleton /></div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-screen bg-slate-50/50">
      {/* Atmospheric Backgrounds */}
      <div className="pointer-events-none fixed inset-0 z-0 bg-transparent transition-colors duration-1000" />
      <div className="pointer-events-none fixed -top-32 right-[-10rem] h-96 w-96 rounded-full bg-brand-500/10 blur-[100px] transition-transform duration-1000" />
      <div className="pointer-events-none fixed bottom-[-10rem] left-[-10rem] h-96 w-96 rounded-full bg-emerald-500/10 blur-[100px] transition-transform duration-1000" />
      <div className="pointer-events-none fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[800px] w-[800px] rounded-full bg-[radial-gradient(circle,_rgba(255,255,255,0.8)_0%,_rgba(255,255,255,0)_60%)] mix-blend-overlay opacity-30" />

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-sm lg:hidden transition-opacity duration-300"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Sidebar Wrapper */}
      <div className={`fixed inset-y-0 left-0 z-50 lg:relative lg:z-30 transition-transform duration-300 transform ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0`}>
        <Sidebar
          activeMachine={machineId}
          onSelectMachine={(id) => {
            setMachineId(id);
            setAnchorTime(null);
            setSidebarOpen(false);
          }}
          activeView={currentView}
          onViewChange={(view) => {
            setCurrentView(view);
            setSidebarOpen(false);
          }}
        />
      </div>

      <main 
        className="relative z-10 flex-1 min-h-screen p-4 md:p-6 lg:p-8 pt-0 lg:pt-0"
        style={{ "--container-padding": "env(safe-area-inset-left, 2rem)" }}
      >
        <div className="mx-auto max-w-[1440px] flex flex-col gap-4 stagger-children">
          {/* Refined Sticky Header */}
          {currentView === "dashboard" && (
            <div className="sticky-header-float">
              <Header
                machineId={machineId}
                hasPendingWindowChanges={hasPendingWindowChanges}
                draftPastWindowMinutes={draftPastWindowMinutes}
                draftFutureWindowMinutes={draftFutureWindowMinutes}
                pastWindowMinutes={pastWindowMinutes}
                futureWindowMinutes={futureWindowMinutes}
                onMachineChange={setMachineId}
                onPastWindowChange={setDraftPastWindowMinutes}
                onFutureWindowChange={setDraftFutureWindowMinutes}
                onApplyWindows={handleApplyWindows}
                onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
                healthStatus={currentHealth.status}
                isSensorFrozen={isSensorFrozen}
                machineInfo={controlRoomData?.machine_info || {}}
                lastUpdatedLabel={latestTimestampLabel}
                anchorTime={anchorTime}
                onAnchorTimeChange={setAnchorTime}
              />
            </div>
          )}

          {backendConnecting && (
            <div className="rounded-2xl border border-amber-200/80 bg-amber-50/80 px-6 py-4 text-sm text-amber-700 font-medium animate-fade-in shadow-sm backdrop-blur-md flex items-center gap-3">
              <div className="w-4 h-4 rounded-full border-2 border-amber-400 border-t-transparent animate-spin shrink-0" />
              Connecting to backend — live stream will resume automatically…
            </div>
          )}
          {error && !backendConnecting && (
            <div className="rounded-2xl border border-red-200/80 bg-red-50/80 px-6 py-4 text-sm text-red-600 font-medium animate-fade-in shadow-sm backdrop-blur-md flex items-center gap-3">
              <AlertTriangle size={18} className="text-red-500 shrink-0" />
              {error}
            </div>
          )}

          <React.Suspense fallback={<div className="p-12 text-center text-slate-400 font-medium flex flex-col items-center gap-4"><div className="w-8 h-8 rounded-full border-4 border-slate-200 border-t-brand-500 animate-spin" />Scaling infrastructure...</div>}>
            {currentView === "ingestion" ? (
              <IngestionHub 
                onFinish={() => setCurrentView("dashboard")} 
                onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
              />
            ) : currentView === "audit" ? (
              <AuditHub onReplayAnomaly={handleReplayAnomaly} />
            ) : currentView === "analytics" ? (
              <AnalyticsHub onViewMachine={(id) => {
                setMachineId(id);
                setAnchorTime(null);
                setCurrentView('dashboard');
                controlRoomRequestRef.current += 1;
                fetchControlRoom({ showFullLoading: true, requestId: controlRoomRequestRef.current, targetMachineId: id, targetAnchorTime: null });
              }} />
            ) : (
              <AnimatePresence mode="wait">
              <motion.div 
                key="dashboard"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4, staggerChildren: 0.1 }}
                className="flex flex-col gap-6"
              >
                <section id="dashboard" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5 scroll-mt-28">
                  {dashboardCards.map((card, idx) => {
                    const Icon = card.icon;
                    return (
                      <motion.div
                        key={card.title}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: idx * 0.05 }}
                        className={`glass-card group relative overflow-hidden flex flex-col justify-between p-5 transition-all duration-300 ${
                          card.clickable
                            ? 'cursor-pointer hover:shadow-xl hover:-translate-y-1 hover:border-brand-200/60 active:scale-95'
                            : 'hover:shadow-lg hover:-translate-y-0.5 hover:border-slate-300/60'
                        }`}
                        onClick={card.onClick}
                      >
                        <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-slate-100 opacity-0 group-hover:opacity-50 blur-2xl transition-opacity duration-300" />
                        <div className="flex items-start justify-between gap-4 mb-4 relative z-10">
                          <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl shadow-sm border border-white/60 transition-transform group-hover:scale-110 ${card.iconClass}`}>
                            <Icon size={20} />
                          </div>
                          <div className="text-right">
                            <p className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-400/80 mb-1">{card.title}</p>
                            <div className={`text-3xl font-black tracking-tight leading-none drop-shadow-sm ${card.valueClass || "text-slate-800"}`}>
                              {card.value}
                            </div>
                          </div>
                        </div>
                        <div className="relative z-10 border-t border-slate-100/50 pt-3">
                          <p className="text-[11px] font-bold text-slate-500/80 tracking-wide line-clamp-1">{card.note}</p>
                        </div>
                      </motion.div>
                    );
                  })}
                </section>

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                  <motion.div 
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="lg:col-span-2 scroll-mt-28" 
                    id="health-monitor"
                  >
                    <HealthMonitor timeline={displayTimeline} riskScore={currentHealth.risk_score} auditAreas={controlRoomData?.audit_reference_areas || []} />
                  </motion.div>
                  
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="lg:col-span-1 scroll-mt-28" 
                    id="root-cause"
                  >
                    <RootCause rootCauses={rootCauses} onSelectSensor={handleSelectSensor} />
                  </motion.div>
                </div>

                <motion.div 
                   initial={{ opacity: 0, y: 20 }}
                   animate={{ opacity: 1, y: 0 }}
                   className="grid grid-cols-1 gap-6 lg:grid-cols-4"
                >
                  <div ref={gridRef} className="lg:col-span-3 min-h-[600px] flex-1">
                    <TelemetryGrid
                      telemetryRows={telemetryRows}
                      selectedSensor={selectedSensor}
                      onSelectSensor={handleSelectSensor}
                    />
                  </div>
                  
                  <div id="prediction-stats" className="lg:col-span-1 scroll-mt-28">
                    <PredictionStats summaryStats={summaryStats} timeline={displayTimeline} />
                  </div>
                </motion.div>
              </motion.div>
              </AnimatePresence>
            )}

            <SensorDrawer
              isOpen={isDrawerOpen}
              onClose={() => setIsDrawerOpen(false)}
              machineId={machineId}
              sensor={selectedSensor}
              sensorData={selectedSensorData}
            />
          </React.Suspense>
        </div>
      </main>
    </div>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, PieChart, Pie, Cell 
} from 'recharts';
import { 
  Activity, Thermometer, Gauge, AlertTriangle, CheckCircle, Zap, ShieldAlert, Bug, ChevronDown 
} from 'lucide-react';

// --- CONFIGURATION ---
const API_BASE = "http://127.0.0.1:8080"; 

// --- COMPONENTS ---

const StatusGauge = ({ riskLevel }) => {
  const data = [
    { name: 'Risk', value: riskLevel * 100 },
    { name: 'Safe', value: 100 - (riskLevel * 100) },
  ];
  const getColor = (risk) => {
    if (risk < 0.3) return "#10b981"; 
    if (risk < 0.7) return "#f59e0b"; 
    return "#ef4444"; 
  };
  return (
    <div className="relative h-48 flex flex-col items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data} cx="50%" cy="70%" startAngle={180} endAngle={0}
            innerRadius={60} outerRadius={80} paddingAngle={0} dataKey="value" stroke="none"
          >
            <Cell fill={getColor(riskLevel)} />
            <Cell fill="#1f2937" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute bottom-6 text-center">
        <div className="text-3xl font-bold text-white">{(riskLevel * 100).toFixed(1)}%</div>
        <div className="text-xs text-gray-400 uppercase tracking-wider">Scrap Probability</div>
      </div>
    </div>
  );
};

const TelemetryCard = ({ icon: Icon, label, value, unit, color = "text-blue-400" }) => (
  <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 flex items-center justify-between">
    <div>
      <div className="text-gray-500 text-xs uppercase mb-1">{label}</div>
      <div className="text-xl font-mono text-white font-bold">{value} <span className="text-sm text-gray-500">{unit}</span></div>
    </div>
    <div className={`p-2 rounded-full bg-gray-700/50 ${color}`}>
      <Icon size={20} />
    </div>
  </div>
);

const SystemHealthChart = ({ data }) => {
  if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-500">Loading Trend...</div>;

  const firstRow = data[0];
  const keys = Object.keys(firstRow);
  const dataKey = keys.find(k => k !== 'time' && k !== 'timestamp' && k !== 'machine_id') || "Injection_pressure";

  return (
    <div className="h-full w-full relative">
      <h3 className="text-white font-semibold flex items-center gap-2 mb-2">
        <Activity size={16} className="text-green-400" /> 
        System Health Monitor: {dataKey.replace(/_/g, ' ')}
      </h3>
      
      <ResponsiveContainer width="100%" height="90%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorGreen" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
          <XAxis 
            dataKey="time" 
            stroke="#9ca3af" 
            tick={{fontSize: 10}} 
            interval={Math.floor(data.length / 5)} 
          />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff' }}
            itemStyle={{ color: '#10b981' }}
          />
          <ReferenceLine x="Now" stroke="white" strokeDasharray="3 3" label={{ value: "NOW", fill: 'white', fontSize: 10 }} />
          <Area 
            type="monotone" 
            dataKey={dataKey} 
            stroke="#10b981" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorGreen)" 
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

const RootCauseChart = ({ parameter, data, limits }) => {
  if (!data || data.length === 0) return null;
  const dataKey = Object.keys(data[0]).find(k => k !== 'time' && k !== 'timestamp' && k !== 'machine_id') || parameter;

  return (
    <div className="h-full w-full bg-red-900/10 border border-red-500/30 rounded-lg p-4 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-red-500 animate-pulse"></div>
      <h3 className="text-red-400 font-bold flex items-center gap-2 mb-2 uppercase text-sm tracking-wide">
        <AlertTriangle size={16} /> 
        Anomaly Detected: {parameter}
      </h3>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#451a1a" />
          <XAxis dataKey="time" stroke="#ef4444" tick={{fontSize: 10}} interval={Math.floor(data.length / 5)} />
          <YAxis stroke="#ef4444" domain={['auto', 'auto']} fontSize={10} />
          <Tooltip contentStyle={{ backgroundColor: '#450a0a', borderColor: '#ef4444', color: '#fff' }} />
          {limits?.max && <ReferenceLine y={limits.max} stroke="#ef4444" strokeDasharray="5 5" label="MAX" />}
          {limits?.min && <ReferenceLine y={limits.min} stroke="#ef4444" strokeDasharray="5 5" label="MIN" />}
          <Line type="monotone" dataKey={dataKey} stroke="#ef4444" strokeWidth={3} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// --- MAIN APP ---
function App() {
  // NEW: State for Machine Selection
  const [machineId, setMachineId] = useState("M-231");
  const [status, setStatus] = useState(null);
  const [mainTrend, setMainTrend] = useState([]);
  const [failTrend, setFailTrend] = useState(null);
  const [failParam, setFailParam] = useState(null);
  const [failLimits, setFailLimits] = useState(null);
  const [debugError, setDebugError] = useState(null);

  useEffect(() => {
    // Reset data when machine changes so graphs don't mix
    setStatus(null); 
    setMainTrend([]);

    const fetchData = async () => {
      try {
        const statusRes = await axios.get(`${API_BASE}/api/status/${machineId}`);
        setStatus(statusRes.data);

        const trendRes = await axios.get(`${API_BASE}/api/trend/${machineId}/Injection_pressure`);
        
        const formattedTrend = trendRes.data.data.map(d => ({
          ...d,
          time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }));
        setMainTrend(formattedTrend);
        setDebugError(null);

        if (statusRes.data.alert_level !== "LOW" && statusRes.data.violations.length > 0) {
            const badParam = statusRes.data.violations[0].parameter;
            setFailParam(badParam);
            const failRes = await axios.get(`${API_BASE}/api/trend/${machineId}/${badParam}`);
            const formattedFail = failRes.data.data.map(d => ({
                ...d,
                time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }));
            setFailTrend(formattedFail);
            setFailLimits(failRes.data.limits);
        } else {
            setFailTrend(null);
        }

      } catch (err) {
        console.error("API Error:", err);
        setDebugError(err.message);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [machineId]); // Re-run when machineId changes

  // Loading Screen
  if (!status) return (
    <div className="bg-gray-900 min-h-screen text-white flex flex-col items-center justify-center gap-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <div>Connecting to {machineId}...</div>
        {debugError && <div className="text-red-400 text-sm border border-red-800 p-2 rounded">Debug: {debugError}</div>}
    </div>
  );

  const isCritical = status.alert_level !== "LOW";

  return (
    <div className="bg-gray-900 min-h-screen p-6 font-sans text-gray-100 flex flex-col">
      
      {/* HEADER */}
      <header className="mb-6 flex justify-between items-center border-b border-gray-800 pb-4">
        <div>
           <h1 className="text-2xl font-bold text-white flex items-center gap-3">
             <Zap className="text-yellow-400" fill="currentColor" />
             Predictive Maintenance Console
           </h1>
           <p className="text-gray-500 text-xs mt-1 ml-9">LIVE MONITORING: {machineId}</p>
        </div>
        
        {/* NEW: Machine Selector Dropdown */}
        <div className="flex items-center gap-4">
          <div className="relative">
             <select 
               value={machineId}
               onChange={(e) => setMachineId(e.target.value)}
               className="bg-gray-800 text-white font-mono border border-gray-600 rounded px-4 py-2 pr-8 appearance-none focus:outline-none focus:border-blue-500 cursor-pointer hover:bg-gray-700 transition"
             >
               <option value="M-231">M-231 (Line 1)</option>
               <option value="M-471">M-471 (Line 2)</option>
               <option value="M-607">M-607 (Line 3)</option>
               <option value="M-612">M-612 (Line 4)</option>
             </select>
             <ChevronDown className="absolute right-2 top-3 text-gray-400 pointer-events-none" size={16} />
          </div>

          <div className={`px-4 py-2 rounded font-bold ${isCritical ? 'bg-red-500/20 text-red-400 border border-red-500' : 'bg-green-500/20 text-green-400 border border-green-500'}`}>
              STATUS: {status.alert_level}
          </div>
        </div>
      </header>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-grow">
        <div className="lg:col-span-4 flex flex-col gap-6">
            <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 relative overflow-hidden">
                <h2 className="text-gray-400 text-sm font-bold uppercase tracking-wider mb-2">Overall Machine Status</h2>
                <StatusGauge riskLevel={status.ml_risk_probability} />
                <div className="mt-4 text-center p-3 rounded bg-gray-900 border border-gray-700">
                    <div className="text-gray-400 text-xs">CURRENT STATE</div>
                    <div className={`text-lg font-bold ${isCritical ? 'text-red-400' : 'text-green-400'}`}>
                        {isCritical ? "PREDICTIVE FAILURE DETECTED" : "OPTIMAL OPERATION"}
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 gap-4">
                <TelemetryCard icon={Thermometer} label="Cyl Temp Zone 3" value="210.4" unit="°C" color="text-orange-400" />
                <TelemetryCard icon={Gauge} label="Injection Pressure" value="1420" unit="bar" color="text-blue-400" />
            </div>
        </div>

        <div className="lg:col-span-8 flex flex-col gap-6">
            <div className={`bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 flex-1 min-h-[300px] transition-all duration-500 ${isCritical ? 'h-[45%]' : 'h-full'}`}>
                <SystemHealthChart data={mainTrend} />
            </div>

            {isCritical && failTrend && (
                <div className="flex-1 min-h-[300px] animate-in fade-in slide-in-from-bottom-10 duration-700">
                    <RootCauseChart parameter={failParam} data={failTrend} limits={failLimits} />
                </div>
            )}
        </div>
      </div>
    </div>
  );
}

export default App;
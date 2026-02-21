import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const TrendMonitor = ({ machineId, parameter }) => {
  const [data, setData] = useState([]);
  const [limits, setLimits] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8080/api/trend/${machineId}/${parameter}`);
        
        // Format timestamp for display (HH:MM)
        const formattedData = response.data.data.map(d => ({
          ...d,
          time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        }));
        
        setData(formattedData);
        setLimits(response.data.limits);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching trend data", error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Live refresh
    return () => clearInterval(interval);
  }, [machineId, parameter]);

  if (loading) return <div className="text-sm text-gray-400">Loading Trend...</div>;

  return (
    <div className="bg-industrial-card p-4 rounded-lg border border-gray-700 shadow-lg h-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-gray-200">📉 Predictive Trend: {parameter}</h3>
        <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
          Last 30m History + Next 60m Forecast
        </span>
      </div>
      
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" stroke="#888" fontSize={12} interval="preserveStartEnd" />
            <YAxis stroke="#888" fontSize={12} domain={['auto', 'auto']} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#333', border: 'none' }}
              labelStyle={{ color: '#fff' }}
            />
            
            {/* Safe Limits (Red Lines) */}
            {limits.max && (
              <ReferenceLine y={limits.max} stroke="#FF5252" strokeDasharray="5 5" label={{ value: 'MAX', fill: '#FF5252', fontSize: 10 }} />
            )}
            {limits.min && (
              <ReferenceLine y={limits.min} stroke="#FF5252" strokeDasharray="5 5" label={{ value: 'MIN', fill: '#FF5252', fontSize: 10 }} />
            )}

            {/* Historical Data (Solid Blue) */}
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              connectNulls={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TrendMonitor;
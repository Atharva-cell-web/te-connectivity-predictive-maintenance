import React, { useState, useEffect } from "react";
import {
  CheckCircle2, XCircle, ShieldCheck, Target, BarChart,
  AlertTriangle, Search, Plus, Trash2, Edit2, Save, X,
  ChevronRight, Calendar, Clock, MessageSquare, Monitor, Activity
} from "lucide-react";
import apiClient from "../utils/apiClient";

export default function AuditHub({ onReplayAnomaly }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);
  const [saving, setSaving] = useState(false);

  // Explicitly generate DD-MM-YYYY
  const today = new Date();
  const d = String(today.getDate()).padStart(2, '0');
  const m = String(today.getMonth() + 1).padStart(2, '0');
  const y = today.getFullYear();
  const defaultDate = `${d}-${m}-${y}`;

  // Form state
  const [formState, setFormState] = useState({
    machine: "",
    date: defaultDate,
    id: "",
    start: "",
    end: "",
    comment: "",
    ignore: false
  });

  const fetchAudit = async () => {
    try {
      setLoading(true);
      const res = await apiClient.get("/api/audit/validation");
      setData(res.data);
    } catch (err) {
      console.error("Audit fetch failed:", err);
      setError("Failed to load audit results. Ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAudit();
  }, []);

  const handleSave = async (e) => {
    if (e) e.preventDefault();

    // Production Validation: Ensure mandatory fields are populated
    if (!formState.machine || !formState.date || !formState.id || !formState.start) {
      alert("⚠️ VERIFICATION REQUIRED:\nPlease ensure Machine Identifier, Audit Date, Case Reference, and Start Interval are fully filled out before committing.");
      return;
    }

    // Automatically enforce 30 minute machine learning forecast window
    let computedEnd = formState.end;
    try {
      const [hours, mins] = formState.start.split(':').map(Number);
      const totalMins = hours * 60 + mins + 30;
      const endHours = Math.floor(totalMins / 60) % 24;
      const endMins = totalMins % 60;
      computedEnd = `${String(endHours).padStart(2, '0')}:${String(endMins).padStart(2, '0')}`;
    } catch (err) {
      computedEnd = formState.start;
    }

    const payload = { ...formState, end: computedEnd };

    try {
      setSaving(true);
      if (editingIndex !== null) {
        await apiClient.put(`/api/audit/case/${editingIndex}`, payload);
      } else {
        await apiClient.post("/api/audit/case", payload);
      }
      fetchAudit();
      resetForm();
    } catch (err) {
      alert("Failed to save audit record: " + (err.response?.data?.detail || err.message));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (index) => {
    if (!window.confirm("Are you sure you want to remove this ground-truth record?")) return;
    try {
      await apiClient.delete(`/api/audit/case/${index}`);
      fetchAudit();
    } catch (err) {
      alert("Failed to delete record: " + (err.response?.data?.detail || err.message));
    }
  };

  const startEdit = (index, record) => {
    setEditingIndex(index);
    setFormState({
      machine: record.machine || "",
      date: record.date || "",
      id: record.id || "",
      start: record.start || "",
      end: record.end || "",
      comment: record.comment || "",
      ignore: record.status === "IGNORE"
    });
    setShowAddForm(true);
  };

  const resetForm = () => {
    setFormState({
      machine: "",
      date: new Date().toISOString().split("T")[0],
      id: "",
      start: "",
      end: "",
      comment: "",
      ignore: false
    });
    setEditingIndex(null);
    setShowAddForm(false);
  };

  if (loading && !data) {
    return (
      <div className="flex h-96 items-center justify-center animate-fade-in">
        <div className="flex flex-col items-center gap-4">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-brand-500 border-t-transparent shadow-glow" />
          <p className="text-[10px] font-black text-slate-400 animate-pulse uppercase tracking-[0.2em]">Synchronizing Ground-Truth Records...</p>
        </div>
      </div>
    );
  }

  const results = data?.results || [];

  return (
    <div id="audit-hub" className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 pb-20">
      {/* Management Action Bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center shadow-2xl">
            <ShieldCheck size={20} className="text-brand-400" />
          </div>
          <div>
            <h1 className="text-lg font-black text-slate-800 tracking-tight uppercase">Audit Management</h1>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Ground-Truth Registry Control</p>
          </div>
        </div>

        <button
          onClick={() => { resetForm(); setShowAddForm(true); }}
          className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white rounded-xl text-[11px] font-black uppercase tracking-widest hover:bg-slate-800 transition-all shadow-xl shadow-slate-900/10 active:scale-95"
        >
          <Plus size={16} /> Add Scrap Case
        </button>
      </div>

      {/* Simplified Accruacy Display for Context */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 flex flex-col justify-center border-l-4 border-l-brand-500">
          <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 flex items-center gap-2 mb-2">
            <Target size={12} className="text-brand-500" /> Model Accuracy
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black text-slate-800">{data?.accuracy || 0}%</span>
            <span className="text-[9px] font-bold text-slate-400 uppercase">Verified</span>
          </div>
        </div>
        <div className="glass-card p-6 flex flex-col justify-center border-l-4 border-l-emerald-500">
          <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 flex items-center gap-2 mb-2">
            <CheckCircle2 size={12} className="text-emerald-500" /> Event Synchronization
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black text-slate-800">{data?.matches || 0}</span>
            <span className="text-[9px] font-bold text-slate-400 uppercase">of {data?.total_cases || 0} Matched</span>
          </div>
        </div>
        <div className="glass-card p-6 bg-slate-900 text-white border-0">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[9px] font-black uppercase tracking-widest text-brand-400">Total Scenarios</p>
            <BarChart size={14} className="text-brand-400" />
          </div>
          <span className="text-3xl font-black">{results.length}</span>
          <p className="mt-1 text-[9px] font-bold text-slate-400 uppercase">Active audit registry entries</p>
        </div>
      </div>

      {/* Add/Edit Form Sidebar (Modal Alternative) */}
      {showAddForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-end bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="w-full max-md:max-w-full max-w-md h-full bg-white shadow-2xl p-8 flex flex-col animate-in slide-in-from-right duration-500 overflow-y-auto">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-brand-50 flex items-center justify-center">
                  {editingIndex !== null ? <Edit2 size={20} className="text-brand-600" /> : <Plus size={20} className="text-brand-600" />}
                </div>
                <div>
                  <h2 className="text-sm font-black text-slate-800 uppercase tracking-widest">
                    {editingIndex !== null ? "Edit Case" : "Register Case"}
                  </h2>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">Enter scrap ground-truth details</p>
                </div>
              </div>
              <button onClick={resetForm} className="p-2 hover:bg-slate-50 rounded-lg transition-colors">
                <X size={20} className="text-slate-400" />
              </button>
            </div>

            <form onSubmit={handleSave} className="flex-1 space-y-6">
              <div className="space-y-4">
                <div className="group">
                  <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">Machine Identifier</label>
                  <div className="relative">
                    <Monitor size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-300" />
                    <input
                      className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-100 rounded-xl text-xs font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all"
                      placeholder="e.g. M356 *"
                      value={formState.machine}
                      onChange={e => setFormState({ ...formState, machine: e.target.value.toUpperCase() })}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">Audit Date (dd-mm-yyyy)</label>
                    <div className="relative">
                      <Calendar size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-300" />
                      <input
                        type="text"
                        className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-100 rounded-xl text-xs font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all"
                        placeholder="e.g. 14-02-2026 *"
                        value={formState.date}
                        onChange={e => setFormState({ ...formState, date: e.target.value })}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">Case Reference</label>
                    <input
                      className="w-full px-4 py-3 bg-slate-50 border border-slate-100 rounded-xl text-xs font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all"
                      placeholder="e.g. Scrap Case #1 *"
                      value={formState.id}
                      onChange={e => setFormState({ ...formState, id: e.target.value })}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">Start Interval</label>
                    <div className="relative">
                      <Clock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-300" />
                      <input
                        type="time"
                        className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-100 rounded-xl text-xs font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all"
                        value={formState.start}
                        onChange={e => setFormState({ ...formState, start: e.target.value })}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">End Interval</label>
                    <div className="relative">
                      <Clock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                      <input
                        type="text"
                        disabled
                        className="w-full pl-10 pr-4 py-3 bg-slate-100 border border-slate-200 rounded-xl text-xs font-bold text-slate-500 cursor-not-allowed"
                        value="+ 30M Forecast Window"
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1.5 block">Operational Comment</label>
                  <div className="relative">
                    <MessageSquare size={14} className="absolute left-3 top-4 text-slate-300" />
                    <textarea
                      rows={3}
                      className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-100 rounded-xl text-xs font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all resize-none"
                      placeholder="Machine stability observations..."
                      value={formState.comment}
                      onChange={e => setFormState({ ...formState, comment: e.target.value })}
                    />
                  </div>
                </div>

                <div className="flex items-center gap-2 p-4 bg-slate-50 rounded-xl border border-dashed border-slate-200">
                  <input
                    type="checkbox"
                    id="ignore-check"
                    className="w-4 h-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                    checked={formState.ignore}
                    onChange={e => setFormState({ ...formState, ignore: e.target.checked })}
                  />
                  <label htmlFor="ignore-check" className="text-[10px] font-black text-slate-500 uppercase tracking-widest cursor-pointer">
                    Exclude from accuracy metrics (Ignore)
                  </label>
                </div>
              </div>

              <div className="pt-4 space-y-3 pb-8">
                <button
                  type="submit"
                  disabled={saving}
                  className={`w-full py-4 bg-slate-900 text-white rounded-xl text-xs font-black uppercase tracking-[0.25em] shadow-xl shadow-slate-900/10 hover:bg-black transition-all active:scale-95 flex items-center justify-center gap-2 ${saving ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  {saving ? (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/50 border-t-white" />
                  ) : (
                    <Save size={16} />
                  )}
                  {saving ? "Processing..." : (editingIndex !== null ? "Update Registry" : "Commit to Audit")}
                </button>
                <button
                  type="button"
                  onClick={resetForm}
                  className="w-full py-4 bg-white text-slate-400 border border-slate-100 rounded-xl text-[10px] font-black uppercase tracking-[0.25em] hover:bg-slate-50 transition-all"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Audit Detail Table */}
      <div className="glass-card overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/30">
          <h2 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-800 flex items-center gap-3">
            <BarChart size={18} className="text-brand-500" />
            Performance Validation Log
          </h2>
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5 text-[9px] font-black text-slate-400 uppercase tracking-widest">
              <div className="w-2 h-2 rounded-full bg-slate-900" /> Management Mode Active
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[1000px]">
            <thead>
              <tr className="bg-slate-50/50">
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Machine Number</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Date</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Scrap Case #</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Time Window</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Scrap predicted?</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Comment</th>
                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {results.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-6 py-20 text-center">
                    <div className="flex flex-col items-center gap-3 opacity-30">
                      <Search size={40} className="text-slate-400" />
                      <p className="text-xs font-black uppercase tracking-widest text-slate-500">No Ground-Truth Records Defined</p>
                    </div>
                  </td>
                </tr>
              )}
              {results.map((row) => {
                const isIgnore = row.status === "IGNORE";
                const currentIndex = row.index;
                return (
                  <tr key={`row-${currentIndex}`} className={`hover:bg-brand-50/20 transition-colors group ${isIgnore ? 'opacity-60 bg-slate-50/30' : ''}`}>
                    <td className="px-6 py-5">
                      <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl ${isIgnore ? 'bg-slate-200 text-slate-500' : 'bg-slate-900 text-white shadow-lg shadow-slate-900/10'} text-[11px] font-black transition-transform group-hover:scale-105`}>
                        {row.machine}
                      </span>
                    </td>
                    <td className="px-6 py-5 text-[11px] font-bold text-slate-600">{row.date}</td>
                    <td className="px-6 py-5 text-[11px] font-black text-slate-400 uppercase tracking-widest">{row.id}</td>
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] font-bold text-slate-800">{row.start}</span>
                        <ChevronRight size={12} className="text-slate-300" />
                        <span className="text-[11px] font-bold text-slate-800">{row.end}</span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      {isIgnore ? (
                        <span className="text-[11px] font-black text-slate-300 uppercase tracking-widest italic flex items-center gap-1.5">
                          <XCircle size={12} /> Ignored
                        </span>
                      ) : (
                        <div className="flex flex-col gap-1">
                          <span className={`text-[13px] font-black uppercase tracking-wider ${row.predicted === 'YES' ? 'text-emerald-600' : 'text-red-500'}`}>
                            {row.predicted}
                          </span>
                          <span className="text-[9px] font-bold text-slate-400 opacity-60 uppercase tracking-tighter">
                            Risk: {(row.max_risk * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-5 max-w-xs">
                      <p className="text-[10px] font-medium leading-relaxed text-slate-500 italic line-clamp-2" title={row.comment}>
                        {row.comment || "—"}
                      </p>
                    </td>
                    <td className="px-6 py-5 text-right">
                      <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => {
                            try {
                              // Senior Pro Fix: Robust parsing for DD-MM-YYYY or YYYY-MM-DD
                              const parts = row.date.split(/[-/]/);
                              let y, m, d;
                              if (parts.length === 3) {
                                if (parts[2].length === 4) {
                                  [d, m, y] = parts;
                                } else {
                                  [y, m, d] = parts;
                                }
                                // Ensure 2-digit padding
                                m = m.padStart(2, '0');
                                d = d.padStart(2, '0');

                                const isoStr = `${y}-${m}-${d}T${row.start}:00Z`;
                                const startTs = new Date(isoStr).getTime();
                                onReplayAnomaly(row.machine, startTs);
                              }
                            } catch (e) {
                              console.error("Could not parse date for replay jump", e);
                            }
                          }}
                          className={`p-2 rounded-lg transition-colors ${!isIgnore && row.start !== "N/A" ? 'text-brand-600 hover:bg-brand-50' : 'text-slate-300 cursor-not-allowed'}`}
                          title="View on Dashboard"
                        >
                          <Activity size={16} />
                        </button>
                        <button
                          onClick={() => startEdit(currentIndex, row)}
                          className="p-2 hover:bg-indigo-50 rounded-lg text-indigo-500 transition-colors"
                          title="Edit Row"
                        >
                          <Edit2 size={16} />
                        </button>
                        <button
                          onClick={() => handleDelete(currentIndex)}
                          className="p-2 hover:bg-red-50 rounded-lg text-red-500 transition-colors"
                          title="Remove Row"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

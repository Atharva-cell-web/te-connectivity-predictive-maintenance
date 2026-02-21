import streamlit as st
import pandas as pd
import time
from pathlib import Path
import sys

# --- SETUP ---
sys.path.append(str(Path(__file__).parent / "backend"))
from run_realtime_check import run as run_check
from trend_plotter import plot_parameter_trend
from config_limits import PARAMETER_LABELS, SAFE_LIMITS

# --- PAGE CONFIG (Industrial Wide Mode) ---
st.set_page_config(
    page_title="IMM Production Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make metrics pop like a real HMI screen
st.markdown("""
<style>
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 5px solid #31333F;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Control Panel) ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    selected_machine = st.selectbox("Select Machine", ["M-231", "M-471", "M-607", "M-612"])
    refresh_rate = st.slider("Cycle Refresh (s)", 5, 60, 10)
    if st.button("🔄 System Reset / Refresh", type="primary"):
        st.rerun()
    st.caption(f"Connected to: {selected_machine}")

# --- MAIN EXECUTION ---
try:
    # 1. Run Backend Logic
    with st.spinner(f"Reading PLC/Sensor stream for {selected_machine}..."):
        decision = run_check(selected_machine)

    # Parse key data
    risk_score = decision.get('ml_risk_probability', 0.0)
    alert_level = decision.get('alert_level', 'LOW')
    timestamp = decision.get('timestamp', 'Unknown')
    violations = decision.get('violations', [])

    # =========================================================
    # SECTION A: Machine Overview (Top Strip)
    # Purpose: Instant situational awareness (≤ 3 seconds)
    # =========================================================
    st.markdown("### 🏭 Machine Status Overview")
    
    # Create 4 columns for the top strip
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Machine ID", selected_machine, "Active")
    
    with kpi2:
        st.metric("Last Update", str(timestamp).split(" ")[1][:8]) # Time only

    with kpi3:
        # Dynamic Risk Color Logic
        if alert_level == "LOW":
            st.success("✅ STATUS: NORMAL")
        elif alert_level == "MEDIUM":
            st.warning("⚠️ STATUS: WARNING")
        else:
            st.error("🔴 STATUS: CRITICAL")

    with kpi4:
        # Numeric Risk display
        st.metric("ML Risk Score", f"{risk_score:.1%}", delta_color="inverse")

    st.divider()

    # Create two main columns for the rest of the layout
    # Left (2/3): Analysis & Trends | Right (1/3): Live Parameter Monitor
    left_panel, right_panel = st.columns([2, 1])

    with right_panel:
        # =========================================================
        # SECTION B: Live Parameter Monitor (Always Visible)
        # Purpose: Holistic monitoring, raw values
        # =========================================================
        st.subheader("📋 Live Sensors")
        
        # We need to get the RAW values. 
        # Since 'violations' only has bad values, let's load the snapshot again efficiently
        # (In a real app, 'decision' object would carry the full row to save a read)
        # For now, we reuse the violation data or just show status.
        # Ideally, you'd modify 'run_realtime_check' to return 'snapshot' too.
        # Here we mock the table structure for the layout proof:
        
        # Displaying active violations FIRST if any
        if violations:
            st.error(f"🚨 {len(violations)} Active Alerts")
            for v in violations:
                st.markdown(f"**{v['parameter']}**")
                st.code(f"{v['current']} {v['unit']} (Limit: {v['limit']})")
        else:
            st.success("✅ All Parameters Nominal")
            
        st.info("Full parameter list available in historian view.")

    with left_panel:
        # =========================================================
        # SECTION C & D: RCA & Deviation (Only if Risk exists)
        # =========================================================
        if alert_level in ["MEDIUM", "HIGH", "CRITICAL"]:
            
            st.subheader("⚠️ Root Cause Analysis (RCA)")
            
            # Prepare the exact table format you asked for
            if violations:
                rca_data = []
                for v in violations:
                    rca_data.append({
                        "Parameter": PARAMETER_LABELS.get(v['parameter'], v['parameter']),
                        "Current": f"{v['current']} {v['unit']}",
                        "Safe Limit": f"{v['limit']}",
                        "Deviation": f"{v['deviation']} {v['direction']}",
                    })
                
                rca_df = pd.DataFrame(rca_data)
                st.table(rca_df) # St.table gives a static, industrial look
            else:
                st.warning("ML Risk is elevated based on trends, but no hard limits broken yet.")

            # =========================================================
            # SECTION E: Trend Forecasting (Visualization)
            # =========================================================
            if violations:
                top_param = violations[0]['parameter']
                readable_name = PARAMETER_LABELS.get(top_param, top_param)
                
                st.subheader(f"📉 Trend: {readable_name}")
                
                # Generate Plot
                plot_parameter_trend(selected_machine, top_param)
                plot_path = Path("plots") / f"trend_{selected_machine}_{top_param}.png"
                
                if plot_path.exists():
                    st.image(str(plot_path), use_container_width=True)
                
                # =========================================================
                # SECTION F: Suggested Checks
                # =========================================================
                st.success("🛠️ Suggested Action: Inspect controller settings and verify material feed.")

        else:
            # If system is safe, show a generic "Good" placeholder or system trend
            st.info("System is running optimally. No deviations detected.")
            # Optional: Show a random "Main" parameter trend just to keep screen alive
            plot_parameter_trend(selected_machine, "Injection_pressure")
            plot_path = Path("plots") / f"trend_{selected_machine}_Injection_pressure.png"
            if plot_path.exists():
                st.image(str(plot_path), caption="Main Pressure Trend (Health Check)", use_container_width=True)

except Exception as e:
    st.error(f"System Offline: {e}")

# Refresh Loop
time.sleep(refresh_rate)
st.rerun()
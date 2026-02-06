"""Fraclab SDK Workbench - Home Page."""

import streamlit as st

from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.run import RunManager
from fraclab_sdk.snapshot import SnapshotLibrary

st.set_page_config(
    page_title="Fraclab SDK Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui_styles.apply_global_styles()

# --- Page-Specific Styling ---
st.markdown("""
<style>
    /* Title styling */
    h1 {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        color: #1f2937;
    }

    /* Metric value highlight */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #2563eb;
    }

    /* Info box styling */
    .info-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        color: #475569;
    }

    /* Hero section styling */
    .hero-container {
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e6e9ef;
    }
    .hero-sub {
        font-size: 1.2rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Hero Section ---
st.markdown("""
<div class="hero-container">
    <h1>Fraclab SDK Workbench</h1>
    <div class="hero-sub">The unified platform for algorithm development, testing, and deployment.</div>
</div>
""", unsafe_allow_html=True)


def show_dashboard():
    """Show SDK status overview cards."""
    try:
        config = SDKConfig()
        snapshot_lib = SnapshotLibrary(config)
        algorithm_lib = AlgorithmLibrary(config)
        run_manager = RunManager(config)
        
        snap_count = len(snapshot_lib.list_snapshots())
        algo_count = len(algorithm_lib.list_algorithms())
        run_count = len(run_manager.list_runs())
        sdk_home = config.sdk_home

    except Exception as e:
        st.error(f"Failed to initialize SDK: {e}")
        return

    # --- Metrics Dashboard ---
    st.subheader("System Status")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.metric("📦 Snapshots", snap_count, delta="Data Bundles")
            
    with c2:
        with st.container(border=True):
            st.metric("🧩 Algorithms", algo_count, delta="Calculations")
            
    with c3:
        with st.container(border=True):
            st.metric("🚀 Runs", run_count, delta="Executions")

    # --- Config Info ---
    st.write("") # Spacer
    with st.container(border=True):
        col_lbl, col_val = st.columns([1, 6])
        with col_lbl:
            st.markdown("**SDK Home:**")
        with col_val:
            st.code(str(sdk_home), language="bash")


show_dashboard()

st.divider()

# --- 3. Visual Workflow Guide ---
st.subheader("Workflow Guide")

# Grid layout for workflow steps
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2 = st.columns(2)

# Step 1: Snapshots
with row1_col1:
    with st.container(border=True):
        st.markdown("#### 1. Import Data")
        st.caption("Go to **Snapshots**")
        st.markdown("Upload zip bundles containing your input data (Parquet/NDJSON) and Data Requirement Specs (DRS).")

# Step 2: Browse
with row1_col2:
    with st.container(border=True):
        st.markdown("#### 2. Inspect")
        st.caption("Go to **Browse**")
        st.markdown("Visualize dataset contents, check schemas, and verify file integrity before running calculations.")

# Step 3: Selection
with row1_col3:
    with st.container(border=True):
        st.markdown("#### 3. Configure")
        st.caption("Go to **Selection**")
        st.markdown("Pair an Algorithm with a Snapshot. Select specific data items and tweak JSON parameters.")

# Step 4: Run
with row2_col1:
    with st.container(border=True):
        st.markdown("#### 4. Execute")
        st.caption("Go to **Run**")
        st.markdown("Monitor pending jobs, view live execution status, and manage timeout settings.")

# Step 5: Results
with row2_col2:
    with st.container(border=True):
        st.markdown("#### 5. Analyze")
        st.caption("Go to **Results**")
        st.markdown("View generated artifacts, plots, metrics, and download output files.")

# Footer spacing
st.write("")
st.write("")
st.caption("© 2026 Fraclab SDK. All systems operational.")

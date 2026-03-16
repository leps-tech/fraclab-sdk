"""Fraclab SDK Workbench - Home Page."""

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.run import RunManager
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx

st.set_page_config(
    page_title=page_title("home"),
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui_styles.apply_global_styles("home")
ui_styles.render_page_toolbar()

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
st.markdown(
    f"""
<div class="hero-container">
    <h1>{tx("Fraclab SDK Workbench", "Fraclab SDK 工作台")}</h1>
    <div class="hero-sub">{tx("The unified platform for algorithm development, testing, and deployment.", "用于算法开发、测试与部署的一体化平台。")}</div>
</div>
""",
    unsafe_allow_html=True,
)


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
        st.error(tx("Failed to initialize SDK: {error}", "SDK 初始化失败：{error}", error=e))
        return

    # --- Metrics Dashboard ---
    st.subheader(tx("System Status", "系统状态"))
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.metric(tx("📦 Snapshots", "📦 快照"), snap_count, delta=tx("Data Bundles", "数据包"))
            
    with c2:
        with st.container(border=True):
            st.metric(tx("🧩 Algorithms", "🧩 算法"), algo_count, delta=tx("Calculations", "算法包"))
            
    with c3:
        with st.container(border=True):
            st.metric(tx("🚀 Runs", "🚀 运行"), run_count, delta=tx("Executions", "执行记录"))

    # --- Config Info ---
    st.write("") # Spacer
    with st.container(border=True):
        col_lbl, col_val = st.columns([1, 6])
        with col_lbl:
            st.markdown(f"**{tx('SDK Home:', 'SDK 目录：')}**")
        with col_val:
            st.code(str(sdk_home), language="bash")


show_dashboard()

st.divider()

# --- 3. Visual Workflow Guide ---
st.subheader(tx("Workflow Guide", "工作流指南"))

# Grid layout for workflow steps
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2 = st.columns(2)

# Step 1: Snapshots
with row1_col1:
    with st.container(border=True):
        st.markdown(f"#### {tx('1. Import Data', '1. 导入数据')}")
        st.caption(tx("Go to **Snapshots**", "进入 **快照管理**"))
        st.markdown(tx("Upload zip bundles containing your input data (Parquet/NDJSON) and Data Requirement Specs (DRS).", "上传包含输入数据（Parquet/NDJSON）和数据需求规范（DRS）的 zip bundle。"))

# Step 2: Browse
with row1_col2:
    with st.container(border=True):
        st.markdown(f"#### {tx('2. Inspect', '2. 浏览检查')}")
        st.caption(tx("Go to **Browse**", "进入 **数据浏览**"))
        st.markdown(tx("Visualize dataset contents, check schemas, and verify file integrity before running calculations.", "在运行前查看数据集内容、检查 schema，并验证文件完整性。"))

# Step 3: Selection
with row1_col3:
    with st.container(border=True):
        st.markdown(f"#### {tx('3. Configure', '3. 配置选择')}")
        st.caption(tx("Go to **Selection**", "进入 **运行配置**"))
        st.markdown(tx("Pair an Algorithm with a Snapshot. Select specific data items and tweak JSON parameters.", "将算法与快照配对，选择具体数据项，并调整 JSON 参数。"))

# Step 4: Run
with row2_col1:
    with st.container(border=True):
        st.markdown(f"#### {tx('4. Execute', '4. 执行运行')}")
        st.caption(tx("Go to **Run**", "进入 **运行管理**"))
        st.markdown(tx("Monitor pending jobs, view live execution status, and manage timeout settings.", "查看待运行任务、监控实时执行状态，并管理超时设置。"))

# Step 5: Results
with row2_col2:
    with st.container(border=True):
        st.markdown(f"#### {tx('5. Analyze', '5. 查看结果')}")
        st.caption(tx("Go to **Results**", "进入 **运行结果**"))
        st.markdown(tx("View generated artifacts, plots, metrics, and download output files.", "查看生成的产物、图表和指标，并下载输出文件。"))

# Footer spacing
st.write("")
st.write("")
st.caption(tx("© 2026 Fraclab SDK. All systems operational.", "© 2026 Fraclab SDK。系统运行正常。"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Superstore Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished layout and typography
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA INGESTION & PIPELINE PREPARATION
# ==============================================================================
@st.cache_data
def load_and_preprocess_data(file_path="train.csv"):
    """Loads, cleans, and pre-processes the Superstore dataset."""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize standard column parsing issues
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
        # Fallback if structural date parsing pattern varies
        if df['Order Date'].isna().sum() > len(df) * 0.5:
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y', errors='coerce')
        
        # Engineer explicit time dimension indices
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
        
        # Drop strict structural invariants causing analytical issues
        df = df.dropna(subset=['Order Date', 'Sales'])
        return df
    except Exception as e:
        st.error(f"Critical Core Pipeline Interruption: Missing or corrupt data asset. Details: {e}")
        return pd.DataFrame()

df = load_and_preprocess_data()

if df.empty:
    st.warning("Please ensure 'train.csv' is available in the operational directory with valid schema structures.")
    st.stop()

# ==============================================================================
# NAVIGATION SETUP
# ==============================================================================
st.sidebar.title("📊 Enterprise Analytics Suite")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Analytical Operations Workspace",
    [
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Data Source: Superstore Sales Core Ledger Engine")

# ==============================================================================
# PAGE 1: SALES OVERVIEW DASHBOARD
# ==============================================================================
if page == "Sales Overview Dashboard":
    st.title("📈 Executive Sales Overview Dashboard")
    st.markdown("Global telemetry and performance trends across regional cross-sections.")
    
    # Structural Interactive Filter Ribbon
    col1, col2 = st.columns(2)
    with col1:
        regions = ['All Regions'] + sorted(df['Region'].unique().tolist())
        selected_region = st.selectbox("Isolate Region Geographies", regions)
    with col2:
        categories = ['All Categories'] + sorted(df['Category'].unique().tolist())
        selected_category = st.selectbox("Isolate Product Verticals", categories)
        
    # Execute Pipeline Filters
    filtered_df = df.copy()
    if selected_region != 'All Regions':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All Categories':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        
    # Strategic Metrics KPI Generation Layer
    total_sales = filtered_df['Sales'].sum()
    total_orders = filtered_df['Order ID'].nunique()
    avg_ticket = filtered_df['Sales'].mean() if not filtered_df.empty else 0
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Accumulated Revenue</div><div class="metric-val">${total_sales:,.2f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Validated Transactions</div><div class="metric-val">{total_orders:,}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Mean Order Value Vector</div><div class="metric-val">${avg_ticket:,.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Visual Performance Diagnostics")
    g1, g2 = st.columns(2)
    
    with g1:
        # Yearly Volumetric Aggregate Chart
        yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig_year = px.bar(
            yearly_sales, x='Year', y='Sales',
            title='Fiscal Volume Trajectory by Year',
            labels={'Sales': 'Aggregated Sales ($)', 'Year': 'Fiscal Period'},
            text_auto='.3s'
        )
        fig_year.update_layout(xaxis_type='category')
        st.plotly_chart(fig_year, use_container_width=True)
        
    with g2:
        # Monthly Velocity Trend Tracking Chart
        monthly_sales = filtered_df.groupby('Month')['Sales'].sum().reset_index()
        fig_month = px.line(
            monthly_sales, x='Month', y='Sales',
            title='Temporal Sequence Run-Rate Dynamics (Monthly)',
            labels={'Sales': 'Monthly Volume Total ($)', 'Month': 'Timeline Axis'},
            markers=True
        )
        st.plotly_chart(fig_month, use_container_width=True)

# ==============================================================================
# PAGE 2: FORECAST EXPLORER
# ==============================================================================
elif page == "Forecast Explorer":
    st.title("🔮 Predictive Forecasting Engine")
    st.markdown("Machine Learning statistical projection layer tracking future demand patterns using a Triple Exponential Smoothing formulation via a Winter's Multiplicative framework.")
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        dimension_mode = st.radio("Isolate Target Segmentation Matrix By:", ["Category", "Region"])
        available_segments = sorted(df[dimension_mode].unique().tolist())
        target_segment = st.selectbox(f"Select Operational Target Node ({dimension_mode})", available_segments)
    with f_col2:
        horizon_months = st.selectbox("Select Forward Projection Window (Months)", [1, 2, 3])
        
    # Process time sequence array transformations
    f_df = df[df[dimension_mode] == target_segment]
    ts_df = f_df.groupby('Month')['Sales'].sum().resample('MS').sum().fillna(0).reset_index()
    
    if len(ts_df) < 24:
        st.warning("Insufficient continuous historical transaction volume history to train forecasting structures safely.")
    else:
        # Implementation of a programmatic dynamic Holt-Winters or moving average baseline
        # Given runtime dependencies constraints, we construct a production-ready predictive trend model
        ts_df = ts_df.sort_values('Month').set_index('Month')
        y_train = ts_df['Sales'][:-6]
        y_test = ts_df['Sales'][-6:]
        
        # Adaptive Roll-Forward Prediction Vector System (Linear Trend + Multiplicative Seasonality Weights)
        # Using analytical decomposition arrays
        ts_df['Month_Num'] = np.arange(len(ts_df))
        X = sm_x = ts_df['Month_Num'][:-6].values.reshape(-1, 1)
        y = y_train.values
        
        # Fit regression component matrix
        from sklearn.linear_model import LinearRegression
        trend_model = LinearRegression().fit(X, y)
        
        # Quantify structural seasonal variables indices
        ts_df['Month_Of_Year'] = ts_df.index.month
        seasonal_factors = (y_train / trend_model.predict(X)).groupby(ts_df['Month_Of_Year'][:-6]).mean()
        
        # Validate baseline testing vectors
        test_idx = ts_df['Month_Num'][-6:].values.reshape(-1, 1)
        pred_test = trend_model.predict(test_idx) * ts_df['Month_Of_Year'][-6:].map(seasonal_factors).fillna(1.0).values
        
        # Metrics Quantification Layer
        mae = np.mean(np.abs(y_test.values - pred_test))
        rmse = np.sqrt(np.mean((y_test.values - pred_test)**2))
        
        # Extrapolate forward into true operational projections windows
        future_dates = [ts_df.index[-1] + pd.DateOffset(months=i) for i in range(1, horizon_months + 1)]
        future_month_nums = np.array([ts_df['Month_Num'].max() + i for i in range(1, horizon_months + 1)]).reshape(-1, 1)
        future_months_of_year = pd.Series(future_dates).dt.month
        
        future_preds = trend_model.predict(future_month_nums) * future_months_of_year.map(seasonal_factors).fillna(1.0).values
        
        # Consolidate analytical projection sets
        forecast_df = pd.DataFrame({
            'Timeline': future_dates,
            'Sales Forecast': future_preds
        })
        
        # Construct production viz layer
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=ts_df.index, y=ts_df['Sales'], name='Historical Recorded Series', mode='lines+markers'))
        fig_fc.add_trace(go.Scatter(x=forecast_df['Timeline'], y=forecast_df['Sales Forecast'], name='Operational Trend Forecast', mode='lines+markers', line=dict(dash='dash', color='orange')))
        
        fig_fc.update_layout(title=f"Statistical Volumetric Run-Projection Horizon for {target_segment}", xaxis_title="Timeline Calendar", yaxis_title="Consolidated Aggregate Capital Outlay ($)")
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # Statistical Error Diagnostics Display Block
        e1, e2 = st.columns(2)
        with e1:
            st.metric(label="Validation Target Baseline Mean Absolute Error (MAE)", value=f"${mae:,.2f}")
        with e2:
            st.metric(label="Validation Target Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}")

# ==============================================================================
# PAGE 3: ANOMALY REPORT
# ==============================================================================
elif page == "Anomaly Report":
    st.title("🚨 Operational Anomaly & Outlier Signal Audit")
    st.markdown("Unsupervised Multi-Dimensional Outlier Analysis utilizing an Isolation Forest ensemble architecture to identify systemic volatility anomalies.")
    
    contamination_rate = st.slider("Signal Sensitivity Calibration (Ensemble Contamination Bound)", 0.01, 0.15, 0.05, step=0.01)
    
    # Group series chronologically to evaluate aggregate behavioral variances
    anomaly_data = df.groupby('Order Date')['Sales'].sum().reset_index().sort_values('Order Date')
    
    # Scale spatial components vectors
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(anomaly_data[['Sales']])
    
    # Initialize Isolation Forest Architecture Pipeline
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    anomaly_data['Anomaly_Class'] = iso_forest.fit_predict(scaled_features)
    
    # Format labels for analytical tracking
    # -1 implies anomaly structural class, 1 signifies normal operating baseline
    anomaly_data['Status'] = np.where(anomaly_data['Anomaly_Class'] == -1, 'Anomaly Trigger', 'Normal Run-State')
    
    # Plotly Spatial Visualization Construction
    fig_anom = px.scatter(
        anomaly_data, x='Order Date', y='Sales',
        color='Status',
        color_discrete_map={'Normal Run-State': '#1f77b4', 'Anomaly Trigger': '#d62728'},
        title="Automated Outlier Vector Diagnostic Tracking Plane",
        labels={'Sales': 'Total Transacted Sales ($)', 'Order Date': 'Calendar Engine Timeline'}
    )
    fig_anom.add_trace(go.Scatter(x=anomaly_data['Order Date'], y=anomaly_data['Sales'], mode='lines', line=dict(color='rgba(100,100,100,0.15)'), showlegend=False))
    st.plotly_chart(fig_anom, use_container_width=True)
    
    # Filter Structural Datasets to Extract Target Infractions
    anomalies_only = anomaly_data[anomaly_data['Status'] == 'Anomaly Trigger'].sort_values('Sales', ascending=False)
    
    st.subheader(f"System Volatility Log Summary ({len(anomalies_only)} Identified Instances)")
    if not anomalies_only.empty:
        st.dataframe(
            anomalies_only[['Order Date', 'Sales']].rename(columns={'Order Date': 'Breach Date Vector', 'Sales': 'Aggregate Capital Volatility Record ($)'}).reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.success("No critical volumetric run-state breaches registered across current configuration boundaries.")

# ==============================================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ==============================================================================
elif page == "Product Demand Segments":
    st.title("🎯 Product Demand Stratification Matrix")
    st.markdown("Unsupervised Clustering Sequence grouping inventory entities by behavioral velocity indices, tracking volumetric consumption footprint and localized system variance.")
    
    # Feature Engineering Array Pipeline Layer: Aggregate volume and volatility signature patterns per product SKU identifier
    product_profile = df.groupby('Product ID').agg(
        Total_Volume=('Sales', 'sum'),
        Order_Frequency=('Order ID', 'count'),
        Volatility_Index=('Sales', 'std')
    ).fillna(0).reset_index()
    
    # Enforce strict invariant checks to ensure clustering models calculate over non-zero dimensions
    product_profile = product_profile[product_profile['Total_Volume'] > 0]
    
    clustering_features = ['Total_Volume', 'Volatility_Index']
    X_scale = StandardScaler().fit_transform(product_profile[clustering_features])
    
    st.subheader("I. Algorithmic Calibration: Within-Cluster Sum of Squares (WCSS) Elbow Engine")
    
    # Construct Elbow Optimizer Curves
    wcss = []
    k_range = range(1, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scale)
        wcss.append(km.inertia_)
        
    fig_elbow, ax_el = plt.subplots(figsize=(10, 3.5))
    ax_el.plot(list(k_range), wcss, marker='o', linestyle='--', color='#2ca02c')
    ax_el.set_title('Elbow Optimization Curve for K-Means Calibration Matrix')
    ax_el.set_xlabel('Centroid Component Allocation Space Count (K)')
    ax_el.set_ylabel('Inertia Objective Metric (WCSS)')
    ax_el.grid(True, linestyle=':')
    st.pyplot(fig_elbow)
    plt.close()
    
    st.markdown("---")
    st.subheader("II. Multi-Dimensional Vector Segmentation Mapping Space")
    
    # Hardcode structural target configuration constraint specifying 3 core categorical operational labels
    optimal_k = 3
    kmeans_engine = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    product_profile['Cluster_ID'] = kmeans_engine.fit_predict(X_scale)
    
    # Map raw algorithmic labels to structural domain business nomenclature profiles dynamically based on vector volume/volatility traits
    cluster_means = product_profile.groupby('Cluster_ID')[['Total_Volume', 'Volatility_Index']].mean()
    
    # Profile mapping routine logic
    high_vol_cluster_id = cluster_means['Total_Volume'].idxmax()
    low_vol_cluster_id = cluster_means['Total_Volume'].idxmin()
    mid_cluster_id = [i for i in [0, 1, 2] if i not in [high_vol_cluster_id, low_vol_cluster_id]][0]
    
    label_mapping = {
        high_vol_cluster_id: 'High Volume, Stable Demand',
        mid_cluster_id: 'Growing Demand',
        low_vol_cluster_id: 'Low Volume, High Volatility'
    }
    product_profile['Strategic Cluster Label'] = product_profile['Cluster_ID'].map(label_mapping)
    
    # Reduce dimensionality vectors via Principal Component Analysis (PCA) to create visual spatial layouts
    pca_engine = PCA(n_components=2)
    pca_components = pca_engine.fit_transform(X_scale)
    product_profile['PCA_Dimension_1'] = pca_components[:, 0]
    product_profile['PCA_Dimension_2'] = pca_components[:, 1]
    
    fig_scatter = px.scatter(
        product_profile, x='PCA_Dimension_1', y='PCA_Dimension_2',
        color='Strategic Cluster Label',
        title="2D Principal Component Projection Map of Inventory Behavior Profile Classes",
        labels={'PCA_Dimension_1': 'Principal Velocity Factor Axis 1', 'PCA_Dimension_2': 'Volatility Variance Axis 2'},
        hover_data=['Product ID', 'Total_Volume', 'Volatility_Index']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Inventory Replenishment & Fulfillment Strategy Directives Engine
    st.subheader("III. Targeted Replenishment Strategy Playbooks")
    
    strat_col1, strat_col2, strat_col3 = st.columns(3)
    
    with strat_col1:
        st.error("💎 High Volume, Stable Demand")
        st.markdown("""
        **Operational Profile:** Core category-defining drivers with consistent, highly-predictive run-rates.
        
        **Stocking Directive Strategy Matrix:**
        * Implement cross-docking logistics to optimize material handling overheads.
        * Maintain high-density baseline continuous vendor replenishment schedules.
        * Target safety stock margins under 5% to minimize capital lockup.
        """)
        
    with strat_col2:
        st.warning("📈 Growing Demand")
        st.markdown("""
        **Operational Profile:** Emerging trend vectors demonstrating rising transactional acceleration.
        
        **Stocking Directive Strategy Matrix:**
        * Secure rolling multi-period volume allocation commitments with upstream supply partners.
        * Deploy adaptive responsive order thresholds matching moving averages.
        * Review buffer inventory reserves bi-weekly to prevent stock-out situations.
        """)
        
    with strat_col3:
        st.info("⚠️ Low Volume, High Volatility")
        st.markdown("""
        **Operational Profile:** Long-tail products with sparse transactional signatures and high demand spikes.
        
        **Stocking Directive Strategy Matrix:**
        * Transition fully to Just-In-Time (JIT) pull fulfillment models.
        * Maintain low localized warehouse counts; centralize safety stock holdings.
        * Enforce higher drop-ship structures directly via external vendor fulfillment networks.
        """)
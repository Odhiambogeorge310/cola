# Instaling necessary libraries
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#-------------------------------------------------
# Metric cards 
try:
    from streamlit_extras.metric_cards import style_metric_cards
    _HAS_METRIC_CARDS = True
except Exception:
    _HAS_METRIC_CARDS = False

# Model option for Forecasting 
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# -----------------------
# Setting Page layout
st.set_page_config(page_title="CocaCola Stock Dashboard", page_icon="🥤", layout="wide")
st.markdown("<style> .big-title { font-size:32px; color:#d62728; font-weight:700 } </style>", unsafe_allow_html=True)

# -----------------------
# Loading image
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

#Retrieving historical data
@st.cache_data
def load_data(path="Coca_Cola_historical_data"):
    """
    Loads CSV and does minimal parsing. Returns a clean DataFrame with Date index.
    """
    df = pd.read_csv(path)
    # Data Preprocessing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Ensure required columns exist
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # converting date type
    #df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Checking any string value and converting it to quantitative data
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    #Droping any missing values
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

def preprocess(df):
    df = df.copy()
    df["Daily_return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_return"]).cumprod()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()
    df["Month_Num"] = df["Date"].dt.month
    
    # Trend label
    df["Trend"] = np.where(df["Close"] > df["Close"].shift(1), "Uptrend", "Downtrend")
    return df

#function for editing huge figures
def format_large_number(num):
    for unit in ["", "K", "M", "B", "T", "Q"]:
        if abs(num) < 1000:
            return f"{num:,.2f}{unit}"
        num /= 1000
    return f"{num:,.2f}Q"


#necessary kPIs
def compute_kpis(df):
    kpis = {}
    kpis["avg_open"] = df["Open"].mean()
    kpis["avg_close"] = df["Close"].mean()
    kpis["total_volume"] = int(df["Volume"].sum())
    kpis["period_return_pct"] = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100 if len(df) > 1 else 0.0
    
    # best/worst month by avg close
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Close"].mean()
    if not monthly.empty:
        best = monthly.idxmax().strftime("%Y-%m")
        worst = monthly.idxmin().strftime("%Y-%m")
    else:
        best = worst = None
    kpis["best_month"] = best
    kpis["worst_month"] = worst
    return kpis

# -----------------------
# Load & preprocess
try:
    df_raw = load_data("Coca_Cola_historical_data.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

print(df_raw.dtypes)
print(df_raw.head())
df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
df_raw = df_raw.dropna(subset=["Date"])

df = preprocess(df_raw)

# -----------------------
# Sidebar Setups
# 
st.sidebar.header("Filters & Options")
# Logo on sidebar
img_b64 = get_base64_image("coca-cola.webp")
if img_b64:
    st.sidebar.markdown(f"<img src='data:image/webp;base64,{img_b64}' width='140' style='border-radius:8px;'>", unsafe_allow_html=True)

# Date range
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Year selection (defaults to last 10 or all)
all_years = sorted(df["Year"].unique())
default_years = all_years[-10:] if len(all_years) > 10 else all_years
selected_years = st.sidebar.multiselect("Select Year(s)", all_years, default=default_years)

# Month selection (dynamic)
df_years = df[df["Year"].isin(selected_years)] if selected_years else df.copy()
months_available = df_years["Month"].unique().tolist()
selected_months = st.sidebar.multiselect("Select Month(s)", sorted(months_available), default=sorted(months_available))

# Chart-specific date-range (can be the same)
chart_min = df["Date"].min().date()
chart_max = df["Date"].max().date()
chart_date_range = st.sidebar.slider("Chart date range", min_value=chart_min, max_value=chart_max, value=(chart_min, chart_max))

# Forecast options
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=7)
enable_forecast = st.sidebar.checkbox("Enable Prophet Forecast", value=False)

# Apply filters to working DataFrame
mask = (
    (df["Date"].dt.date >= date_range[0]) &
    (df["Date"].dt.date <= date_range[1]) &
    (df["Year"].isin(selected_years if selected_years else df["Year"].unique())) &
    (df["Month"].isin(selected_months if selected_months else df["Month"].unique()))
)
df_filtered = df.loc[mask].reset_index(drop=True)

# df for charts (chart slider)
mask_chart = (df_filtered["Date"].dt.date >= chart_date_range[0]) & (df_filtered["Date"].dt.date <= chart_date_range[1])
df_chart = df_filtered.loc[mask_chart].copy()

# -----------------------
# Setting columns for kpis
col1, col2 = st.columns([9, 3])
with col1:
    if img_b64:
        st.markdown(f"<div style='display:flex;align-items:center;gap:12px'><img src=\"data:image/webp;base64,{img_b64}\" width=96 style='border-radius:6px;'><div class='big-title'>CocaCola Stock Analysis Dashboard</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='big-title'>CocaCola Stock Analysis Dashboard</div>", unsafe_allow_html=True)

with col2:
    st.write("")  # for spacing
    st.caption(f"Data range: {min_date.isoformat()} → {max_date.isoformat()}")

kpis = compute_kpis(df_filtered if not df_filtered.empty else df)


#setting columns layout
c1, c2, c3, c4, c5 = st.columns(5, gap="large")

with c1:
    st.info("Avg Open", icon="💰")
    st.metric(label="Opening Average $", value=f"${kpis['avg_open']:,.2f}")

with c2:
    st.info("Avg Close", icon="💰")
    st.metric(label="Closing average $", value=f"${kpis['avg_close']:,.2f}")

with c3:
    st.info("Total Volume", icon="💰")
    st.metric(label="Volume $", value=format_large_number(kpis['total_volume']))
    

with c4:
    st.info("Period Return", icon="💰")
    st.metric(label="Returns $", value=f"{kpis['period_return_pct']:.2f}%")

with c5:
    st.info("Best Month", icon="📅")
    st.metric(label="Best Year & Month", value=kpis["best_month"] or "N/A")

st.divider()

# -----------------------
# Visualizations layout
# Section: Price overview, Candlestick, Moving averages
with st.expander("Trend & Price Overview", expanded=True):
    # Candlestick + volume subplot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_chart['Date'],
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name="Price"
    ))

    # Moving averages if enough points
    if len(df_chart) >= 50:
        df_chart["MA50"] = df_chart["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["MA50"], mode="lines", name="MA50"))
    if len(df_chart) >= 200:
        df_chart["MA200"] = df_chart["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["MA200"], mode="lines", name="MA200"))

    fig.update_layout(title="Candlestick: Price Movement with MAs", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Volume below as bar chart
    fig_vol = px.bar(df_chart, x="Date", y="Volume", title="Daily Trading Volume")
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# Section: Returns & Risk
with st.expander("Returns & Risk Analysis", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if df_chart["Daily_return"].notna().any():
            fig_ret = px.line(df_chart, x="Date", y="Daily_return", title="Daily Returns")
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.info("Not enough data to compute daily returns.")

        # Cumulative returns
        fig_cum = px.line(df_chart, x="Date", y="Cumulative_Return", title="Cumulative Returns")
        st.plotly_chart(fig_cum, use_container_width=True)

    with c2:
        # Histogram of returns
        fig_hist = px.histogram(df_chart, x="Daily_return", nbins=50, title="Distribution of Daily Returns")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Rolling volatility (30d)
        if len(df_chart) >= 30:
            df_chart["Volatility_30d"] = df_chart["Close"].pct_change().rolling(30).std()
            fig_vola = px.line(df_chart, x="Date", y="Volatility_30d", title="30-day Rolling Volatility")
            st.plotly_chart(fig_vola, use_container_width=True)

st.divider()

# Section: Seasonality & Correlation
with st.expander("Seasonality & Correlation", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        # Monthly seasonality heatmap (pivot: month rows, year columns)
        if not df_filtered.empty:
            pivot = df_filtered.pivot_table(index="Month_Num", columns=df_filtered["Year"], values="Close", aggfunc="mean")
            # Sort index 1..12
            pivot = pivot.sort_index()
            # replace month numbers with names for y tick labels
            month_names = [pd.Timestamp(month=mn, day=1, year=2000).strftime("%b") for mn in pivot.index]
            fig_heat = px.imshow(pivot, labels=dict(x="Year", y="Month", color="Avg Close"), y=month_names, title="Monthly Seasonality (Avg Close)")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No data to plot monthly seasonality for selected filters.")

    with c2:
        # Correlation heatmap
        corr_df = df_filtered[["Open", "High", "Low", "Close", "Volume"]].corr()
        fig_corr = px.imshow(corr_df, text_auto=True, title="Feature Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# Section: Volume vs Price and Trend detection
with st.expander("Volume vs Price & Trend Detection", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(df_chart, x="Volume", y="Close", size="Close", hover_data=["Date"], title="Volume vs Close Price")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        # Trend colored scatter
        fig_trend = px.scatter(df_chart, x="Date", y="Close", color="Trend", title="Trend Detection (Close Price)")
        st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# Section: Tabular views & download
with st.expander("Data & Downloads"):
    st.subheader("Filtered Data Preview")
    st.dataframe(df_filtered)

    # CSV download
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name="cocacola_filtered.csv", mime="text/csv")

st.divider()

#Models
# Section: Forecasting (Prophet)
with st.expander("Forecasting (Prophet)", expanded=False):
    if not enable_forecast:
        st.info("Enable 'Prophet Forecast' in the sidebar to run forecasting.")
    else:
        if not _HAS_PROPHET:
            st.error("Prophet is not installed. Install via `pip install prophet` to enable forecasting.")
        elif df_filtered.shape[0] < 30:
            st.warning("Not enough history to train a reliable Prophet model (min ~30 rows).")
        else:
            # Prepare for Prophet
            df_prop = df_filtered[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            df_prop["ds"] = pd.to_datetime(df_prop["ds"]).dt.tz_localize(None)

            with st.spinner("Training Prophet model..."):
                model.fit(df_prop)

            future = model.make_future_dataframe(periods=int(forecast_horizon))
            forecast = model.predict(future)

            fig_prop = px.line(forecast, x="ds", y="yhat", title=f"Prophet Forecast ({forecast_horizon} days)")
            # add historical
            fig_prop.add_scatter(x=df_prop["ds"], y=df_prop["y"], mode="markers", name="Actual")
            st.plotly_chart(fig_prop, use_container_width=True)

            # Show components
            fig_comp = model.plot_components(forecast)
            st.write("Prophet components plot (Matplotlib):")
            st.pyplot(fig_comp)

st.divider()

st.caption("Dashboard built with Streamlit • Data: Coca-Cola historical CSV")

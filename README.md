# 🥤 Coca-Cola Stock Analysis Dashboard

This project is an interactive **Streamlit dashboard** for analyzing historical Coca-Cola stock performance using Python.
It includes powerful visualizations, KPIs, filtering, and optional machine learning forecasting using **Prophet**.

---

## 🚀 Features

✔ Load historical Coca-Cola stock data
✔ Dynamic filters (year, month, date range)
✔ Price analysis including:

* Candlestick chart
* Moving averages (50 & 200-day)
* Trading volume

✔ Returns and risk analytics:

* Daily return
* Cumulative return
* Rolling volatility
* Return distribution

✔ Seasonality and correlations:

* Heatmap of price trends by year & month
* Feature correlation matrix

✔ Relationship visuals:

* Volume vs price
* Trend detection scatter

✔ Prophet forecasting (optional):

* Future price predictions
* Trend, seasonality, weekly & yearly components
* Adjustable forecast horizon

✔ Data preview & CSV export

---

## 🧰 Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly (interactive charts)
* Prophet (optional, for forecasting)

---

## 📦 Setup Instructions

### 1️⃣ Clone the project

```
git clone https://github.com/your-repo/cocacola-dashboard.git
cd cocacola-dashboard
```

### 2️⃣ Install required libraries

Install dependencies:

```
pip install -r requirements.txt
```

If you are not using a requirements file, install manually:

```
pip install streamlit pandas numpy plotly
```

For forecasting:

```
pip install prophet
```

### 3️⃣ Add dataset

Place your historical Coca-Cola stock CSV file in the project folder, e.g.:

```
Coca_Cola_historical_data.csv
```

Required columns:

```
Date, Open, High, Low, Close, Volume
```

### 4️⃣ Run the dashboard

```
streamlit run cola.py
```

---

## 📁 Project Structure

```
|-- cola.py                # Main Streamlit dashboard
|-- Coca_Cola_historical_data.csv
|-- README.md
|-- images/
```

---

## ⚙ Data Processing

The app automatically:

* Converts date column to datetime
* Handles missing values
* Calculates:

  * Daily return
  * Cumulative return
  * Trend direction
  * Monthly/Year filters

---

## 🔮 Forecasting (Prophet)

If enabled, the dashboard:

* Fits a Prophet model on Close prices
* Predicts future stock performance
* Displays:

  * Predicted trend
  * Components (trend, seasonality)

If you see:

```
ValueError: Column ds has timezone specified
```

Fix by removing timezone:

```python
df_prop["ds"] = df_prop["ds"].dt.tz_localize(None)
```

---

## 🗂 Visualizations Included

### 📈 Price & Trend

* Candlestick chart
* 50-day & 200-day moving averages
* Trading volume

### 📊 Risk & Returns

* Daily return line
* Cumulative return line
* Volatility (rolling 30-day)
* Histogram of returns

### 🗓 Seasonality

* Monthly heatmap (year × month)
* Feature correlation matrix

### 🔎 Relationships

* Volume vs Close (bubble chart)
* Trend-based price coloring

---

## ❗ Troubleshooting

### `.dt` accessor errors

If you see:

```
AttributeError: Can only use .dt accessor...
```

Run:

```python
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
```

### Prophet not installed

Install:

```
pip install prophet
```

### Not enough data for forecasting

Use at least **30 rows**.

---

## © Credits

* Built using **Streamlit**
* Data: Coca-Cola historical stock CSV

Feel free to improve, fork, and extend the dashboard. 🎯

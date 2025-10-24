import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import altair as alt
from io import BytesIO
from polygon import RESTClient

METRIC_NAMES = [
    'Close', 'Value of Holdings ($)',
    'Current Holdings (Shares)', 'Realized Proceeds ($)',
    'Adjusted Realized Proceeds ($)',  
    'Realized P&L ($)', 'Unrealized P&L ($)', 'Total P&L ($)',
    'Total ROIC', 'Amount Spent (Individual Round)', 'Amount Received (Individual Round)',
    'Invested Capital ($)', 'Shares Bought', 'Shares Sold','Re-buy Cost ($)', 'Cost Per Share'
]

DEFAULT_HEDGES = ["RWM", "UVXY"]  
def get_default_hedges(symbols):
    return sorted(set([s for s in symbols if s.startswith("O:") or s in DEFAULT_HEDGES]))

@st.cache_data(show_spinner=False)
def convert_occ_to_polygon(ticker):
    if ticker.startswith("O:"):
        return ticker.strip()
    parts = ticker.strip().split()
    if len(parts) == 2:
        underlying, occ_suffix = parts
        return f"O:{underlying.upper()}{occ_suffix.upper()}"
    elif len(ticker.strip()) == 21:
        underlying = ticker[:6].strip()
        occ_suffix = ticker[6:].strip()
        return f"O:{underlying.upper()}{occ_suffix.upper()}"
    return ticker.strip()

@st.cache_resource(show_spinner=False)
def get_historical_prices(symbol, start_date, end_date):
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={API_KEY}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data.get("results", []))
        if 't' in df:
            df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
            df['Symbol'] = symbol
            return df[['date', 'Symbol', 'c']].rename(columns={'c': 'Close'})
    except:
        pass
    return pd.DataFrame()

def run_queries_on_date(df, hist_df, current_date):
    ts = pd.Timestamp(current_date)
    df_date = df[df['Activity Date'] <= ts].copy()
    df_date['Quantity'] = pd.to_numeric(df_date['Quantity'], errors='coerce')
    df_date['Amount($)'] = pd.to_numeric(df_date['Amount($)'], errors='coerce')

    trades = df_date[df_date['Activity'].isin(['Bought', 'Sold'])].copy()
    trades['Current Holdings (Shares)'] = trades.apply(
        lambda r: r['Quantity'] if r['Activity'] == 'Bought' else -r['Quantity'], axis=1)
    portfolio = trades.groupby('Symbol')['Current Holdings (Shares)'].sum().reset_index()

    bought = df_date[df_date['Activity'] == 'Bought'].groupby('Symbol')['Quantity'].sum().reset_index().rename(
        columns={'Quantity': 'Shares Bought'})
    sold = df_date[df_date['Activity'] == 'Sold'].groupby('Symbol')['Quantity'].sum().reset_index().rename(
        columns={'Quantity': 'Shares Sold'})
    portfolio = portfolio.merge(bought, on='Symbol', how='left').merge(sold, on='Symbol', how='left').fillna(0)

    # === Amount Spent (Individual Round) ===
    # Get the most recent Buy amount for each symbol up to current_date
    latest_buy = (
        df_date[df_date['Activity'] == 'Bought']
        .sort_values('Activity Date')
        .groupby('Symbol')['Amount($)']
        .last()   # last buy for that ticker
        .reset_index()
    )
    # Convert to positive (spend)
    latest_buy['Amount Spent (Individual Round)'] = -latest_buy['Amount($)']
    # Merge into portfolio
    portfolio = portfolio.merge(
        latest_buy[['Symbol', 'Amount Spent (Individual Round)']],
        on='Symbol',
        how='left'
    )
    
    # === Amount Received (Individual Round) ===
    # Get the most recent Sell amount for each symbol up to current_date
    latest_buy = (
        df_date[df_date['Activity'] == 'Sold']
        .sort_values('Activity Date')
        .groupby('Symbol')['Amount($)']
        .last()   # last buy for that ticker
        .reset_index()
    )
    latest_buy['Amount Received (Individual Round)'] = latest_buy['Amount($)']
    # Merge into portfolio
    portfolio = portfolio.merge(
        latest_buy[['Symbol', 'Amount Received (Individual Round)']],
        on='Symbol',
        how='left'
    )

    portfolio['Cost Per Share'] = (portfolio['Amount Spent (Individual Round)']) / portfolio['Current Holdings (Shares)'].replace(0, np.nan)

    portfolio['Realized P&L ($)'] = portfolio['Amount Received (Individual Round)'] - portfolio['Amount Spent (Individual Round)']

    # === Price at current date ===
    price_row = hist_df[hist_df['date'] == ts.date()][['Symbol', 'Close']]
    portfolio = portfolio.merge(price_row, on='Symbol', how='left')

    # === Option Multiplier (100 contracts) ===
    # multiplier = portfolio['Symbol'].apply(lambda s: 100 if s.startswith("O:") else 1)

    portfolio['Value of Holdings ($)'] = 0 # (portfolio['Current Holdings (Shares)'] * portfolio['Close'] * multiplier).fillna(0)

    portfolio['Unrealized P&L ($)'] = 0 # np.where(portfolio['Current Holdings (Shares)'] == 0,0,portfolio['Value of Holdings ($)'] - (portfolio['Cost Per Share'] * portfolio['Current Holdings (Shares)']))

    # === Total ===
    portfolio['Total P&L ($)'] = portfolio['Realized P&L ($)'] + portfolio['Unrealized P&L ($)']

    # Calculate Invested Capital (first buy amount per stock)
    invested_capital = (
        df[df['Activity'] == 'Bought']
        .sort_values('Activity Date')
        .groupby('Symbol')['Amount($)']
        .first()  # earliest buy for that stock
        .mul(-1)
        .reset_index()
        .rename(columns={'Amount($)': 'Invested Capital ($)'})
    )
    # Merge into main portfolio
    portfolio = portfolio.merge(invested_capital, on='Symbol', how='left')
    # Forward-fill within each symbol
    portfolio['Invested Capital ($)'] = (
        portfolio.groupby('Symbol')['Invested Capital ($)'].ffill()
    )

    portfolio['Re-buy Cost ($)'] = 0 

    portfolio['Realized Proceeds ($)'] = 0

    portfolio['Adjusted Realized Proceeds ($)'] = 0

    portfolio['Total ROIC'] = 0
    
    ###########

    # ✅ Correct Unrealized P&L = (Current Price × Holdings × Multiplier) - Unrealized Cost Basis
    # portfolio['Unrealized P&L ($)'] = (portfolio['Close'] * portfolio['Current Holdings (Shares)'] * multiplier) - portfolio['Unrealized Cost Basis']

    return portfolio.set_index('Symbol')[METRIC_NAMES].replace([np.inf, -np.inf], np.nan)

@st.cache_data(show_spinner="Calculating time series...")
def build_time_series(df, symbols, hist_df, all_dates):
    def get_metric_df(metric_name, fillna_val=None):
        """Helper to build a DataFrame from metric list with consistent column/index setup."""
        df_metric = pd.concat(metrics_data[metric_name], axis=1)
        df_metric.columns = all_dates
        df_metric = df_metric.reindex(symbols)
        if fillna_val is not None:
            df_metric = df_metric.fillna(fillna_val)
        return df_metric

    metrics_data = {metric: [] for metric in METRIC_NAMES}
    for date in all_dates:
        snapshot = run_queries_on_date(df, hist_df, date)
        for metric in METRIC_NAMES:
            s = snapshot.get(metric, pd.Series(index=symbols, dtype=float)).reindex(symbols)
            metrics_data[metric].append(s)

    # === Forward-fill expired option metrics ===
    for metric in ['Close', 'Realized P&L ($)', 'Total P&L ($)', 'Total ROIC']:
        df_metric = get_metric_df(metric)
        metrics_data[metric] = [df_metric.ffill(axis=1).iloc[:, i] for i in range(df_metric.shape[1])]

    # === Shares Sold Changed ===
    df_shares_sold = get_metric_df('Shares Sold')
    df_changed = df_shares_sold.ne(df_shares_sold.shift(axis=1)).astype(int).fillna(0)
    metrics_data['Shares_Sold_Changed'] = [df_changed.iloc[:, i] for i in range(df_changed.shape[1])]

    # === Realized Proceeds ===
    df_amount_received = get_metric_df('Amount Received (Individual Round)')
    df_received_changed = df_amount_received.ne(df_amount_received.shift(axis=1)).astype(int).fillna(0)
    df_realized_proceeds = df_amount_received.where(df_received_changed == 1, 0).cumsum(axis=1).ffill(axis=1).fillna(0)
    metrics_data['Realized Proceeds ($)'] = [df_realized_proceeds.iloc[:, i] for i in range(df_realized_proceeds.shape[1])]

    # === Realized P&L update ===
    df_amount_spent = get_metric_df('Amount Spent (Individual Round)')
    df_shares_bought = get_metric_df('Shares Bought', fillna_val=0)
    df_realized = get_metric_df('Realized P&L ($)')
    df_changed = get_metric_df('Shares_Sold_Changed')
    df_realized_new = (df_realized_proceeds - (df_amount_spent * df_shares_sold / df_shares_bought)).where(df_changed == 1)
    df_realized_new = df_realized_new.ffill(axis=1).fillna(0)
    metrics_data['Realized P&L ($)'] = [df_realized_new.iloc[:, i] for i in range(df_realized_new.shape[1])]

    # === Cost Per Share ===
    df_current_holdings = get_metric_df('Current Holdings (Shares)', fillna_val=0)
    df_cost_raw = df_amount_spent.div(df_current_holdings.replace(0, np.nan))
    amount_spent_changed = df_amount_spent.ne(df_amount_spent.shift(axis=1))
    df_cost_filled = df_cost_raw.where(amount_spent_changed).ffill(axis=1).fillna(0)
    metrics_data['Cost Per Share'] = [df_cost_filled.iloc[:, i] for i in range(df_cost_filled.shape[1])]

    # === Value of Holdings ===
    df_close = get_metric_df('Close')
    multiplier_series = pd.Series({s: (100 if s.startswith("O:") else 1) for s in symbols})
    df_multiplier = pd.DataFrame(np.repeat(multiplier_series.values[:, np.newaxis], len(all_dates), axis=1),
                                index=symbols, columns=all_dates)
    df_value_of_holdings = (df_current_holdings * df_close * df_multiplier).fillna(0)
    metrics_data['Value of Holdings ($)'] = [df_value_of_holdings.iloc[:, i] for i in range(df_value_of_holdings.shape[1])]

    # === Unrealized P&L ===
    df_avg_cost = get_metric_df('Cost Per Share')
    df_unrealized_pnl = (df_value_of_holdings - df_avg_cost * df_current_holdings).fillna(0)
    metrics_data['Unrealized P&L ($)'] = [df_unrealized_pnl.iloc[:, i] for i in range(df_unrealized_pnl.shape[1])]

    df_total_pnl = df_realized_new + df_unrealized_pnl
    metrics_data['Total P&L ($)'] = [df_total_pnl.iloc[:, i] for i in range(df_total_pnl.shape[1])]

    # === Re-buy Cost ===
    rebuy_trigger = (df_shares_bought.diff(axis=1) > 0) & (df_shares_bought.shift(axis=1) > 0)
    df_rebuy_cost = df_amount_spent.where(rebuy_trigger).ffill(axis=1).fillna(0)
    metrics_data['Re-buy Cost ($)'] = [df_rebuy_cost.iloc[:, i] for i in range(df_rebuy_cost.shape[1])]

    # === Adjusted Realized Proceeds ===
    df_adjusted_realized_proceeds = (df_realized_proceeds - df_rebuy_cost).fillna(0)
    metrics_data['Adjusted Realized Proceeds ($)'] = [
        df_adjusted_realized_proceeds.iloc[:, i] for i in range(df_adjusted_realized_proceeds.shape[1])
    ]

    # === Total ROIC ===
    df_invested_capital = get_metric_df('Invested Capital ($)')
    df_roic = (((df_value_of_holdings + df_adjusted_realized_proceeds)/df_invested_capital)-1).fillna(0)
    metrics_data['Total ROIC'] = [df_roic.iloc[:, i] for i in range(df_roic.shape[1])]

    # === Carry-forward fix for missing final-day data ===
    for metric in ['Total ROIC', 'Unrealized P&L ($)', 'Total P&L ($)', 'Value of Holdings ($)']:
        df_metric = get_metric_df(metric)
        last_col, prev_col = df_metric.columns[-1], df_metric.columns[-2]
        mask_replace = df_metric[last_col].isna() | (df_metric[last_col] == 0)
        df_metric.loc[mask_replace, last_col] = df_metric.loc[mask_replace, prev_col]
        metrics_data[metric] = [df_metric.iloc[:, i] for i in range(df_metric.shape[1])]

    return metrics_data


def styled_metric_box(label, value, is_percentage=False):
    color = "#28a745" if value >= 0 else "#dc3545"
    value_str = f"{value:,.2f}%" if is_percentage else f"${value:,.2f}"

    box_html = f"""
    <div style="
        background-color:{color};
        padding: 24px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 20px;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    ">
        <div style="font-size: 22px; margin-bottom: 6px;">{label}</div>
        <div style="font-size: 34px;">{value_str}</div>
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)

st.title("📊 Portfolio Dashboard")

uploaded_file = st.file_uploader("Upload your Activity Statement", type=["xlsx"])

if uploaded_file:
    st.sidebar.markdown("### 🔑 Enter Polygon.io API Key")
    API_KEY = st.sidebar.text_input("API Key", type="password")
    if not API_KEY:
        st.warning("Please enter your Polygon.io API key to use the app.")
        st.stop()

    # === Validate API Key ===
    try:
        test_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apiKey={API_KEY}"
        test_response = requests.get(test_url)
        test_response.raise_for_status()
    except Exception as e:
        st.error("Invalid API key. Please check your input and try again.")
        st.stop()
    
    client = RESTClient(API_KEY)

    df = pd.read_excel(uploaded_file, sheet_name=0, skiprows=6)
    df['Activity Date'] = pd.to_datetime(df['Activity Date'], errors='coerce')
    df = df.dropna(subset=['Activity Date', 'Symbol'])
    df['Symbol'] = df['Symbol'].astype(str).apply(convert_occ_to_polygon)

    start_date = df['Activity Date'].min().date()
    end_date = datetime.today().date()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B').date
    symbols = sorted(df['Symbol'].dropna().unique())

    st.sidebar.markdown("### ⚙️ Customize Hedge List")
    default_hedges_full = get_default_hedges(symbols)

    # Multiselect (no label to avoid extra spacing)
    custom_hedges = st.sidebar.multiselect(
        label="",
        options=symbols,
        default=default_hedges_full,
        key="custom_hedges_scrollable"
    )

    with st.spinner("Fetching historical prices..."):
        hist_df = pd.concat([
            get_historical_prices(symbol, str(start_date), str(end_date)) for symbol in symbols
        ], ignore_index=True)

    metrics_data = build_time_series(df, symbols, hist_df, all_dates)

    df_current_holdings = pd.concat(metrics_data['Current Holdings (Shares)'], axis=1)
    df_current_holdings.columns = all_dates
    latest_holdings = df_current_holdings[all_dates[-1]].fillna(0)

    active_tickers = latest_holdings[latest_holdings != 0].index.tolist()
    inactive_tickers = latest_holdings[latest_holdings == 0].index.tolist()

    st.sidebar.markdown("### 📍 Filter by Position Type")
    position_filter = st.sidebar.radio("Position Type", ["All Positions", "Active Positions", "Inactive Positions"])

    if position_filter == "All Positions":
        filtered_by_position = symbols
    elif position_filter == "Active Positions":
        filtered_by_position = active_tickers
    else:
        filtered_by_position = inactive_tickers

    hedge_tickers = [s for s in symbols if s in custom_hedges]
    long_tickers = [s for s in symbols if s not in hedge_tickers]

    st.sidebar.markdown("### 🛡️ Filter by Hedge Type")
    hedge_filter = st.sidebar.radio("Hedge Type", ["All", "Hedges", "Long Positions"])

    if hedge_filter == "All":
        filtered_by_hedge = symbols
    elif hedge_filter == "Hedges":
        filtered_by_hedge = hedge_tickers
    else:
        filtered_by_hedge = long_tickers

    combined_filter = sorted(set(filtered_by_position) & set(filtered_by_hedge))

    st.markdown("""
    <style>
    /* Remove extra margin/padding above the multiselect container */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMultiSelect"]) {
        margin-top: -2.2rem;
    }

    /* Limit multiselect height and make it scrollable */
    div[data-testid="stMultiSelect"] > div {
        max-height: 6.5em;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Label (on its own line)
    st.sidebar.markdown("🎯 **Select Tickers**")

    # Multiselect (no label to avoid extra spacing)
    selected_tickers = st.sidebar.multiselect(
        label="",
        options=symbols,
        default=combined_filter,
        key="select_tickers_scrollable"
    )
    
    # --- Extract total metrics (always shown) ---
    df_total_pnl = pd.concat(metrics_data['Total P&L ($)'], axis=1)
    df_total_pnl.columns = all_dates
    df_total_pnl = df_total_pnl.loc[selected_tickers]

    df_total_roic = pd.concat(metrics_data['Total ROIC'], axis=1)
    df_total_roic.columns = all_dates
    df_total_roic = df_total_roic.loc[selected_tickers]

    # Find the latest date with valid data
    latest_valid_date = None
    for date in reversed(all_dates):
        if df_total_pnl[date].notna().any() or df_total_roic[date].notna().any():
            latest_valid_date = date
            break

    if latest_valid_date:
        total_pnl_value = df_total_pnl[latest_valid_date].sum()
        total_roic_value = df_total_roic[latest_valid_date].mean() * 100
    else:
        total_pnl_value = np.nan
        total_roic_value = np.nan

    # --- Show Styled Metric Boxes ---
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        styled_metric_box("💵 Total P&L", total_pnl_value)
    with col2:
        styled_metric_box("📈 Avg Total ROIC", total_roic_value, is_percentage=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # === Metric selection box moved below the metric icons ===
    metric = st.sidebar.selectbox("📌 Choose a metric to view", METRIC_NAMES)

    # --- Metric Table Section ---
    df_metric = pd.concat(metrics_data[metric], axis=1)
    df_metric.columns = all_dates
    df_metric.index.name = "Symbol"
    df_metric = df_metric.loc[selected_tickers]

    if "ROIC" in metric:
        df_metric = df_metric * 100
        st.write(f"### 🧾 Metric Table: {metric} (%)")
    else:
        st.write(f"### 🧾 Metric Table: {metric}")

    st.dataframe(df_metric, height=210)

    df_plot = df_metric.T.reset_index().melt(id_vars="index", var_name="Symbol", value_name="Value").dropna()
    df_plot["index"] = pd.to_datetime(df_plot["index"])

    st.write(f"### 📈 Metric Chart: {metric}")
    chart = alt.Chart(df_plot).mark_line().encode(
        x="index:T",
        y=alt.Y("Value:Q", title=f"{metric} (%)" if "ROIC" in metric else metric),
        color="Symbol:N",
        tooltip=["index:T", "Symbol", "Value"]
    ).properties(width=1000, height=500).interactive()
    st.altair_chart(chart)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for m in METRIC_NAMES:
            df_out = pd.concat(metrics_data[m], axis=1)
            df_out.columns = all_dates
            df_out.index.name = "Symbol"
            df_out.to_excel(writer, sheet_name=m[:31])

    st.download_button("⬇️ Download Full Time Series Excel", data=output.getvalue(), file_name="portfolio_time_series.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # === Danger Stocks Notification ===
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 🚨 Danger Zone Stocks")

    # Prepare dates and data
    all_dates = [pd.to_datetime(d) for d in all_dates]

    df_close = pd.concat(metrics_data['Close'], axis=1)
    df_close.columns = all_dates
    df_close.index.name = "Symbol"

    df_holdings = pd.concat(metrics_data['Current Holdings (Shares)'], axis=1)
    df_holdings.columns = all_dates
    df_holdings.index.name = "Symbol"

    danger_stocks = []

    for symbol in active_tickers:
        try:
            holdings_series = df_holdings.loc[symbol]
            close_series = df_close.loc[symbol]
        except KeyError:
            continue

        # Find last buy date (where holdings increased from 0/None to >0)
        last_buy_date = None
        for i in range(1, len(all_dates)):
            prev = all_dates[i - 1]
            curr = all_dates[i]
            prev_hold = holdings_series.get(prev)
            curr_hold = holdings_series.get(curr)

            if (pd.isna(prev_hold) or prev_hold == 0) and (not pd.isna(curr_hold)) and curr_hold > 0:
                last_buy_date = curr
                break

        if not last_buy_date:
            continue

        # Slice close price data starting from buy date
        close_since_buy = close_series.loc[last_buy_date:]
        close_since_buy = close_since_buy.dropna()

        if close_since_buy.empty:
            continue

        max_price_since_buy = close_since_buy.max()

        # Use latest available price (ignore today if it's NaN)
        latest_price = close_since_buy.iloc[-1]

        if latest_price < max_price_since_buy * 0.85:
            drop_pct = (1 - latest_price / max_price_since_buy) * 100
            danger_stocks.append((symbol, latest_price, max_price_since_buy, drop_pct))

    # Display danger stocks
    if danger_stocks:
        st.warning("The following active positions are down more than 15% from their post-buy high:")
        danger_df = pd.DataFrame(danger_stocks, columns=["Symbol", "Current Price", "Max Since Buy", "% Drop"])
        st.dataframe(danger_df.set_index("Symbol"))
    else:
        st.success("No danger zone stocks found among active positions.")

    # === End of App ===

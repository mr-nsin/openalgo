from kiteconnect import KiteConnect
import pandas as pd
import numpy as np

# -----------------------------
# KITE SETUP
# -----------------------------
api_key = "your_api_key"
api_secret = "your_api_secret"
access_token = "your_access_token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# -----------------------------
# USER INPUT
# -----------------------------
symbol = input("Enter NSE symbol (e.g., RELIANCE, TCS): ").upper()
from_date = input("Enter start date (YYYY-MM-DD): ")
to_date = input("Enter end date (YYYY-MM-DD): ")

# Fetch instrument token
instruments = pd.DataFrame(kite.instruments("NSE"))
if symbol not in instruments['tradingsymbol'].values:
    raise ValueError(f"Symbol {symbol} not found in NSE instruments.")
instrument_token = int(instruments.loc[instruments['tradingsymbol'] == symbol, 'instrument_token'].values[0])

# -----------------------------
# PARAMETERS
# -----------------------------
short_sma_period = 50
mid_ema_period = 100
long_sma_period = 200
swing_lookback = 5
volume_filter = True
macd_filter = True
initial_capital = 100000

# -----------------------------
# FETCH HISTORICAL DATA
# -----------------------------
data = kite.historical_data(instrument_token, from_date, to_date, interval="day")
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['date'])
df = df[['Date','open','high','low','close','volume']]
df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# CALCULATE INDICATORS
# -----------------------------
df['SMA50'] = df['Close'].rolling(short_sma_period).mean()
df['EMA100'] = df['Close'].ewm(span=mid_ema_period, adjust=False).mean()
df['SMA200'] = df['Close'].rolling(long_sma_period).mean()

if macd_filter:
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# -----------------------------
# GENERATE SIGNALS
# -----------------------------
df['Signal'] = 0
for i in range(long_sma_period, len(df)):
    buy_cond = df['SMA50'][i] > df['SMA200'][i] and df['SMA50'][i-1] <= df['SMA200'][i-1]
    sell_cond = df['SMA50'][i] < df['SMA200'][i] and df['SMA50'][i-1] >= df['SMA200'][i-1]

    if df['Close'][i] <= df['EMA100'][i]: buy_cond = False
    if df['Close'][i] >= df['EMA100'][i]: sell_cond = False

    if volume_filter and df['Volume'][i] < df['Volume'][i-1]: buy_cond = sell_cond = False
    if macd_filter:
        if df['MACD'][i] < df['MACD_signal'][i]: buy_cond = False
        if df['MACD'][i] > df['MACD_signal'][i]: sell_cond = False

    if buy_cond: df.loc[i,'Signal'] = 1
    elif sell_cond: df.loc[i,'Signal'] = -1

# -----------------------------
# BACKTEST
# -----------------------------
position = 0
entry_price = 0
stop_loss = 0
capital = initial_capital
trades = []

for i in range(long_sma_period, len(df)):
    swing_low = df['Low'][i-swing_lookback:i].min()
    swing_high = df['High'][i-swing_lookback:i].max()

    if position == 0:
        if df['Signal'][i] == 1:
            position = 1
            entry_price = df['Close'][i]
            stop_loss = swing_low
        elif df['Signal'][i] == -1:
            position = -1
            entry_price = df['Close'][i]
            stop_loss = swing_high
    elif position == 1:
        stop_loss = max(stop_loss, swing_low)
        if df['Close'][i] < stop_loss or df['Signal'][i] == -1:
            trade_pnl = df['Close'][i] - entry_price
            capital += trade_pnl
            trades.append({'Type':'Long','Entry':entry_price,'Exit':df['Close'][i],'PnL':trade_pnl})
            position = 0
    elif position == -1:
        stop_loss = min(stop_loss, swing_high)
        if df['Close'][i] > stop_loss or df['Signal'][i] == 1:
            trade_pnl = entry_price - df['Close'][i]
            capital += trade_pnl
            trades.append({'Type':'Short','Entry':entry_price,'Exit':df['Close'][i],'PnL':trade_pnl})
            position = 0

# -----------------------------
# RESULTS
# -----------------------------
total_trades = len(trades)
winning_trades = [t for t in trades if t['PnL']>0]
losing_trades = [t for t in trades if t['PnL']<=0]
win_rate = len(winning_trades)/total_trades*100 if total_trades>0 else 0
total_profit = sum(t['PnL'] for t in trades)

print("=== BACKTEST RESULTS ===")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {len(winning_trades)}")
print(f"Losing Trades: {len(losing_trades)}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total P&L: {total_profit:.2f}")
print(f"Final Capital: {capital:.2f}")

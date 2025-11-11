from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
import os

# -----------------------------
# KITE SETUP
# -----------------------------
api_key = "your_api_key"
api_secret = "your_api_secret"
access_token = "your_access_token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# -----------------------------
# TELEGRAM SETUP
# -----------------------------
TELEGRAM_TOKEN = "7714134495:AAHbFQujX8GEcPAkLg0c6_h4DDt1wzcWadc"
CHAT_ID = "541238511"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

# -----------------------------
# PARAMETERS
# -----------------------------
short_sma_period = 50
mid_ema_period = 100
long_sma_period = 200
swing_lookback = 5
volume_filter = True
macd_filter = True
scan_interval = 30 * 60  # 30 minutes

# -----------------------------
# GET NIFTY 500 SYMBOLS
# -----------------------------
# Replace with your actual Nifty500 list
nifty500_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']

instruments = pd.DataFrame(kite.instruments("NSE"))
instrument_tokens = {}
for sym in nifty500_symbols:
    token = instruments.loc[instruments['tradingsymbol'] == sym, 'instrument_token'].values
    if len(token) > 0:
        instrument_tokens[sym] = int(token[0])

# -----------------------------
# TRACK POSITIONS
# -----------------------------
positions = {sym: {'position': 0, 'entry': 0, 'sl': 0} for sym in instrument_tokens.keys()}

# -----------------------------
# CREATE SIGNAL LOG FILE
# -----------------------------
log_file = "swing_signals.csv"
if not os.path.exists(log_file):
    df_log = pd.DataFrame(columns=['Datetime', 'Symbol', 'Type', 'Price', 'SL'])
    df_log.to_csv(log_file, index=False)

# -----------------------------
# CALCULATE INDICATORS
# -----------------------------
def calculate_indicators(df):
    df['SMA50'] = df['Close'].rolling(short_sma_period).mean()
    df['EMA100'] = df['Close'].ewm(span=mid_ema_period, adjust=False).mean()
    df['SMA200'] = df['Close'].rolling(long_sma_period).mean()
    if macd_filter:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# -----------------------------
# SCAN FUNCTION
# -----------------------------
def scan_symbols():
    global positions
    print(f"\nScanning at {datetime.now()}...\n")
    for sym, token in instrument_tokens.items():
        try:
            # Fetch last 250 30-min candles
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=20)  # enough 30-min candles
            data = kite.historical_data(token, from_dt, to_dt, interval="30minute")
            df = pd.DataFrame(data)
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
            df = calculate_indicators(df)
            i = len(df) - 1

            swing_low = df['Low'][-swing_lookback:].min()
            swing_high = df['High'][-swing_lookback:].max()

            # Buy/Sell conditions
            buy_cond = df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and df['SMA50'].iloc[i-1] <= df['SMA200'].iloc[i-1]
            sell_cond = df['SMA50'].iloc[i] < df['SMA200'].iloc[i] and df['SMA50'].iloc[i-1] >= df['SMA200'].iloc[i-1]

            if df['Close'].iloc[i] <= df['EMA100'].iloc[i]: buy_cond = False
            if df['Close'].iloc[i] >= df['EMA100'].iloc[i]: sell_cond = False
            if volume_filter and df['Volume'].iloc[i] < df['Volume'].iloc[i-1]: buy_cond = sell_cond = False
            if macd_filter:
                if df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]: buy_cond = False
                if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]: sell_cond = False

            # Manage positions
            pos = positions[sym]
            if pos['position'] == 0:
                if buy_cond:
                    pos['position'] = 1
                    pos['entry'] = df['Close'].iloc[i]
                    pos['sl'] = swing_low
                    message = f"[LONG ENTRY] {sym} | Price: {pos['entry']:.2f} | SL: {pos['sl']:.2f}"
                    print(message)
                    send_telegram_message(message)
                    log_signal(sym, "LONG ENTRY", pos['entry'], pos['sl'])
                elif sell_cond:
                    pos['position'] = -1
                    pos['entry'] = df['Close'].iloc[i]
                    pos['sl'] = swing_high
                    message = f"[SHORT ENTRY] {sym} | Price: {pos['entry']:.2f} | SL: {pos['sl']:.2f}"
                    print(message)
                    send_telegram_message(message)
                    log_signal(sym, "SHORT ENTRY", pos['entry'], pos['sl'])
            elif pos['position'] == 1:
                pos['sl'] = max(pos['sl'], swing_low)
                if df['Close'].iloc[i] < pos['sl'] or sell_cond:
                    message = f"[LONG EXIT] {sym} | Price: {df['Close'].iloc[i]:.2f}"
                    print(message)
                    send_telegram_message(message)
                    log_signal(sym, "LONG EXIT", df['Close'].iloc[i], pos['sl'])
                    pos['position'] = 0
            elif pos['position'] == -1:
                pos['sl'] = min(pos['sl'], swing_high)
                if df['Close'].iloc[i] > pos['sl'] or buy_cond:
                    message = f"[SHORT EXIT] {sym} | Price: {df['Close'].iloc[i]:.2f}"
                    print(message)
                    send_telegram_message(message)
                    log_signal(sym, "SHORT EXIT", df['Close'].iloc[i], pos['sl'])
                    pos['position'] = 0

        except Exception as e:
            print(f"Error scanning {sym}: {e}")

# -----------------------------
# LOG SIGNAL TO CSV
# -----------------------------
def log_signal(symbol, trade_type, price, sl):
    df_log = pd.read_csv(log_file)
    new_row = {'Datetime': datetime.now(), 'Symbol': symbol, 'Type': trade_type, 'Price': price, 'SL': sl}
    df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
    df_log.to_csv(log_file, index=False)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    scan_symbols()
    print(f"Waiting for {scan_interval//60} minutes for next scan...\n")
    time.sleep(scan_interval)

# Indicators by Market Segment and Category

This document provides a comprehensive breakdown of all indicators calculated for different market segments (Equity, Futures, Options, Index) organized by indicator categories.

---

## 📊 **EQUITY** (Stocks)

### **Trend Indicators**
- ✅ **EMA (Exponential Moving Average)**: ema_9, ema_21, ema_50, ema_200
- ✅ **SMA (Simple Moving Average)**: sma_20, sma_50
- ✅ **Supertrend**: supertrend_7_3, supertrend_10_3, supertrend_signal_7_3, supertrend_signal_10_3
- ✅ **Parabolic SAR**: parabolic_sar

### **Momentum Indicators**
- ✅ **RSI (Relative Strength Index)**: rsi_14
- ✅ **MACD**: macd_line, macd_signal, macd_histogram
- ✅ **Stochastic Oscillator**: stoch_k, stoch_d
- ✅ **Williams %R**: williams_r
- ✅ **CCI (Commodity Channel Index)**: cci_20
- ✅ **ADX (Average Directional Index)**: adx_14, di_plus, di_minus

### **Volatility Indicators**
- ✅ **ATR (Average True Range)**: atr_14
- ✅ **Bollinger Bands**: bb_upper, bb_middle, bb_lower, bb_width, bb_percent

### **Volume Indicators**
- ✅ **Volume SMA**: volume_sma_20
- ✅ **VWAP (Volume Weighted Average Price)**: vwap
- ✅ **OBV (On Balance Volume)**: obv
- ✅ **MFI (Money Flow Index)**: mfi_14

### **Price Action Indicators**
- ✅ **Pivot Points**: pivot_point, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3

### **Market Microstructure** (5 levels)
- ✅ **Bid/Ask Levels**: bid_1 to bid_5, ask_1 to ask_5
- ✅ **Bid/Ask Quantities**: bid_qty_1 to bid_qty_5, ask_qty_1 to ask_qty_5
- ✅ **Derived Metrics**: bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty

### **Change Metrics**
- ✅ **Price Change**: price_change, price_change_pct
- ✅ **High-Low Percentage**: high_low_pct

---

## 📈 **FUTURES** (F&O Contracts)

### **Trend Indicators** (Same as Equity)
- ✅ **EMA**: ema_9, ema_21, ema_50, ema_200
- ✅ **SMA**: sma_20, sma_50
- ✅ **Supertrend**: supertrend_7_3, supertrend_10_3, supertrend_signal_7_3, supertrend_signal_10_3

### **Momentum Indicators** (Same as Equity)
- ✅ **RSI**: rsi_14
- ✅ **MACD**: macd_line, macd_signal, macd_histogram
- ✅ **Stochastic**: stoch_k, stoch_d

### **Volatility Indicators** (Same as Equity)
- ✅ **ATR**: atr_14
- ✅ **Bollinger Bands**: bb_upper, bb_middle, bb_lower, bb_width

### **Volume Indicators** (Same as Equity)
- ✅ **VWAP**: vwap
- ✅ **OBV**: obv

### **Futures-Specific Indicators**
- ✅ **Open Interest Change**: oi_change, oi_change_pct
- ✅ **Volume/OI Ratio**: volume_oi_ratio
- ✅ **Price-OI Correlation**: price_oi_correlation
- ✅ **Volume Change**: volume_change, volume_change_pct

### **Basis and Spread Analysis**
- ✅ **Spot Price**: spot_price (if available)
- ✅ **Basis**: basis (Futures - Spot)
- ✅ **Basis Percentage**: basis_pct
- ✅ **Cost of Carry**: cost_of_carry

### **Market Microstructure** (Same as Equity)
- ✅ **Bid/Ask Levels**: bid_1 to bid_5, ask_1 to ask_5
- ✅ **Bid/Ask Quantities**: bid_qty_1 to bid_qty_5, ask_qty_1 to ask_qty_5
- ✅ **Derived Metrics**: bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty

### **Change Metrics** (Same as Equity)
- ✅ **Price Change**: price_change, price_change_pct

---

## 🎯 **OPTIONS** (Call & Put Options)

### **Options Greeks (Primary)**
- ✅ **Delta**: Price sensitivity to underlying
- ✅ **Gamma**: Delta sensitivity
- ✅ **Theta**: Time decay
- ✅ **Vega**: Volatility sensitivity
- ✅ **Rho**: Interest rate sensitivity

### **Volatility Metrics**
- ✅ **Implied Volatility (IV)**: implied_volatility
- ✅ **Historical Volatility (HV)**: historical_volatility (20-day)
- ✅ **IV Rank**: iv_rank (0-100)
- ✅ **IV Percentile**: iv_percentile

### **Value Components**
- ✅ **Intrinsic Value**: intrinsic_value
- ✅ **Time Value**: time_value
- ✅ **Moneyness**: moneyness (S/K for calls, K/S for puts)

### **Advanced Greeks**
- ✅ **Lambda (Leverage)**: lambda_greek (Delta * S / Premium)
- ✅ **Epsilon**: Dividend sensitivity
- ✅ **Vera**: Volatility elasticity
- ✅ **Charm**: Delta decay (if advanced enabled)
- ✅ **Vanna**: Delta-Volatility sensitivity (if advanced enabled)
- ✅ **Volga**: Volatility-Gamma sensitivity (if advanced enabled)

### **Risk Metrics**
- ✅ **Probability ITM**: probability_itm
- ✅ **Probability of Profit**: probability_profit
- ✅ **Max Pain**: max_pain

### **Technical Indicators** (Limited - for Premium Analysis)
- ✅ **RSI**: rsi_14
- ✅ **EMA**: ema_9, ema_21
- ✅ **ATR**: atr_14
- ✅ **Bollinger Bands**: bb_upper, bb_lower

### **Options-Specific Market Data**
- ✅ **Put/Call Ratio**: put_call_ratio
- ✅ **Max Pain Distance**: max_pain_distance
- ✅ **Volatility Skew**: skew

### **Market Microstructure** (Same as Equity)
- ✅ **Bid/Ask Levels**: bid_1 to bid_5, ask_1 to ask_5
- ✅ **Bid/Ask Quantities**: bid_qty_1 to bid_qty_5, ask_qty_1 to ask_qty_5
- ✅ **Derived Metrics**: bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty

### **Change Metrics**
- ✅ **Price Change**: price_change, price_change_pct
- ✅ **OI Change**: oi_change, oi_change_pct
- ✅ **IV Change**: iv_change
- ✅ **Delta Change**: delta_change

---

## 📉 **INDEX** (Market Indices - NIFTY, BANKNIFTY, etc.)

### **Trend Indicators**
- ✅ **EMA**: ema_9, ema_21, ema_50, ema_200
- ✅ **SMA**: sma_20, sma_50, sma_100 (INDEX-specific)
- ✅ **Supertrend**: supertrend_7_3, supertrend_10_3, supertrend_signal_7_3, supertrend_signal_10_3
- ✅ **Parabolic SAR**: parabolic_sar

### **Momentum Indicators**
- ✅ **RSI**: rsi_14
- ✅ **MACD**: macd_line, macd_signal, macd_histogram
- ✅ **Stochastic**: stoch_k, stoch_d

### **Volatility Indicators**
- ✅ **ATR**: atr_14
- ✅ **Bollinger Bands**: bb_upper, bb_middle, bb_lower, bb_width, bb_percent

### **Price Action Indicators**
- ✅ **Pivot Points**: pivot_point, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3

### **Index-Specific Indicators** (Optional - may not be available)
- ✅ **Advance/Decline Ratio**: advance_decline_ratio
- ✅ **High/Low Index**: high_low_index
- ✅ **McClellan Oscillator**: mcclellan_oscillator

### **Volatility Measures** (Optional - may not be available)
- ✅ **Realized Volatility**: realized_volatility (20-day)
- ✅ **GARCH Volatility**: garch_volatility (forecast)

### **Market Microstructure** (If available from API)
- ✅ **Bid/Ask Levels**: bid_1 to bid_5, ask_1 to ask_5
- ✅ **Bid/Ask Quantities**: bid_qty_1 to bid_qty_5, ask_qty_1 to ask_qty_5
- ✅ **Derived Metrics**: bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty

### **Change Metrics**
- ✅ **Price Change**: price_change, price_change_pct
- ✅ **High-Low Percentage**: high_low_pct

### **❌ NOT Available for INDEX**
- ❌ **Volume Indicators**: No volume data for indices (no volume_sma_20, vwap, obv)

---

## 📋 **Summary by Category**

### **Trend Indicators**
- **Equity**: ✅ Full set (EMA 9/21/50/200, SMA 20/50, Supertrend, Parabolic SAR)
- **Futures**: ✅ Full set (same as Equity)
- **Options**: ❌ Not calculated (focus on Greeks)
- **Index**: ✅ Full set (EMA 9/21/50/200, SMA 20/50/100, Supertrend, Parabolic SAR)

### **Momentum Indicators**
- **Equity**: ✅ Full set (RSI, MACD, Stochastic, Williams %R, CCI, ADX)
- **Futures**: ✅ Full set (RSI, MACD, Stochastic)
- **Options**: ✅ Limited (RSI only - for premium analysis)
- **Index**: ✅ Full set (RSI, MACD, Stochastic)

### **Volatility Indicators**
- **Equity**: ✅ Full set (ATR, Bollinger Bands)
- **Futures**: ✅ Full set (ATR, Bollinger Bands)
- **Options**: ✅ Limited (ATR, Bollinger Bands - for premium analysis)
- **Index**: ✅ Full set (ATR, Bollinger Bands)

### **Volume Indicators**
- **Equity**: ✅ Full set (Volume SMA, VWAP, OBV, MFI)
- **Futures**: ✅ Limited (VWAP, OBV)
- **Options**: ❌ Not calculated
- **Index**: ❌ Not available (indices don't have volume)

### **Options Greeks**
- **Equity**: ❌ Not applicable
- **Futures**: ❌ Not applicable
- **Options**: ✅ Full set (Delta, Gamma, Theta, Vega, Rho, Lambda, Epsilon, Vera, Charm, Vanna, Volga)
- **Index**: ❌ Not applicable

### **Market Microstructure**
- **Equity**: ✅ Full set (5 levels bid/ask + derived metrics)
- **Futures**: ✅ Full set (5 levels bid/ask + derived metrics)
- **Options**: ✅ Full set (5 levels bid/ask + derived metrics)
- **Index**: ✅ If available from API (5 levels bid/ask + derived metrics)

### **Price Action Indicators**
- **Equity**: ✅ Full set (Pivot Points + Support/Resistance)
- **Futures**: ❌ Not calculated
- **Options**: ❌ Not calculated
- **Index**: ✅ Full set (Pivot Points + Support/Resistance)

### **Segment-Specific Indicators**
- **Equity**: None
- **Futures**: ✅ OI metrics, Basis analysis, Cost of carry
- **Options**: ✅ IV metrics, Value components, Risk metrics, Options-specific market data
- **Index**: ✅ Market breadth indicators (A/D ratio, High/Low index, McClellan), Volatility measures

---

## 🔧 **Calculation Configuration**

All indicators are controlled by `CalculationConfig` in `data_models.py`:

- `calculate_trend_indicators`: Controls EMA, SMA
- `calculate_momentum_indicators`: Controls RSI, MACD, Stochastic
- `calculate_volatility_indicators`: Controls ATR, Bollinger Bands
- `calculate_volume_indicators`: Controls VWAP, OBV (disabled for INDEX)
- `calculate_greeks`: Controls Options Greeks (Options only)
- `calculate_iv`: Controls Implied Volatility calculation (Options only)
- `calculate_advanced_greeks`: Controls advanced Greeks (Options only)

---

## 📝 **Notes**

1. **Volume-based indicators** are automatically excluded for INDEX instruments as indices don't have volume data.

2. **Options indicators** focus primarily on Greeks and volatility metrics, with limited technical indicators for premium analysis.

3. **Futures indicators** include all equity indicators plus futures-specific metrics like OI changes and basis analysis.

4. **Market microstructure** data (bid/ask levels) is included for all segments if available from the API.

5. **Index-specific indicators** (A/D ratio, McClellan oscillator) are optional and may not always be available.


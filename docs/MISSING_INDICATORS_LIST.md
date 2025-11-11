# Missing Indicators - Comprehensive List by Market Segment

This document lists all missing indicators that should be added, organized by category and market segment (Equity, Futures, Options, Index).

---

## 🔴 **CRITICAL MISSING INDICATORS** (High Priority)

### **1. Volume Divergence Indicators** ⭐ CRUCIAL
**Status**: ❌ NOT IMPLEMENTED  
**Applicable to**: Equity, Futures, Options

#### **Volume-Price Divergence**
- `volume_price_divergence` (BYTE): 
  - 1 = Bullish divergence (price down, volume up)
  - 2 = Bearish divergence (price up, volume down)
  - 0 = No divergence
- `volume_divergence_strength` (DOUBLE): Strength of divergence (0-100)
- `volume_divergence_confirmed` (BYTE): Whether divergence is confirmed (1=Yes, 0=No)

#### **RSI-Volume Divergence**
- `rsi_volume_divergence` (BYTE): Divergence between RSI and volume
- `rsi_volume_divergence_signal` (BYTE): Buy/Sell signal from divergence

#### **MACD-Volume Divergence**
- `macd_volume_divergence` (BYTE): Divergence between MACD and volume
- `macd_volume_divergence_signal` (BYTE): Buy/Sell signal

#### **Price-Volume Divergence Detection**
- `price_volume_divergence_type` (BYTE): Type of divergence detected
- `divergence_period` (INT): Period over which divergence is detected

---

### **2. Ichimoku Cloud** ⭐ VERY POPULAR
**Status**: ❌ NOT IMPLEMENTED  
**Applicable to**: Equity, Futures, Index

- `ichimoku_tenkan_sen` (DOUBLE): Tenkan-sen (Conversion Line) - 9-period
- `ichimoku_kijun_sen` (DOUBLE): Kijun-sen (Base Line) - 26-period
- `ichimoku_senkou_span_a` (DOUBLE): Senkou Span A (Leading Span A)
- `ichimoku_senkou_span_b` (DOUBLE): Senkou Span B (Leading Span B) - 52-period
- `ichimoku_chikou_span` (DOUBLE): Chikou Span (Lagging Span)
- `ichimoku_cloud_top` (DOUBLE): Top of cloud (max of Span A and B)
- `ichimoku_cloud_bottom` (DOUBLE): Bottom of cloud (min of Span A and B)
- `ichimoku_cloud_color` (BYTE): 1=Green (bullish), 0=Red (bearish)
- `ichimoku_signal` (BYTE): 1=Buy, 0=Sell, 2=Neutral

---

### **3. Aroon Indicator** ⭐ MENTIONED IN SCHEMA BUT NOT IMPLEMENTED
**Status**: ❌ NOT IMPLEMENTED  
**Applicable to**: Equity, Futures, Index

- `aroon_up` (DOUBLE): Aroon Up (0-100)
- `aroon_down` (DOUBLE): Aroon Down (0-100)
- `aroon_oscillator` (DOUBLE): Aroon Oscillator (Up - Down)
- `aroon_signal` (BYTE): 1=Strong uptrend, -1=Strong downtrend, 0=Neutral

---

### **4. Volume Profile Indicators** ⭐ CRUCIAL FOR VOLUME ANALYSIS
**Status**: ❌ NOT IMPLEMENTED  
**Applicable to**: Equity, Futures, Options

#### **Volume Profile Metrics**
- `volume_profile_poc` (DOUBLE): Point of Control (price with highest volume)
- `volume_profile_vah` (DOUBLE): Value Area High (70% volume area top)
- `volume_profile_val` (DOUBLE): Value Area Low (70% volume area bottom)
- `volume_profile_balance` (DOUBLE): Volume-weighted balance point
- `volume_profile_distribution` (DOUBLE): Distribution of volume across price levels

#### **Volume Rate & Ratios**
- `volume_rate` (DOUBLE): Current volume / Average volume (already in schema but not calculated)
- `volume_ratio` (DOUBLE): Current volume / Previous volume
- `volume_trend` (BYTE): 1=Increasing, 0=Decreasing
- `volume_surge` (BYTE): 1=Volume surge detected (>2x average), 0=Normal

---

### **5. Accumulation/Distribution Line** ⭐ IMPORTANT VOLUME INDICATOR
**Status**: ❌ NOT IMPLEMENTED (mentioned in schema)  
**Applicable to**: Equity, Futures

- `ad_line` (DOUBLE): Accumulation/Distribution Line
- `ad_line_slope` (DOUBLE): Slope of A/D line
- `ad_line_signal` (BYTE): 1=Accumulation, -1=Distribution, 0=Neutral

---

### **6. Chaikin Money Flow (CMF)** ⭐ IMPORTANT
**Status**: ❌ NOT IMPLEMENTED (mentioned in schema)  
**Applicable to**: Equity, Futures

- `cmf_20` (DOUBLE): 20-period Chaikin Money Flow
- `cmf_signal` (BYTE): 1=Buy, -1=Sell, 0=Neutral

---

### **7. Time Weighted Average Price (TWAP)** ⭐ IMPORTANT
**Status**: ❌ NOT IMPLEMENTED (mentioned in schema)  
**Applicable to**: Equity, Futures, Options

- `twap` (DOUBLE): Time Weighted Average Price

---

## 🟡 **IMPORTANT MISSING INDICATORS** (Medium Priority)

### **8. Additional Momentum Indicators**

#### **Rate of Change (ROC)**
- `roc_10` (DOUBLE): 10-period Rate of Change
- `roc_14` (DOUBLE): 14-period Rate of Change
- `roc_20` (DOUBLE): 20-period Rate of Change

#### **Momentum Oscillator**
- `momentum_10` (DOUBLE): 10-period Momentum
- `momentum_14` (DOUBLE): 14-period Momentum

#### **Ultimate Oscillator**
- `ultimate_oscillator` (DOUBLE): Ultimate Oscillator (7, 14, 28 periods)

#### **Awesome Oscillator**
- `awesome_oscillator` (DOUBLE): Awesome Oscillator (5, 34 periods)

---

### **9. Additional Trend Indicators**

#### **Hull Moving Average (HMA)**
- `hma_9` (DOUBLE): 9-period Hull Moving Average
- `hma_21` (DOUBLE): 21-period Hull Moving Average
- `hma_50` (DOUBLE): 50-period Hull Moving Average

#### **TEMA (Triple Exponential Moving Average)**
- `tema_9` (DOUBLE): 9-period TEMA
- `tema_21` (DOUBLE): 21-period TEMA

#### **DEMA (Double Exponential Moving Average)**
- `dema_9` (DOUBLE): 9-period DEMA
- `dema_21` (DOUBLE): 21-period DEMA

#### **Kaufman Adaptive Moving Average (KAMA)**
- `kama_10` (DOUBLE): 10-period KAMA
- `kama_21` (DOUBLE): 21-period KAMA

#### **ZigZag Indicator**
- `zigzag_high` (DOUBLE): ZigZag high points
- `zigzag_low` (DOUBLE): ZigZag low points
- `zigzag_trend` (BYTE): Current trend direction

---

### **10. Additional Volatility Indicators**

#### **Keltner Channels**
- `keltner_upper` (DOUBLE): Keltner Channel Upper
- `keltner_middle` (DOUBLE): Keltner Channel Middle (EMA)
- `keltner_lower` (DOUBLE): Keltner Channel Lower
- `keltner_width` (DOUBLE): Channel width

#### **Donchian Channels**
- `donchian_upper` (DOUBLE): Donchian Channel Upper (20-period high)
- `donchian_middle` (DOUBLE): Donchian Channel Middle
- `donchian_lower` (DOUBLE): Donchian Channel Lower (20-period low)
- `donchian_width` (DOUBLE): Channel width

#### **Standard Deviation**
- `std_dev_20` (DOUBLE): 20-period Standard Deviation
- `std_dev_50` (DOUBLE): 50-period Standard Deviation

#### **Volatility Ratio**
- `volatility_ratio` (DOUBLE): Current volatility / Historical volatility

---

### **11. Volume-Based Indicators (Additional)**

#### **Volume EMA**
- `volume_ema_20` (DOUBLE): 20-period Volume EMA (mentioned in schema but not calculated)

#### **Volume Rate of Change**
- `volume_roc` (DOUBLE): Volume Rate of Change

#### **Volume Oscillator**
- `volume_oscillator` (DOUBLE): Volume Oscillator (fast MA - slow MA)

#### **Ease of Movement (EOM)**
- `eom_14` (DOUBLE): 14-period Ease of Movement

#### **Negative Volume Index (NVI)**
- `nvi` (DOUBLE): Negative Volume Index

#### **Positive Volume Index (PVI)**
- `pvi` (DOUBLE): Positive Volume Index

#### **Price Volume Trend (PVT)**
- `pvt` (DOUBLE): Price Volume Trend

---

### **12. Support/Resistance Detection**

#### **Dynamic Support/Resistance**
- `dynamic_support` (DOUBLE): Dynamic support level
- `dynamic_resistance` (DOUBLE): Dynamic resistance level
- `support_strength` (DOUBLE): Strength of support (0-100)
- `resistance_strength` (DOUBLE): Strength of resistance (0-100)

#### **Fibonacci Retracements**
- `fib_23_6` (DOUBLE): Fibonacci 23.6% level
- `fib_38_2` (DOUBLE): Fibonacci 38.2% level
- `fib_50_0` (DOUBLE): Fibonacci 50.0% level
- `fib_61_8` (DOUBLE): Fibonacci 61.8% level
- `fib_78_6` (DOUBLE): Fibonacci 78.6% level

#### **Fibonacci Extensions**
- `fib_ext_127_2` (DOUBLE): Fibonacci 127.2% extension
- `fib_ext_161_8` (DOUBLE): Fibonacci 161.8% extension
- `fib_ext_200_0` (DOUBLE): Fibonacci 200.0% extension

---

### **13. Market Structure Indicators**

#### **Higher Highs / Lower Lows**
- `higher_high` (BYTE): 1=Higher high detected, 0=No
- `lower_low` (BYTE): 1=Lower low detected, 0=No
- `market_structure` (BYTE): 1=Uptrend, -1=Downtrend, 0=Sideways

#### **Swing Highs/Lows**
- `swing_high` (DOUBLE): Recent swing high
- `swing_low` (DOUBLE): Recent swing low
- `swing_high_count` (INT): Count of swing highs
- `swing_low_count` (INT): Count of swing lows

#### **Breakout Detection**
- `breakout_up` (BYTE): 1=Upward breakout detected, 0=No
- `breakout_down` (BYTE): 1=Downward breakout detected, 0=No
- `breakout_strength` (DOUBLE): Strength of breakout

---

### **14. Order Flow Indicators** (For Futures/Options)

#### **Cumulative Volume Delta (CVD)**
- `cvd` (DOUBLE): Cumulative Volume Delta
- `cvd_trend` (BYTE): Trend of CVD

#### **Volume Delta**
- `volume_delta` (DOUBLE): Buy volume - Sell volume
- `volume_delta_pct` (DOUBLE): Volume delta as % of total volume

#### **Order Flow Imbalance**
- `order_flow_imbalance` (DOUBLE): Order flow imbalance ratio

---

## 🟢 **NICE-TO-HAVE INDICATORS** (Low Priority)

### **15. Advanced Oscillators**

#### **TRIX Indicator**
- `trix_14` (DOUBLE): 14-period TRIX

#### **Detrended Price Oscillator (DPO)**
- `dpo_20` (DOUBLE): 20-period DPO

#### **Percentage Price Oscillator (PPO)**
- `ppo` (DOUBLE): Percentage Price Oscillator

#### **Mass Index**
- `mass_index` (DOUBLE): Mass Index

---

### **16. Market Breadth Indicators** (For Index)

#### **Advance/Decline Line** (Already mentioned but not calculated)
- `ad_line_index` (DOUBLE): Advance/Decline Line for index

#### **New Highs/New Lows**
- `new_highs_52w` (INT): 52-week new highs count
- `new_lows_52w` (INT): 52-week new lows count
- `new_highs_lows_ratio` (DOUBLE): Ratio of new highs to new lows

#### **Percent of Stocks Above Moving Average**
- `pct_above_sma_20` (DOUBLE): % of stocks above 20 SMA
- `pct_above_sma_50` (DOUBLE): % of stocks above 50 SMA
- `pct_above_sma_200` (DOUBLE): % of stocks above 200 SMA

---

### **17. Options-Specific Additional Indicators**

#### **Put/Call Ratio (Volume)**
- `put_call_volume_ratio` (DOUBLE): Put/Call volume ratio (already have but can enhance)

#### **Open Interest Put/Call Ratio**
- `put_call_oi_ratio` (DOUBLE): Put/Call OI ratio

#### **Options Flow Metrics**
- `unusual_options_activity` (BYTE): 1=Unusual activity detected, 0=Normal
- `options_flow_bullish` (DOUBLE): Bullish options flow
- `options_flow_bearish` (DOUBLE): Bearish options flow

#### **Greeks Changes**
- `delta_change` (DOUBLE): Change in delta (already have)
- `gamma_change` (DOUBLE): Change in gamma
- `theta_change` (DOUBLE): Change in theta
- `vega_change` (DOUBLE): Change in vega

---

### **18. Futures-Specific Additional Indicators**

#### **Contango/Backwardation**
- `contango_backwardation` (BYTE): 1=Contango, -1=Backwardation, 0=Neutral
- `spread_to_spot` (DOUBLE): Futures spread to spot

#### **Open Interest Analysis**
- `oi_breakdown_long` (DOUBLE): Estimated long OI
- `oi_breakdown_short` (DOUBLE): Estimated short OI
- `oi_concentration` (DOUBLE): OI concentration ratio

---

### **19. Pattern Recognition**

#### **Candlestick Patterns**
- `candlestick_pattern` (SYMBOL): Detected candlestick pattern name
- `pattern_strength` (DOUBLE): Pattern strength (0-100)
- `pattern_signal` (BYTE): 1=Bullish, -1=Bearish, 0=Neutral

#### **Chart Patterns**
- `chart_pattern` (SYMBOL): Detected chart pattern (Head & Shoulders, etc.)
- `pattern_target` (DOUBLE): Pattern target price

---

### **20. Statistical Indicators**

#### **Z-Score**
- `z_score_20` (DOUBLE): 20-period Z-Score
- `z_score_50` (DOUBLE): 50-period Z-Score

#### **Percentile Rank**
- `percentile_rank_20` (DOUBLE): 20-period Percentile Rank
- `percentile_rank_50` (DOUBLE): 50-period Percentile Rank

#### **Linear Regression**
- `linear_regression_slope` (DOUBLE): Linear regression slope
- `linear_regression_r2` (DOUBLE): R-squared value

---

## 📊 **SUMMARY BY MARKET SEGMENT**

### **EQUITY** - Missing Indicators (Priority Order)

#### **Critical (Must Have)**
1. ✅ Volume Divergence (all types)
2. ✅ Ichimoku Cloud
3. ✅ Aroon Indicator
4. ✅ Volume Profile (POC, VAH, VAL)
5. ✅ Accumulation/Distribution Line
6. ✅ Chaikin Money Flow
7. ✅ TWAP

#### **Important (Should Have)**
8. ✅ Rate of Change (ROC)
9. ✅ Hull Moving Average (HMA)
10. ✅ Keltner Channels
11. ✅ Donchian Channels
12. ✅ Volume EMA
13. ✅ Volume Oscillator
14. ✅ Dynamic Support/Resistance
15. ✅ Market Structure (HH/LL)

#### **Nice to Have**
16. ✅ TRIX
17. ✅ Ultimate Oscillator
18. ✅ ZigZag
19. ✅ Candlestick Patterns
20. ✅ Z-Score

---

### **FUTURES** - Missing Indicators (Priority Order)

#### **Critical (Must Have)**
1. ✅ Volume Divergence
2. ✅ Ichimoku Cloud
3. ✅ Aroon Indicator
4. ✅ Volume Profile
5. ✅ Accumulation/Distribution Line
6. ✅ Chaikin Money Flow
7. ✅ TWAP
8. ✅ Cumulative Volume Delta (CVD)
9. ✅ Order Flow Imbalance

#### **Important (Should Have)**
10. ✅ Contango/Backwardation detection
11. ✅ OI Breakdown (Long/Short)
12. ✅ Rate of Change
13. ✅ Hull Moving Average
14. ✅ Keltner Channels
15. ✅ Market Structure

#### **Nice to Have**
16. ✅ Volume Delta
17. ✅ TRIX
18. ✅ ZigZag

---

### **OPTIONS** - Missing Indicators (Priority Order)

#### **Critical (Must Have)**
1. ✅ Volume Divergence (for premium analysis)
2. ✅ TWAP
3. ✅ Put/Call OI Ratio
4. ✅ Options Flow Metrics
5. ✅ Greeks Changes (Gamma, Theta, Vega)

#### **Important (Should Have)**
6. ✅ Unusual Options Activity detection
7. ✅ Volume Profile (for premium)
8. ✅ Rate of Change (for premium)

#### **Nice to Have**
9. ✅ Pattern Recognition (for premium)

---

### **INDEX** - Missing Indicators (Priority Order)

#### **Critical (Must Have)**
1. ✅ Ichimoku Cloud
2. ✅ Aroon Indicator
3. ✅ Market Breadth Indicators (A/D Line, New Highs/Lows)
4. ✅ Percent of Stocks Above MA
5. ✅ Advance/Decline Ratio (enhancement)

#### **Important (Should Have)**
6. ✅ Rate of Change
7. ✅ Hull Moving Average
8. ✅ Keltner Channels
9. ✅ Market Structure (HH/LL)
10. ✅ Dynamic Support/Resistance

#### **Nice to Have**
11. ✅ TRIX
12. ✅ ZigZag
13. ✅ Z-Score

---

## 🎯 **IMPLEMENTATION PRIORITY**

### **Phase 1: Critical Indicators** (Implement First)
1. Volume Divergence (all types) - **HIGHEST PRIORITY**
2. Ichimoku Cloud
3. Aroon Indicator
4. Volume Profile (POC, VAH, VAL)
5. Accumulation/Distribution Line
6. Chaikin Money Flow
7. TWAP

### **Phase 2: Important Indicators**
8. Rate of Change (ROC)
9. Hull Moving Average (HMA)
10. Keltner Channels
11. Donchian Channels
12. Volume EMA
13. Volume Oscillator
14. Dynamic Support/Resistance
15. Market Structure (HH/LL)
16. Cumulative Volume Delta (for Futures)

### **Phase 3: Nice-to-Have Indicators**
17. All remaining indicators from the list

---

## 📝 **NOTES**

1. **Volume Divergence** is the most critical missing indicator - it's essential for confirming price movements and identifying potential reversals.

2. **Ichimoku Cloud** is extremely popular in trading and provides comprehensive trend analysis.

3. **Aroon Indicator** is already mentioned in the schema but not implemented - should be prioritized.

4. **Volume Profile** is crucial for understanding price levels with high trading activity.

5. **Market Structure** indicators help identify trend changes and are essential for systematic trading.

6. Many indicators mentioned in `IndicatorColumnDefinitions` are not yet implemented - these should be prioritized.

7. Options and Futures have unique requirements that should be addressed with segment-specific indicators.



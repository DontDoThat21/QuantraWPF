# TFT Requirements Checklist for Quantra

## Summary: Current Status

| Component | Training | Inference | Status |
|-----------|----------|-----------|--------|
| **Lookback Window** | ? Real sequences (60 days) | ? Synthetic (repeated values) | **NEEDS FIX** |
| **Known Future Inputs** | ? Implemented (13 features) | ? Not passed | **NEEDS FIX** |
| **Static Covariates** | ?? Partial (basic features) | ?? Partial | **NEEDS ENHANCEMENT** |
| **Multi-Horizon Targets** | ? Implemented | ? Working | ? **GOOD** |
| **OHLCV Data** | ? From database | ? From cache | ? **GOOD** |

---

## ? Critical Issues to Fix

### **1. Synthetic Lookback Window in Inference**

**Problem:**
```python
# tft_integration.py line 256 - CURRENT (WRONG)
X_past = np.tile(feature_values.reshape(1, 1, n_features), (1, lookback, 1))
# Creates: [100, 100, 100, ..., 100] (60 repeated values)
```

**Required Fix:**
```python
# CORRECT - Pass actual historical sequence
X_past = historical_sequence_array  # Shape: (1, 60, n_features)
# Contains: [98.0, 98.2, 98.5, ..., 100.0] (real historical values)
```

**Implementation Steps:**

1. **Modify C# `RealTimeInferenceService`:**
```csharp
// Add new property to pass historical data
public class PredictionRequest {
    public Dictionary<string, double> CurrentFeatures { get; set; }
    public List<Dictionary<string, double>> HistoricalSequence { get; set; } // NEW
    public Dictionary<string, string> StaticFeatures { get; set; }
}
```

2. **Update `StockDataCacheService` to provide historical sequences:**
```csharp
public List<HistoricalPrice> GetRecentHistory(string symbol, int days = 60)
{
    // Fetch last 60 days of OHLCV data for the symbol
    return GetCachedPrices(symbol)
        .OrderByDescending(p => p.Date)
        .Take(days)
        .Reverse()  // Oldest first
        .ToList();
}
```

3. **Update `tft_integration.py` predict_single():**
```python
def predict_single(self, 
                  historical_sequence: List[Dict[str, float]],  # NEW parameter
                  static_dict: Optional[Dict[str, Any]] = None,
                  lookback: int = 60):
    """
    Args:
        historical_sequence: List of dicts with OHLCV data for last 60 days
                            [{'close': 98.0, 'volume': 1M, ...}, ..., {'close': 100.0, ...}]
    """
    # Convert list of dicts to (1, lookback, features) array
    feature_matrix = np.array([[list(d.values()) for d in historical_sequence]])
    X_past = feature_matrix.astype(np.float32)  # Shape: (1, 60, n_features)
```

---

### **2. Known Future Inputs Not Passed to TFT**

**Problem:** Training data now includes calendar features, but inference doesn't use them.

**Training (FIXED):**
```python
# train_from_database.py - NOW INCLUDES:
X_future_train  # Shape: (n_samples, 65, 12) - calendar features for 60 past + 5 future days
```

**Inference (NEEDS FIX):**
```python
# tft_integration.py predict_single() - CURRENTLY MISSING
# Need to generate calendar features for prediction date + horizon
```

**Required Fix:**
```python
def predict_single(self, historical_sequence, static_dict, target_date=None):
    # Generate calendar features for historical window + future horizon
    if target_date is None:
        target_date = datetime.now()
    
    # Create date range for past 60 days + future 30 days
    date_range = pd.date_range(
        start=target_date - timedelta(days=60),
        end=target_date + timedelta(days=30),
        freq='D'
    )
    
    # Generate calendar features
    calendar_features = self._generate_calendar_features(date_range)
    # Shape: (90, 12) - calendar features for entire window
    
    # Pass to TFT model
    outputs = self.predict(X_past, X_static, X_future=calendar_features)
```

---

## ? What's Currently Working

### **1. Training Pipeline Enhanced (train_from_database.py)**

? **Added Known Future Features:**
```python
def add_known_future_features(df):
    # Calendar features (13 total):
    - dayofweek (0-6)
    - day (1-31)
    - month (1-12)
    - quarter (1-4)
    - year
    - is_month_end (0/1)
    - is_quarter_end (0/1)
    - is_year_end (0/1)
    - is_month_start (0/1)
    - is_friday (0/1)
    - is_monday (0/1)
    - is_potential_holiday_week (0/1)
```

? **Proper Temporal Sequences:**
```python
window_size=60  # Increased from 20 to 60 days (TFT minimum)
target_days=5   # Multi-horizon targets: [5, 10, 20, 30] days
```

### **2. Multi-Horizon Predictions (Working)**

? TFT model outputs predictions for multiple horizons:
```python
horizons = [5, 10, 20, 30]  # Days ahead
predictions = {
    'horizon_5': [q10, q25, median, q75, q90],
    'horizon_10': [...],
    'horizon_20': [...],
    'horizon_30': [...]
}
```

---

## ?? Additional Missing Features from Alpha Vantage

### **Static Covariates to Add**

Alpha Vantage `OVERVIEW` endpoint provides these (currently not used):

| Feature | Endpoint Field | Implementation |
|---------|---------------|----------------|
| **52-Week High** | `52WeekHigh` | `static_dict['52_week_high'] = float(overview['52WeekHigh'])` |
| **52-Week Low** | `52WeekLow` | `static_dict['52_week_low'] = float(overview['52WeekLow'])` |
| **Dividend Yield** | `DividendYield` | `static_dict['dividend_yield'] = float(overview['DividendYield'])` |
| **Profit Margin** | `ProfitMargin` | `static_dict['profit_margin'] = float(overview['ProfitMargin'])` |
| **Operating Margin** | `OperatingMarginTTM` | `static_dict['operating_margin'] = float(overview['OperatingMarginTTM'])` |
| **ROE** | `ReturnOnEquityTTM` | `static_dict['roe'] = float(overview['ReturnOnEquityTTM'])` |
| **ROA** | `ReturnOnAssetsTTM` | `static_dict['roa'] = float(overview['ReturnOnAssetsTTM'])` |
| **EPS** | `EPS` | `static_dict['eps'] = float(overview['EPS'])` |
| **PEG Ratio** | `PEGRatio` | `static_dict['peg_ratio'] = float(overview['PEGRatio'])` |
| **Book Value** | `BookValue` | `static_dict['book_value'] = float(overview['BookValue'])` |

**Implementation:**
```csharp
// In HistoricalDataService.cs or new service
public async Task<CompanyOverview> GetCompanyOverview(string symbol)
{
    var url = $"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={_apiKey}";
    var response = await _httpClient.GetAsync(url);
    var json = await response.Content.ReadAsStringAsync();
    return JsonConvert.DeserializeObject<CompanyOverview>(json);
}
```

### **Macro Economic Indicators (Advanced)**

Alpha Vantage provides economic indicators that can be used as known-future events:

| Indicator | Function | Use Case |
|-----------|----------|----------|
| **Federal Funds Rate** | `FEDERAL_FUNDS_RATE` | Interest rate decisions (known schedule) |
| **CPI** | `CPI` | Inflation data (known release dates) |
| **Unemployment** | `UNEMPLOYMENT` | Jobs report (known release schedule) |
| **Real GDP** | `REAL_GDP` | GDP releases (known schedule) |

**Implementation (Advanced):**
```python
def add_macro_event_flags(df):
    """Add binary flags for scheduled macro economic releases"""
    # FOMC meetings (8 per year, known schedule)
    fomc_dates = ['2024-01-31', '2024-03-20', '2024-05-01', ...]
    df['is_fomc_day'] = df['date'].dt.strftime('%Y-%m-%d').isin(fomc_dates).astype(int)
    
    # CPI release (typically 2nd or 3rd week of month)
    df['is_cpi_week'] = ((df['day'] >= 10) & (df['day'] <= 20)).astype(int)
    
    # Jobs report (first Friday of month)
    df['is_jobs_report'] = ((df['dayofweek'] == 4) & (df['day'] <= 7)).astype(int)
    
    return df
```

---

## ?? Implementation Priority

### **Phase 1: Critical Fixes (Required for TFT to Work)**
1. ? **Add calendar features to training** (DONE in train_from_database.py)
2. ? **Fix synthetic lookback in inference** (modify C# + tft_integration.py)
3. ? **Pass calendar features during inference** (modify predict_single)

### **Phase 2: Enhanced Features (Performance Boost)**
4. ?? **Add Alpha Vantage OVERVIEW static features**
5. ?? **Implement proper market calendar** (use `pandas_market_calendars`)
6. ?? **Add holiday flags** (NYSE calendar)

### **Phase 3: Advanced (Maximum Performance)**
7. ?? **Add macro event flags** (FOMC, CPI, Jobs Report)
8. ?? **Sector-specific features** (sector momentum, sector rotation)
9. ?? **Volatility regime indicators** (VIX levels, volatility clusters)

---

## ?? Expected Performance Improvements

With all TFT requirements properly implemented:

| Feature Set | Expected Improvement | Reason |
|-------------|---------------------|--------|
| **Real Lookback Window** | +15-20% accuracy | Model sees actual price trends |
| **Calendar Features** | +10-15% accuracy | Captures day-of-week, month-end effects |
| **Market Calendar** | +5-10% accuracy | Handles holidays, low-volume days |
| **Static Covariates** | +5-8% accuracy | Sector, fundamentals, size effects |
| **Macro Events** | +3-5% accuracy | Pre/post announcement effects |
| **TOTAL POTENTIAL** | **+38-58% improvement** | Over current synthetic baseline |

---

## ?? References

- **TFT Paper:** https://arxiv.org/abs/1912.09363
- **Alpha Vantage API:** https://www.alphavantage.co/documentation/
- **Market Calendars:** https://github.com/rsheftel/pandas_market_calendars
- **Fed Schedule:** https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- **BLS Schedule:** https://www.bls.gov/schedule/news_release/

---

## ? Next Steps

1. **Test current training with calendar features:**
   ```bash
   python train_from_database.py "connection_string" output.json tft tft 100
   ```

2. **Modify C# to pass historical sequences** (see Fix #1 above)

3. **Update `tft_integration.py` to accept historical data** (see Fix #1 above)

4. **Add Alpha Vantage OVERVIEW fetching** (see Static Covariates section)

5. **Implement market calendar library** (Phase 2)

6. **Validate TFT predictions with proper inputs**

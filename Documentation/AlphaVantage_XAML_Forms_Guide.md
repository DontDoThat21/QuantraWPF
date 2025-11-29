# Alpha Vantage API - XAML Forms Guide for Financial Analysis Application

## Overview

This document provides a comprehensive guide for creating XAML markup forms in a WPF financial analysis application that integrates with the Alpha Vantage API. The focus is on supporting Python-based ML predictions for stock price analysis.

---

## Table of Contents

1. Core Stock Data Forms
2. Technical Indicators Forms
3. Fundamental Data Forms
4. Time Series Data Forms
5. Forex and Cryptocurrency Forms
6. Intelligence and News Forms
7. Python ML Prediction Integration Forms
8. Sample XAML Templates

---

## 1. Core Stock Data Forms

### 1.1 Global Quote Form

**API Endpoint:** `GLOBAL_QUOTE`

**Purpose:** Display real-time stock quote information

**Recommended XAML Components:**

| Component | Purpose | Binding Property |
|-----------|---------|------------------|
| TextBlock | Symbol display | `{Binding Symbol}` |
| TextBlock | Current Price | `{Binding Price, StringFormat=\${0:F2}}` |
| TextBlock | Change | `{Binding Change, StringFormat={}{0:F2}}` |
| TextBlock | Change Percent | `{Binding ChangePercent, StringFormat={}{0:P2}}` |
| TextBlock | Day High | `{Binding DayHigh, StringFormat=\${0:F2}}` |
| TextBlock | Day Low | `{Binding DayLow, StringFormat=\${0:F2}}` |
| TextBlock | Volume | `{Binding Volume, StringFormat={}{0:N0}}` |
| TextBlock | Last Updated | `{Binding LastUpdated, StringFormat={}{0:MM/dd/yyyy HH:mm}}` |

**Color Coding:**

- Green: Positive change
- Red: Negative change
- White/Gray: Neutral

---

### 1.2 Symbol Search Form

**API Endpoint:** `SYMBOL_SEARCH`

**Purpose:** Search and select stock symbols

**Recommended XAML Components:**

| Component | Purpose | Properties |
|-----------|---------|------------|
| TextBox | Search input | `TextChanged` event for live search |
| Popup | Dropdown results | `StaysOpen="False"` |
| ListBox | Search results | Custom ItemTemplate with Symbol, Name, Region |
| Button | Confirm selection | Click handler |

---

## 2. Technical Indicators Forms

### 2.1 Momentum Indicators Panel

**API Endpoints:** `RSI`, `STOCH`, `CCI`, `MOM`, `WILLR`, `ADX`, `AROON`

**Purpose:** Display momentum-based technical indicators

**Recommended XAML Components:**

| Indicator | Display Type | Color Logic |
|-----------|--------------|-------------|
| RSI | Progress Bar + Value | <30 Green (Oversold), >70 Red (Overbought) |
| Stochastic %K/%D | Dual Line Chart | Crossover signals |
| CCI | Value with Color | <-100 Green, >100 Red |
| Williams %R | Progress Bar | <-80 Green, >-20 Red |
| ADX | Gauge/Value | >25 Strong Trend (Blue) |

---

### 2.2 Trend Indicators Panel

**API Endpoints:** `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MACD`, `BBANDS`

**Purpose:** Display trend-following indicators

**Recommended XAML Components:**

| Indicator | Display Type | Properties |
|-----------|--------------|------------|
| Moving Averages | Chart Overlay | Multiple lines with legend |
| MACD | Histogram + Lines | MACD, Signal, Histogram values |
| Bollinger Bands | Chart Overlay | Upper, Middle, Lower bands |

**Bollinger Bands Data:**

- Upper Band: Price level
- Middle Band: Price level (SMA)
- Lower Band: Price level
- Band Width: Percentage
- %B Position: 0-1 value

**MACD Data:**

- MACD Line: Value
- Signal Line: Value
- Histogram: Value with Bullish/Bearish indicator

---

### 2.3 Volume Indicators Panel

**API Endpoints:** `OBV`, `AD`, `ADOSC`, `VWAP`

**Purpose:** Display volume-based technical indicators

**Recommended XAML Components:**

| Indicator | Display Type | Purpose |
|-----------|--------------|---------|
| OBV | Line Chart | Cumulative volume flow |
| AD (Accum/Dist) | Line Chart | Money flow direction |
| ADOSC | Histogram | A/D momentum |
| VWAP | Value + Comparison | Fair value benchmark |

**VWAP Display:**

- VWAP Value
- Current Price
- Position (Above/Below VWAP)
- Distance (Percentage)

---

### 2.4 Volatility Indicators Panel

**API Endpoints:** `ATR`, `NATR`, `TRANGE`

**Purpose:** Display volatility measurements

**Recommended XAML Components:**

| Indicator | Display Type | Purpose |
|-----------|--------------|---------|
| ATR | Value + Historical | Average price movement |
| NATR | Percentage | Normalized ATR |
| True Range | Current Value | Today's volatility |

---

## 3. Fundamental Data Forms

### 3.1 Company Overview Form

**API Endpoint:** `OVERVIEW`

**Purpose:** Display company fundamental data

**Company Information Section:**

- Name, Symbol, Exchange
- Sector, Industry
- Description (expandable TextBlock)

**Valuation Metrics Section:**

| Metric | Display Format |
|--------|----------------|
| Market Cap | Currency (Trillions/Billions) |
| P/E Ratio | Decimal |
| PEG Ratio | Decimal |
| Book Value | Currency |
| Dividend Yield | Percentage |
| EPS | Currency |
| 52-Week High | Currency |
| 52-Week Low | Currency |

---

### 3.2 Income Statement Form

**API Endpoint:** `INCOME_STATEMENT`

**Purpose:** Display quarterly/annual income data

**Recommended XAML Components:**

| Data Point | Display Type | Format |
|------------|--------------|--------|
| Total Revenue | DataGrid Column | Currency (Billions) |
| Gross Profit | DataGrid Column | Currency |
| Operating Income | DataGrid Column | Currency |
| Net Income | DataGrid Column | Currency |
| EPS | DataGrid Column | Decimal |

---

### 3.3 Balance Sheet Form

**API Endpoint:** `BALANCE_SHEET`

**Purpose:** Display company balance sheet data

**Key Sections:**

- Total Assets
- Total Liabilities
- Shareholders' Equity
- Current Ratio
- Debt-to-Equity Ratio

---

### 3.4 Cash Flow Form

**API Endpoint:** `CASH_FLOW`

**Purpose:** Display cash flow statement data

**Key Sections:**

- Operating Cash Flow
- Investing Cash Flow
- Financing Cash Flow
- Free Cash Flow

---

### 3.5 Earnings Form

**API Endpoint:** `EARNINGS`

**Purpose:** Display quarterly/annual earnings data

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| DataGrid | Historical earnings data |
| Chart | EPS trend visualization |
| TextBlocks | Upcoming earnings date |

---

## 4. Time Series Data Forms

### 4.1 Intraday Chart Form

**API Endpoint:** `TIME_SERIES_INTRADAY`

**Purpose:** Display intraday price data (1min, 5min, 15min, 30min, 60min)

**Recommended XAML Components:**

| Component | Purpose | Properties |
|-----------|---------|------------|
| CartesianChart | Price visualization | LiveCharts.Wpf |
| ComboBox | Interval selector | 1min, 5min, 15min, 30min, 60min |
| DatePicker | Date range | Start/End dates |
| ToggleButtons | Chart type | Candlestick, Line, Area |

---

### 4.2 Daily/Weekly/Monthly Chart Form

**API Endpoints:** `TIME_SERIES_DAILY`, `TIME_SERIES_DAILY_ADJUSTED`, `TIME_SERIES_WEEKLY`, `TIME_SERIES_MONTHLY`

**Purpose:** Display historical price data

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| Time Range Buttons | 1D, 5D, 1M, 6M, 1Y, 5Y, ALL |
| CartesianChart | Historical price chart |
| Volume Chart | Secondary chart below price |
| DataGrid | OHLCV data table |

---

## 5. Forex and Cryptocurrency Forms

### 5.1 Forex Exchange Rate Form

**API Endpoints:** `CURRENCY_EXCHANGE_RATE`, `FX_DAILY`, `FX_INTRADAY`

**Purpose:** Display forex pair data

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| ComboBox (From) | Source currency selector |
| ComboBox (To) | Target currency selector |
| TextBlock | Exchange rate display |
| Chart | Historical exchange rate |

**Forex Data Display:**

- Exchange Rate
- Bid Price
- Ask Price
- Last Refresh Timestamp

---

### 5.2 Cryptocurrency Form

**API Endpoints:** `DIGITAL_CURRENCY_DAILY`, `DIGITAL_CURRENCY_WEEKLY`, `CRYPTO_INTRADAY`

**Purpose:** Display cryptocurrency data

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| ComboBox | Cryptocurrency selector (BTC, ETH, etc.) |
| ComboBox | Market selector (USD, EUR, etc.) |
| Chart | Price history |
| TextBlocks | Market cap, volume |

---

## 6. Intelligence and News Forms

### 6.1 News Sentiment Form

**API Endpoint:** `NEWS_SENTIMENT`

**Purpose:** Display market news and sentiment analysis

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| ListView | News articles list |
| ProgressBar | Sentiment score visualization |
| TextBlock | Article summary |
| Hyperlink | Link to full article |

**News Item Template Fields:**

- Sentiment Label (Bullish/Bearish/Neutral)
- Headline
- Source
- Time ago
- Sentiment Score (0-1)
- Summary

---

### 6.2 Top Gainers/Losers Form

**API Endpoint:** `TOP_GAINERS_LOSERS`

**Purpose:** Display market movers

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| TabControl | Gainers, Losers, Most Active tabs |
| DataGrid | Symbol, Price, Change%, Volume |
| Button | Refresh data |

---

### 6.3 Insider Transactions Form

**API Endpoint:** `INSIDER_TRANSACTIONS`

**Purpose:** Display insider trading activity

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| DataGrid | Transaction details |
| FilterComboBox | Transaction type filter |
| DateRangePicker | Date range filter |

---

## 7. Python ML Prediction Integration Forms

### 7.1 Prediction Analysis Control

**Purpose:** Display ML-generated price predictions

**Recommended XAML Components:**

| Component | Purpose | Binding |
|-----------|---------|---------|
| DataGrid | Top predictions list | `{Binding TopPredictions}` |
| ProgressBar | Confidence score | `{Binding Confidence}` |
| TextBlock | Predicted action | `{Binding PredictedAction}` |
| TextBlock | Target price | `{Binding TargetPrice}` |
| TextBlock | Potential return | `{Binding PotentialReturn}` |
| Chart | Prediction visualization | Price + Predicted line |

**Prediction Grid Columns:**

| Column | Binding | Format |
|--------|---------|--------|
| Symbol | `{Binding Symbol}` | Text |
| Action | `{Binding PredictedAction}` | BUY/SELL with color |
| Confidence | `{Binding Confidence}` | Progress bar + percentage |
| Current | `{Binding CurrentPrice}` | Currency |
| Target | `{Binding TargetPrice}` | Currency |
| Potential | `{Binding PotentialReturn}` | Percentage |
| Signals | Indicator badges | RSI, MACD, ADX, VOL |

---

### 7.2 Trading Strategy Selector

**Purpose:** Select and configure trading strategies for predictions

**Recommended XAML Components:**

| Component | Purpose |
|-----------|---------|
| ComboBox | Strategy selector |
| CheckBox list | Indicator toggles |
| Slider | Confidence threshold |
| ComboBox | Timeframe selector |

**Available Strategies:**

- SMA Crossover
- RSI Divergence
- Bollinger Bands
- MACD Crossover

---

### 7.3 Backtesting Results Form

**Purpose:** Display historical strategy performance

**Recommended XAML Components:**

| Metric | Display Type |
|--------|--------------|
| Total Return | Percentage with color |
| Win Rate | Progress bar |
| Sharpe Ratio | Value |
| Max Drawdown | Value with color |
| Profit Factor | Value |

---

### 7.4 Sentiment Correlation Panel

**Purpose:** Correlate sentiment with price predictions

**Recommended XAML Sections:**

- Overall Sentiment Score
- News Sentiment
- Social Media Sentiment  
- Analyst Sentiment
- Sentiment vs. Price Chart

---

## 8. Sample XAML Templates

### 8.1 Stock Quote Card Template

**Note:** This template uses styles from `EnhancedStyles.xaml`. Ensure the ResourceDictionary is properly merged.

```xml
<Border Background="#262638" BorderBrush="#3E3E56" BorderThickness="1" 
        CornerRadius="5" Padding="10">
    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        
        <StackPanel Grid.Row="0" Orientation="Horizontal">
            <TextBlock Text="{Binding Symbol}" 
                       Style="{StaticResource EnhancedHeaderTextBlockStyle}"/>
            <TextBlock Text="{Binding CompanyName}" 
                       Style="{StaticResource EnhancedSmallTextBlockStyle}"
                       Margin="10,0,0,0"/>
        </StackPanel>
        
        <StackPanel Grid.Row="1" Orientation="Horizontal" Margin="0,5">
            <TextBlock Text="{Binding Price, StringFormat=C2}" 
                       Style="{StaticResource EnhancedTextBlockStyle}"
                       FontSize="24"/>
            <TextBlock Text="{Binding ChangePercent, StringFormat=P2}"
                       Style="{StaticResource EnhancedSmallTextBlockStyle}"
                       Margin="10,0,0,0"/>
        </StackPanel>
        
        <StackPanel Grid.Row="2" Orientation="Horizontal">
            <TextBlock Text="{Binding Volume, StringFormat='Vol: {0:N0}'}" 
                       Style="{StaticResource EnhancedSmallTextBlockStyle}"/>
        </StackPanel>
    </Grid>
</Border>
```

---

### 8.2 Indicator Badge Template

**Note:** Uses standardized styles and the application's color scheme.

```xml
<Border Width="50" Height="50" CornerRadius="5" Padding="5"
        Background="#2A2A3B" BorderBrush="#3E3E56" BorderThickness="1">
    <StackPanel VerticalAlignment="Center">
        <TextBlock Text="{Binding Name}" 
                   Style="{StaticResource EnhancedSmallTextBlockStyle}"
                   FontWeight="Bold"
                   HorizontalAlignment="Center"/>
        <TextBlock Text="{Binding Value, StringFormat={}{0:F1}}" 
                   Style="{StaticResource EnhancedSmallTextBlockStyle}"
                   HorizontalAlignment="Center"/>
    </StackPanel>
</Border>
```

---

### 8.3 Prediction Row Template

**Note:** Uses styles compatible with `EnhancedDataGridColumnStyle` for consistent DataGrid styling.

```xml
<DataTemplate>
    <Grid>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="80"/>
            <ColumnDefinition Width="80"/>
            <ColumnDefinition Width="100"/>
            <ColumnDefinition Width="90"/>
            <ColumnDefinition Width="90"/>
            <ColumnDefinition Width="90"/>
        </Grid.ColumnDefinitions>
        
        <TextBlock Grid.Column="0" Text="{Binding Symbol}" 
                   Style="{StaticResource EnhancedTextBlockStyle}"
                   FontWeight="Bold"/>
        
        <TextBlock Grid.Column="1" Text="{Binding PredictedAction}"
                   Style="{StaticResource EnhancedTextBlockStyle}"/>
        
        <Grid Grid.Column="2">
            <ProgressBar Value="{Binding Confidence}" Maximum="1" Height="15"/>
            <TextBlock Text="{Binding Confidence, StringFormat={}{0:P0}}"
                       Style="{StaticResource EnhancedSmallTextBlockStyle}"
                       HorizontalAlignment="Center"/>
        </Grid>
        
        <TextBlock Grid.Column="3" 
                   Text="{Binding CurrentPrice, StringFormat=C2}"
                   Style="{StaticResource EnhancedTextBlockStyle}"/>
        
        <TextBlock Grid.Column="4" 
                   Text="{Binding TargetPrice, StringFormat=C2}"
                   Foreground="Cyan"
                   Style="{StaticResource EnhancedTextBlockStyle}"/>
        
        <TextBlock Grid.Column="5" 
                   Text="{Binding PotentialReturn, StringFormat={}{0:P2}}"
                   Style="{StaticResource EnhancedTextBlockStyle}"/>
    </Grid>
</DataTemplate>
```

---

## Summary: Recommended Forms for Python Prediction Application

### Essential Forms (Priority 1)

1. **Symbol Search Form** - `SYMBOL_SEARCH`
2. **Global Quote Form** - `GLOBAL_QUOTE`
3. **Daily Time Series Form** - `TIME_SERIES_DAILY_ADJUSTED`
4. **Technical Indicators Panel** - `RSI`, `MACD`, `BBANDS`, `VWAP`
5. **Prediction Analysis Control** - ML Integration

### Important Forms (Priority 2)

6. **Company Overview Form** - `OVERVIEW`
7. **Intraday Chart Form** - `TIME_SERIES_INTRADAY`
8. **News Sentiment Form** - `NEWS_SENTIMENT`
9. **Top Movers Form** - `TOP_GAINERS_LOSERS`
10. **Backtesting Results Form** - Historical validation

### Additional Forms (Priority 3)

11. **Forex Exchange Form** - `FX_DAILY`
12. **Cryptocurrency Form** - `DIGITAL_CURRENCY_DAILY`
13. **Earnings Form** - `EARNINGS`
14. **Income Statement Form** - `INCOME_STATEMENT`
15. **Insider Transactions Form** - `INSIDER_TRANSACTIONS`

---

## Design Guidelines

**IMPORTANT:** Always reference the existing `EnhancedStyles.xaml` resource dictionary for consistent styling across all components.

### Using Existing Styles

The application has standardized styles defined in `Quantra/Styles/EnhancedStyles.xaml`. Always use these styles instead of hardcoding properties:

| Style Name | Component Type | Usage |
|------------|----------------|-------|
| `EnhancedButtonStyle` | Button | All buttons |
| `EnhancedComboBoxStyle` | ComboBox | All dropdown selectors |
| `EnhancedTextBoxStyle` | TextBox | All text inputs |
| `EnhancedCheckBoxStyle` | CheckBox | All checkboxes |
| `EnhancedTextBlockStyle` | TextBlock | Standard text |
| `EnhancedSmallTextBlockStyle` | TextBlock | Labels and small text |
| `EnhancedHeaderTextBlockStyle` | TextBlock | Section headers |
| `EnhancedDataGridColumnHeaderStyle` | DataGridColumnHeader | Grid headers |
| `EnhancedDataGridRowStyle` | DataGridRow | Grid rows |
| `EnhancedDataGridCellStyle` | DataGridCell | Grid cells |

### Color Scheme (Quantra Dark Theme)

These colors match the existing application color scheme from `EnhancedStyles.xaml`:

| Element | Color | Notes |
|---------|-------|-------|
| Background | `#2A2A3B` | Main application background |
| Card Background | `#262638` | Panel/card backgrounds |
| Border | `#3E3E56` | Standard borders |
| Border (Accent) | `#007ACC` | Accent borders |
| Border (Hover) | `#1E90FF` | Hover state borders |
| Text Primary | `GhostWhite` | Main text color |
| Text Secondary | `#CCCCCC` | Secondary text |
| Text Disabled | `#666677` | Disabled text |
| Bullish/Buy | `#20C040` | Positive indicators |
| Bearish/Sell | `#C02020` | Negative indicators |
| Neutral | `#3E3E56` | Neutral indicators |
| Strong Trend | `#3E90FF` | Trend strength |

### Font Guidelines

The application uses **Franklin Gothic Medium** as the standard font family:

| Element | Size | Weight | Style Reference |
|---------|------|--------|-----------------|
| Headers | 18-22 | Bold | `EnhancedHeaderTextBlockStyle` |
| Subheaders | 14-16 | SemiBold | Custom or inherited |
| Body | 12 | Normal | `EnhancedTextBlockStyle` |
| Small/Labels | 10 | Normal | `EnhancedSmallTextBlockStyle` |
| Data Values | 12-14 | Normal/Bold | `EnhancedDataGridColumnStyle` |

### Responsive Design

- Use `Grid` with `*` sizing for flexible layouts
- Implement `ScrollViewer` for overflow content
- Use `WrapPanel` for indicator badges
- Enable column resizing in `DataGrid`
- Reference existing styles from `EnhancedStyles.xaml` ResourceDictionary

---

*Document Version: 1.0*

*Created for Quantra WPF Financial Analysis Platform*

*Alpha Vantage API Documentation Reference: https://www.alphavantage.co/documentation/*

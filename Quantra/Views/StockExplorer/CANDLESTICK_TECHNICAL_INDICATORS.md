# Candlestick Chart Technical Indicators Implementation

## Overview
Added comprehensive technical indicator support and drawing tools to the Candlestick Chart Modal for enhanced technical analysis capabilities.

## Features Added

### 1. Technical Indicators (Toggleable)
- **SMA (Simple Moving Average)** - 20-period, displayed in orange
- **EMA (Exponential Moving Average)** - 20-period, displayed in cyan
- **RSI (Relative Strength Index)** - 14-period (note: requires separate panel for full implementation)
- **MACD (Moving Average Convergence Divergence)** - Standard settings (12, 26, 9) (note: requires separate panel)
- **Bollinger Bands** - 20-period with 2 standard deviations
  - Upper band (dashed blue line)
  - Middle band (solid gray line)
  - Lower band (dashed blue line)
- **VWAP (Volume Weighted Average Price)** - Intraday indicator in yellow

### 2. Drawing Tools
- **Add Horizontal Line** - Draw support/resistance levels at specific prices
  - Custom price input
  - Optional label customization
  - Displayed as red dashed lines
- **Clear Lines** - Remove all drawn horizontal lines

### 3. User Interface Enhancements
Added a second row of controls in the chart control panel:
- Left side: Indicator checkboxes with tooltips
- Right side: Drawing tool buttons

## Technical Implementation

### Files Modified
1. **CandlestickChartModal.xaml**
   - Restructured control panel to support two rows
   - Added checkboxes for each technical indicator
   - Added buttons for drawing tools
   - Maintained existing interval selector, zoom controls, and cache status

2. **CandlestickChartModal.xaml.cs**
   - Added properties for indicator visibility with INotifyPropertyChanged
   - Implemented `UpdateIndicators()` method that:
     - Removes old indicator series
     - Calculates selected indicators using TechnicalIndicatorService
     - Adds new indicator series to the chart with proper styling
   - Added horizontal line drawing functionality
   - Integrated TechnicalIndicatorService into constructor

### Files Created
1. **HorizontalLineDialog.xaml** - Dialog for adding price levels
2. **HorizontalLineDialog.xaml.cs** - Code-behind for the dialog

## Usage Instructions

### Adding Indicators
1. Open the candlestick chart for any symbol
2. Check the desired indicator checkbox in the "Indicators" section
3. The indicator will immediately overlay on the chart
4. Uncheck to remove the indicator

### Drawing Support/Resistance Lines
1. Click "? Add Level" button
2. Enter the desired price level
3. Optionally enter a custom label
4. Click "Add" - the line will appear on the chart
5. Use "?? Clear Lines" to remove all drawn lines

## Color Coding
- **Green candles**: Price up (Close > Open)
- **Red candles**: Price down (Close < Open)
- **Orange line**: SMA (Simple Moving Average)
- **Cyan line**: EMA (Exponential Moving Average)
- **Yellow line**: VWAP (Volume Weighted Average Price)
- **Blue dashed lines**: Bollinger Bands (Upper/Lower)
- **Gray line**: Bollinger Bands (Middle)
- **Red dashed lines**: User-drawn support/resistance levels

## Future Enhancements
1. **Separate RSI Panel** - Add a dedicated panel below the main chart for RSI (typically 0-100 scale)
2. **Separate MACD Panel** - Add another panel for MACD line, signal line, and histogram
3. **Trend Lines** - Allow users to draw diagonal trend lines
4. **Fibonacci Retracement** - Add Fibonacci retracement level drawing tool
5. **Custom Indicator Parameters** - Allow users to customize periods for MA, RSI, etc.
6. **Pattern Recognition** - Highlight common candlestick patterns (doji, hammer, etc.)
7. **Alert Lines** - Set price alerts on drawn lines

## Dependencies
- LiveCharts.Wpf - For chart rendering
- TechnicalIndicatorService - For calculating indicators
- StockDataCacheService - For data retrieval
- AlphaVantageService - For market data

## Notes
- RSI and MACD indicators are enabled in the checkbox but require additional UI panels for proper display
  - Current implementation would need separate chart areas below the main candlestick chart
  - This requires more extensive XAML layout changes
- Indicators are recalculated whenever:
  - New data is loaded
  - User toggles an indicator on/off
  - Interval is changed
- Performance optimized by using cached historical data when available
- All drawing tools persist until explicitly cleared by the user

## Integration Points
The candlestick chart is accessed from:
- Stock Explorer view - "?? Chart" button
- Provides real-time data with configurable refresh intervals
- Supports multiple timeframes (1min, 5min, 15min, 30min, 60min)

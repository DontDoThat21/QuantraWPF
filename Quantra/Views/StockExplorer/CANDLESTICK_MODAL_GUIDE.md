# Real-Time Candlestick Chart Modal - Implementation Guide

## Overview

This feature adds a real-time candlestick chart modal window that opens when you double-click on any stock symbol in the StockExplorer grid. The modal displays intraday OHLCV (Open, High, Low, Close, Volume) data with automatic refresh capabilities, respecting AlphaVantage API rate limits.

## Features

### ? Core Features

1. **Real-Time Candlestick Chart**
   - Displays OHLCV data using LiveCharts.Wpf CandleSeries
   - Green candles for price increases, red for decreases
   - Smooth, professional chart rendering

2. **Volume Chart**
   - Separate volume chart below the main candlestick chart
   - Volume bars with semi-transparent coloring
   - Formatted with K/M/B suffixes for readability

3. **Auto-Refresh**
   - Toggle auto-refresh ON/OFF
   - Default refresh interval: 15 seconds
   - Respects AlphaVantage API rate limits (5 calls/minute for free tier)
   - Manual refresh button available

4. **Multiple Timeframes**
   - 1 minute intervals
   - 5 minute intervals (default)
   - 15 minute intervals
   - 30 minute intervals
   - 60 minute intervals

5. **Price Information Display**
   - Last price with color coding (green=up, red=down)
   - Price change (absolute and percentage)
   - Last update timestamp
   - API usage indicator

## How to Use

### Opening the Chart

**Method 1: Double-Click (Recommended)**
- Navigate to the StockExplorer view
- Locate a stock in the DataGrid
- **Double-click on any row** to open the candlestick chart modal

**Method 2: Programmatic**
```csharp
var modal = new CandlestickChartModal("AAPL", alphaVantageService, loggingService);
modal.Owner = Window.GetWindow(this);
modal.ShowDialog();
```

### Using the Chart

1. **Change Timeframe**
   - Use the "Interval" dropdown at the top
   - Select from 1min, 5min, 15min, 30min, or 60min
   - Chart automatically reloads with the new interval

2. **Auto-Refresh**
   - Toggle the "Auto-Refresh" switch ON/OFF
   - When ON, chart updates every 15 seconds
   - When OFF, manual refresh only

3. **Manual Refresh**
   - Click the "? Refresh Now" button
   - Immediately fetches latest data from AlphaVantage API

4. **Close**
   - Click the "? Close" button
   - Or click the X on the window title bar
   - Auto-refresh automatically stops when closed

## Architecture

### Files Created

```
Quantra/Views/StockExplorer/
??? CandlestickChartModal.xaml           # Modal window UI
??? CandlestickChartModal.xaml.cs        # Modal window code-behind
??? CANDLESTICK_MODAL_GUIDE.md           # This documentation
```

### Files Modified

```
Quantra/Views/StockExplorer/
??? StockExplorer.xaml                   # Added MouseDoubleClick event
??? StockExplorer.UIEventHandlers.cs     # Added event handler method
```

## API Reference

### CandlestickChartModal Class

```csharp
public partial class CandlestickChartModal : Window, INotifyPropertyChanged
```

#### Constructor

```csharp
public CandlestickChartModal(
    string symbol,                      // Stock symbol (e.g., "AAPL")
    AlphaVantageService alphaVantageService,  // Alpha Vantage service instance
    LoggingService loggingService       // Logging service instance
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Symbol` | `string` | Stock symbol being displayed |
| `IsAutoRefreshEnabled` | `bool` | Auto-refresh toggle state |
| `IsLoading` | `bool` | Loading state indicator |
| `IsDataLoaded` | `bool` | Data successfully loaded |
| `LastPrice` | `double` | Last closing price |
| `PriceChange` | `double` | Price change (absolute) |
| `PriceChangePercent` | `double` | Price change (percentage) |
| `CandlestickSeries` | `SeriesCollection` | LiveCharts candlestick data |
| `VolumeSeries` | `SeriesCollection` | LiveCharts volume data |
| `TimeLabels` | `List<string>` | X-axis time labels |

#### Methods

| Method | Description |
|--------|-------------|
| `LoadCandlestickDataAsync()` | Loads intraday data from Alpha Vantage |
| `UpdateChartWithData(List<HistoricalPrice> data)` | Updates chart series with data |
| `UpdatePriceInfo(List<HistoricalPrice> data)` | Updates price display information |
| `StartAutoRefresh()` | Starts the auto-refresh timer |
| `StopAutoRefresh()` | Stops the auto-refresh timer |

### Event Handler in StockExplorer

```csharp
private void StockDataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
```

Opens the candlestick chart modal when a user double-clicks on a stock in the DataGrid.

## Data Flow

```
User Double-Clicks Stock Row
    ?
StockDataGrid_MouseDoubleClick() triggered
    ?
Create CandlestickChartModal instance
    ?
Modal Window Opens
    ?
LoadCandlestickDataAsync() automatically called
    ?
Fetch intraday data from AlphaVantage API
    ?
UpdateChartWithData() populates series
    ?
UpdatePriceInfo() updates header info
    ?
Chart displays with auto-refresh (if enabled)
    ?
Every 15 seconds: LoadCandlestickDataAsync() repeats
```

## API Rate Limiting

### AlphaVantage Free Tier Limits
- **5 calls per minute**
- **500 calls per day**

### Implementation Strategy
- Default refresh interval: **15 seconds** (4 calls/minute - safe buffer)
- Respects API limits automatically
- Manual refresh available anytime
- Auto-refresh can be toggled OFF to conserve API calls

### Recommended Usage
1. **Individual Stock Analysis**: Keep auto-refresh ON (15s)
2. **Multiple Charts Open**: Toggle auto-refresh OFF for all but one
3. **Batch Analysis**: Use manual refresh only
4. **Premium API Key**: Adjust `REFRESH_INTERVAL_SECONDS` to shorter intervals

## Customization

### Change Refresh Interval

In `CandlestickChartModal.xaml.cs`:

```csharp
private const int REFRESH_INTERVAL_SECONDS = 15; // Change to desired seconds
```

### Change API Rate Limit

In `CandlestickChartModal.xaml.cs`:

```csharp
private const int API_RATE_LIMIT_CALLS = 5; // Change to your tier's limit
```

### Change Default Interval

In `CandlestickChartModal.xaml`:

```xml
<ComboBox x:Name="IntervalComboBox" 
          SelectedIndex="1"  <!-- Change: 0=1min, 1=5min, 2=15min, 3=30min, 4=60min -->
          ...>
```

### Change Candle Limit

In `CandlestickChartModal.xaml.cs`, line ~176:

```csharp
if (sortedData.Count > 100)  // Change to desired max candles
{
    sortedData = sortedData.Skip(sortedData.Count - 100).ToList();
}
```

## Styling

### Color Scheme (Quantra Dark Theme)

| Element | Color | Code |
|---------|-------|------|
| Background | Dark Purple | `#23233A` |
| Card Background | Darker Purple | `#2D2D4D` |
| Border | Gray-Blue | `#3E3E56` |
| Text Primary | White | `GhostWhite` |
| Bullish Candles | Green | `#20C040` |
| Bearish Candles | Red | `#C02020` |
| Volume Bars | Gray-Blue | `#606080` (semi-transparent) |

### Font

- **Primary Font**: Franklin Gothic Medium
- **Header Size**: 24px (symbol)
- **Body Size**: 12px (labels, text)
- **Small Size**: 10px (timestamps)

## Error Handling

### No Data Available
- Displays "No candlestick data available" message
- Hides chart area
- Allows manual refresh

### API Errors
- Caught and logged via `LoggingService`
- User-friendly error modal displayed
- Auto-refresh continues on next cycle

### Rate Limit Exceeded
- Error logged
- Auto-refresh continues (will succeed when limit resets)
- Manual refresh button remains enabled

## Performance Considerations

### Chart Rendering
- Limited to **100 candles** max for optimal performance
- Sorted by date ascending
- Efficient LiveCharts rendering

### Memory Management
- Modal properly disposes of timer on close
- Cancellation token prevents memory leaks
- Chart data cleared on close

### API Efficiency
- Caches data for short periods
- Respects rate limits automatically
- Uses "compact" output size (100 data points max from API)

## Troubleshooting

### Chart Not Opening
**Issue**: Double-click doesn't open modal
**Solution**: 
- Ensure stock row is selected
- Check console for errors
- Verify AlphaVantageService is properly initialized

### No Data Displayed
**Issue**: Modal opens but chart is empty
**Solution**:
- Check API key validity
- Verify symbol exists and has intraday data
- Check API rate limits (may be exceeded)
- Try manual refresh

### Auto-Refresh Not Working
**Issue**: Chart doesn't update automatically
**Solution**:
- Verify auto-refresh toggle is ON
- Check logs for API errors
- Ensure API rate limits not exceeded
- Try manual refresh first

### Slow Performance
**Issue**: Chart loads slowly or lags
**Solution**:
- Reduce max candles (line ~176 in code-behind)
- Increase refresh interval
- Close unused modals
- Check network connection

## Future Enhancements

### Potential Features (Not Implemented)
1. **Technical Indicators Overlay**
   - SMA/EMA lines
   - Bollinger Bands
   - VWAP line
   
2. **Chart Zoom & Pan**
   - Mouse wheel zoom
   - Click-and-drag panning
   - Reset zoom button

3. **Export Functionality**
   - Save chart as PNG
   - Export data to CSV
   - Copy to clipboard

4. **Customizable Alerts**
   - Price alerts
   - Volume alerts
   - Technical indicator signals

5. **Multiple Symbol Comparison**
   - Side-by-side charts
   - Overlay multiple symbols
   - Correlation analysis

6. **Extended Timeframes**
   - Daily bars
   - Weekly bars
   - Historical data (non-intraday)

## Integration Examples

### From Custom Control

```csharp
// In your custom control or view
private void ShowCandlestickChart(string symbol)
{
    var modal = new CandlestickChartModal(
        symbol,
        _alphaVantageService, // Your service instance
        _loggingService       // Your logging instance
    );
    
    modal.Owner = Window.GetWindow(this);
    modal.ShowDialog();
}
```

### From ViewModel

```csharp
// Command in ViewModel
public ICommand OpenChartCommand => new RelayCommand<string>(
    symbol =>
    {
        var modal = new CandlestickChartModal(
            symbol,
            _alphaVantageService,
            _loggingService
        );
        
        modal.Owner = Application.Current.MainWindow;
        modal.ShowDialog();
    },
    symbol => !string.IsNullOrEmpty(symbol)
);
```

### Context Menu Integration

```xml
<!-- In your DataGrid XAML -->
<DataGrid.ContextMenu>
    <ContextMenu>
        <MenuItem Header="Open Candlestick Chart" 
                  Click="OpenCandlestickChart_Click"/>
    </ContextMenu>
</DataGrid.ContextMenu>
```

```csharp
// In code-behind
private void OpenCandlestickChart_Click(object sender, RoutedEventArgs e)
{
    if (StockDataGrid.SelectedItem is QuoteData stock)
    {
        var modal = new CandlestickChartModal(
            stock.Symbol,
            _alphaVantageService,
            _loggingService
        );
        modal.Owner = Window.GetWindow(this);
        modal.ShowDialog();
    }
}
```

## Best Practices

### API Usage
? **DO**:
- Use auto-refresh for active monitoring
- Toggle OFF when not actively viewing
- Use manual refresh for quick checks
- Respect rate limits

? **DON'T**:
- Open multiple charts with auto-refresh enabled
- Set refresh interval below 15 seconds on free tier
- Leave charts open unnecessarily
- Ignore API rate limit warnings

### Performance
? **DO**:
- Limit candles displayed (100 is optimal)
- Close modals when done
- Use appropriate intervals for your needs
- Monitor memory usage

? **DON'T**:
- Display thousands of candles
- Open many modals simultaneously
- Use 1-minute intervals unless necessary
- Keep charts open in background

### User Experience
? **DO**:
- Provide clear loading indicators
- Show meaningful error messages
- Enable keyboard shortcuts (ESC to close)
- Preserve user's interval selection

? **DON'T**:
- Block UI during data loading
- Hide errors from user
- Force specific intervals
- Auto-close on errors

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `ESC` | Close modal |
| `F5` | Refresh data |

*Note: Additional shortcuts can be implemented by adding KeyBindings in XAML*

## Dependencies

### Required NuGet Packages
- **LiveCharts.Wpf** (v0.9.7 or higher)
- System.Text.Json (built-in .NET 9)

### Required Services
- `AlphaVantageService` - API data fetching
- `LoggingService` - Error and info logging

### Required Models
- `HistoricalPrice` - OHLCV data structure
- `QuoteData` - Stock quote information

## Support

### Documentation
- This guide (CANDLESTICK_MODAL_GUIDE.md)
- AlphaVantage API Documentation: https://www.alphavantage.co/documentation/
- LiveCharts Documentation: https://lvcharts.net/

### Logging
All operations are logged using `LoggingService`:
- **Info**: Normal operations
- **Warning**: API errors, missing data
- **Error**: Exceptions, critical failures

Check logs for troubleshooting.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024 | Initial implementation |
|  |  | - Real-time candlestick chart |
|  |  | - Volume chart |
|  |  | - Auto-refresh functionality |
|  |  | - Multiple timeframes |
|  |  | - Double-click integration |

---

## Summary

The real-time candlestick chart modal provides a comprehensive intraday analysis tool for stock trading. Key highlights:

? **Easy to Use**: Double-click any stock to view
? **Real-Time Updates**: Auto-refresh every 15 seconds
? **Professional Charts**: LiveCharts.Wpf rendering
? **API Friendly**: Respects rate limits automatically
? **Customizable**: Multiple intervals and settings
? **Production Ready**: Error handling and logging built-in

**Status**: ? **READY FOR PRODUCTION USE** ?

---

*Implementation Date: 2024*
*Status: Complete and Tested*
*Framework: WPF .NET 9 + LiveCharts*

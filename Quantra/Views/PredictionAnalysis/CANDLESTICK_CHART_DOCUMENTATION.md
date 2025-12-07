# Candlestick Chart Implementation for Multi-Horizon Forecasting

## Overview
The PredictionAnalysis view now features a comprehensive candlestick chart that displays OHLCV (Open, High, Low, Close, Volume) data combined with multi-horizon TFT (Temporal Fusion Transformer) predictions.

## Key Features

### 1. **Historical Candlestick Display**
- Displays historical OHLCV data as traditional candlestick charts
- Green candles indicate price increases (Close >= Open)
- Red candles indicate price decreases (Close < Open)
- Automatically fetches up to 3 months of daily historical data

### 2. **Multi-Horizon Forecast Visualization**
- Shows TFT predictions for multiple time horizons (5, 10, 20, 30 days ahead)
- Displays median prediction line connecting to the last historical price
- Shows 90% confidence interval bands (upper and lower bounds)
- Future predicted candles displayed with semi-transparent colors

### 3. **Volume Chart**
- Separate volume chart below the main price chart
- Volume bars colored based on price movement (green for up days, red for down days)
- Scaled with K/M/B suffixes for readability

### 4. **Chart Components**
The implementation is split across multiple files for maintainability:

#### `PredictionAnalysis.CandlestickChart.cs`
Contains the core charting logic:
- `InitializeCandlestickChart()` - Initializes chart data structures
- `UpdateCandlestickChart(symbol, historicalData, tftResult)` - Main update method
- `ClearCandlestickChart()` - Clears chart data
- Properties: `CandlestickSeriesCollection`, `VolumeSeriesCollection`, `ChartDateLabels`

#### `PredictionAnalysis.ChartIntegration.cs`
Handles integration with the analysis workflow:
- `UpdateChartWithPredictionsAsync(symbol)` - Fetches data and updates chart
- `OnPredictionGenerated(prediction)` - Called when new predictions are generated
- `ConvertTFTResultToTFTPredictionResult()` - Converts service results to chart format

## Usage

### Basic Usage
The chart automatically updates when you analyze a symbol:

1. Enter a stock symbol (e.g., "AAPL") in the Symbol Selection section
2. Click the "Analyze" button
3. The chart will display with historical data and predictions (if available)

### Programmatic Usage

```csharp
// Update chart with predictions for a specific symbol
await UpdateChartWithPredictionsAsync("AAPL");

// Clear the chart
ClearCandlestickChart();

// Update chart after generating a prediction
await OnPredictionGenerated(predictionModel);
```

## Chart Series

The chart displays multiple series:

1. **Historical Candlesticks** (`CandleSeries`)
   - Title: "{Symbol} - Historical"
   - Colors: Green (increase) / Red (decrease)
   - MaxColumnWidth: 10

2. **TFT Median Forecast** (`LineSeries`)
   - Title: "TFT Median Forecast"
   - Color: Dodger Blue (#1E90FF)
   - Connects to last historical price
   - StrokeThickness: 3

3. **Upper 90% Confidence Interval** (`LineSeries`)
   - Title: "Upper 90% CI"
   - Color: Orange (#FFA500)
   - Dashed line style
   - StrokeThickness: 2

4. **Lower 10% Confidence Interval** (`LineSeries`)
   - Title: "Lower 10% CI"
   - Color: Orange (#FFA500)
   - Dashed line style
   - StrokeThickness: 2

5. **Future Forecast Candles** (`CandleSeries`)
   - Title: "{Symbol} - Forecast"
   - Colors: Semi-transparent green/red
   - Visualizes predicted OHLC values

6. **Volume Bars** (`ColumnSeries`)
   - Title: "Volume"
   - Color: Dark gray-blue (#606080)
   - MaxColumnWidth: 15

## Data Flow

```
User Input (Symbol)
    ?
AnalyzeButton_Click()
    ?
UpdateChartWithPredictionsAsync(symbol)
    ?
Fetch Historical Data (HistoricalDataService)
    ?
Generate TFT Predictions (TFTPredictionService)
    ?
UpdateCandlestickChart(symbol, data, predictions)
    ?
Update UI (LiveCharts SeriesCollection)
```

## XAML Bindings

The chart binds to the following properties:

```xml
<!-- Main Candlestick Chart -->
<lvc:CartesianChart 
    Series="{Binding CandlestickSeriesCollection}"
    AxisX.Labels="{Binding ChartDateLabels}"
    AxisY.LabelFormatter="{Binding PriceFormatter}"/>

<!-- Volume Chart -->
<lvc:CartesianChart 
    Series="{Binding VolumeSeriesCollection}"
    AxisX.Labels="{Binding ChartDateLabels}"
    AxisY.LabelFormatter="{Binding VolumeFormatter}"/>

<!-- Chart Visibility -->
<Border Visibility="{Binding IsChartVisible, Converter={StaticResource BooleanToVisibilityConverter}}"/>
```

## TFT Prediction Integration

The chart seamlessly integrates with TFT multi-horizon predictions:

### Horizon Keys
- `"5d"` - 5 days ahead
- `"10d"` - 10 days ahead
- `"20d"` - 20 days ahead
- `"30d"` - 30 days ahead

### Prediction Data Structure
```csharp
TFTPredictionResult
{
    Symbol: "AAPL",
    Action: "BUY",
    Confidence: 0.85,
    CurrentPrice: 150.00,
    TargetPrice: 155.00,
    Horizons: {
        "5d": { MedianPrice: 151.50, LowerBound: 149.00, UpperBound: 154.00 },
        "10d": { MedianPrice: 153.00, LowerBound: 148.50, UpperBound: 157.50 },
        "20d": { MedianPrice: 155.00, LowerBound: 148.00, UpperBound: 162.00 },
        "30d": { MedianPrice: 157.00, LowerBound: 147.00, UpperBound: 167.00 }
    }
}
```

## Customization

### Adjusting Colors
Modify the brush colors in `UpdateCandlestickChart()`:

```csharp
// Bullish candle color
IncreaseBrush = new SolidColorBrush(Color.FromRgb(0x90, 0xEE, 0x90))

// Bearish candle color
DecreaseBrush = new SolidColorBrush(Color.FromRgb(0xF0, 0x80, 0x80))

// Prediction line color
Stroke = new SolidColorBrush(Color.FromRgb(0x1E, 0x90, 0xFF))
```

### Adjusting Time Range
Modify the `timeframe` parameter in `UpdateChartWithPredictionsAsync()`:

```csharp
var timeframe = "3month"; // Options: "1day", "1week", "1month", "3month", "1year"
```

### Adjusting Forecast Horizons
Modify the `forecastHorizons` parameter when calling TFT service:

```csharp
forecastHorizons: new List<int> { 5, 10, 20, 30 } // Days ahead
```

## Error Handling

The implementation includes comprehensive error handling:

1. **No Historical Data**: Chart is hidden with warning logged
2. **TFT Prediction Failure**: Chart still displays with historical data only
3. **Invalid Symbol**: Chart is cleared and error logged
4. **Type Conversion Errors**: Gracefully handled with fallback behavior

## Performance Considerations

- Historical data is limited to 3 months to maintain chart readability
- TFT predictions are cached to avoid redundant API calls
- Chart updates are async to prevent UI blocking
- Volume formatter uses K/M/B notation to reduce label clutter

## Future Enhancements

Potential improvements for future versions:

1. **Interactive Tooltips**: Show detailed OHLCV data on hover
2. **Zoom Controls**: Allow users to zoom in/out on specific date ranges
3. **Multiple Timeframes**: Toggle between daily, weekly, monthly views
4. **Technical Indicators Overlay**: Add SMA, EMA, Bollinger Bands to chart
5. **Export Functionality**: Save chart as image or CSV
6. **Real-Time Updates**: Auto-refresh chart during market hours
7. **Comparison Mode**: Compare multiple symbols side-by-side

## Troubleshooting

### Chart Not Displaying
- Check `IsChartVisible` property is set to `true`
- Verify historical data was fetched successfully
- Check console logs for errors

### Predictions Not Showing
- Ensure TFTPredictionService is properly initialized
- Verify TFT model is trained and available
- Check if horizons dictionary contains data

### Incorrect Colors/Styling
- Verify XAML resources are properly loaded
- Check if EnhancedStyles.xaml is referenced
- Ensure BooleanToVisibilityConverter is available

## Dependencies

- **LiveCharts.Wpf**: For chart rendering
- **Quantra.DAL.Services**: For data fetching
- **Quantra.Models**: For data models
- **System.Windows.Media**: For colors and brushes

## References

- LiveCharts Documentation: https://livecharts.dev/docs/WPF/2.0.0-rc6.1/
- Candlestick Charts: https://livecharts.dev/docs/WPF/2.0.0-rc6.1/samples.financial.basicCandlesticks
- TFT Paper: https://arxiv.org/abs/1912.09363

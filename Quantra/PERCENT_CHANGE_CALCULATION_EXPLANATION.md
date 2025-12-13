# % Change Calculation in StockExplorer

## Overview
The **% Change** value displayed in the StockExplorer DataGrid represents the percentage change in the stock's price from the previous trading day's close to the current/latest price.

## Calculation Source

### Primary Source: Alpha Vantage GLOBAL_QUOTE API
The % Change value is primarily retrieved directly from the **Alpha Vantage GLOBAL_QUOTE API** response.

**File**: `AlphaVantageService.cs`
**Method**: `GetQuoteDataAsync(string symbol)`
**Line**: 257

```csharp
QuoteData quoteData = new QuoteData
{
    Symbol = quote["01. symbol"]?.ToString() ?? "",
    Name = null,
    Price = TryParseDouble(quote["05. price"]),
    Change = TryParseDouble(quote["09. change"]),
    ChangePercent = TryParsePercentage(quote["10. change percent"]),  // <-- HERE
    DayHigh = TryParseDouble(quote["03. high"]),
    DayLow = TryParseDouble(quote["04. low"]),
    Volume = TryParseDouble(quote["06. volume"]),
    Date = TryParseDateTime(quote["07. latest trading day"]),
    // ...
};
```

### API Field Mapping
The Alpha Vantage GLOBAL_QUOTE API returns:
- **Field "09. change"**: Absolute price change (e.g., +2.50 or -1.75)
- **Field "10. change percent"**: Percentage change (e.g., "2.5%" or "-1.75%")

## Parsing Logic

The `TryParsePercentage()` helper method handles the conversion:

**File**: `AlphaVantageService.cs`
**Lines**: 2251-2267

```csharp
private static double TryParsePercentage(JToken token)
{
    if (token == null || token.Type == JTokenType.Null)
        return 0.0;

    var value = token.ToString();
    if (string.IsNullOrEmpty(value))
        return 0.0;

    // Remove percentage sign if present (e.g., "2.5%" becomes "2.5")
    value = value.TrimEnd('%');

    if (double.TryParse(value, out double result))
        return result;

    return 0.0;
}
```

### Parsing Behavior:
1. Accepts string like `"2.5%"` or `"2.5"`
2. Strips the `%` symbol if present
3. Parses to double (e.g., `2.5` not `0.025`)
4. Returns `0.0` if parsing fails or value is null

**Note**: The stored value is the percentage as a number (e.g., `2.5` for 2.5%), NOT as a decimal fraction (0.025).

## Formula (Alpha Vantage's Calculation)
According to Alpha Vantage documentation, the percentage change is calculated as:

```
ChangePercent = ((CurrentPrice - PreviousClose) / PreviousClose) × 100
```

Where:
- **CurrentPrice** = Latest trading price
- **PreviousClose** = Previous trading day's closing price

## Display in DataGrid

**File**: `StockExplorer.xaml`
**Column Binding**: `{Binding ChangePercent, StringFormat={}{0:P2}}`

The `{0:P2}` format specifier:
- Treats the value as a percentage
- Displays with 2 decimal places
- Automatically adds the `%` symbol
- Example: `2.5` displays as `"2.50%"`

### Color Coding
The column uses a converter to color-code the values:
- **Positive** (> 0): Green/positive color
- **Negative** (< 0): Red/negative color
- **Zero** (= 0): Default color

```xaml
<DataGridTextColumn Header="% Change" Binding="{Binding ChangePercent, StringFormat={}{0:P2}}" Width="80">
    <DataGridTextColumn.ElementStyle>
        <Style TargetType="TextBlock" BasedOn="{StaticResource EnhancedDataGridColumnStyle}">
            <Setter Property="Foreground" 
                    Value="{Binding ChangePercent, Converter={StaticResource PercentChangeColorConverter}}"/>
        </Style>
    </DataGridTextColumn.ElementStyle>
</DataGridTextColumn>
```

## Cached Data Behavior

### Issue: Cached Data Shows 0%
When stock data is loaded from **cache** (not fresh from API), the ChangePercent is set to **0** because:

1. Historical price data (HistoricalPrice objects) only stores OHLC (Open, High, Low, Close) values
2. They don't store the day-to-day change percentage
3. The comment in code explicitly states: `ChangePercent = 0, // Calculate from previous day if needed`

**Files Affected**:
- `QuoteDataService.cs` (Line 109)
- `StockDataCacheService.cs` (Lines 572, 719, 862)

```csharp
quoteDataList.Add(new QuoteData
{
    Symbol = symbol,
    Price = lastPrice.Close,
    // ...
    Change = 0,          // Calculate from previous day if needed
    ChangePercent = 0,   // Calculate from previous day if needed
    // ...
});
```

### Potential Enhancement
To calculate ChangePercent for cached data, the code would need to:

1. Retrieve the previous day's closing price from historical data
2. Calculate: `((CurrentPrice - PreviousClose) / PreviousClose) × 100`
3. Update the QuoteData object with the calculated value

Example implementation:
```csharp
double change = 0;
double changePercent = 0;

if (prices.Count >= 2)
{
    var currentPrice = prices.Last().Close;
    var previousClose = prices[prices.Count - 2].Close;
    
    if (previousClose != 0)
    {
        change = currentPrice - previousClose;
        changePercent = (change / previousClose) * 100;
    }
}

quoteDataList.Add(new QuoteData
{
    // ...
    Change = change,
    ChangePercent = changePercent,
    // ...
});
```

## Storage in StockExplorerData

When stock data is saved to the **StockExplorerData** table:

**File**: `StockExplorerDataService.cs`
**Method**: `SaveStockDataFromQuoteAsync(QuoteData quoteData)`

```csharp
await SaveStockDataAsync(
    symbol: quoteData.Symbol,
    name: quoteData.Name,
    price: quoteData.Price,
    change: quoteData.Change,
    changePercent: quoteData.ChangePercent,  // <-- Stored as-is from QuoteData
    // ...
);
```

The ChangePercent value is stored as received:
- **Fresh API data**: Contains actual % change from Alpha Vantage
- **Cached data**: Contains 0 (unless calculation is implemented)

## Summary

| Data Source | ChangePercent Value |
|-------------|---------------------|
| **Alpha Vantage GLOBAL_QUOTE API** | Actual % change calculated by Alpha Vantage |
| **Cached Historical Data** | 0 (not calculated) |
| **StockExplorerData Table** | Whatever was in QuoteData when saved |

**Recommendation**: To improve accuracy for cached data, implement the calculation using previous day's close price from historical data. This would ensure all stocks show meaningful % change values regardless of whether data comes from API or cache.

# How to Read StockDataCache Table

## Table Schema

```sql
CREATE TABLE [dbo].[StockDataCache] (
    [Symbol] NVARCHAR(20) NOT NULL,
    [TimeRange] NVARCHAR(50) NOT NULL,
    [Interval] NVARCHAR(50) NULL,
    [Data] NVARCHAR(MAX) NOT NULL,  -- GZip-compressed JSON
    [CacheTime] DATETIME2 NOT NULL,
    CONSTRAINT [PK_StockDataCache] PRIMARY KEY ([Symbol], [TimeRange])
)
```

## Data Format

The `Data` column contains:
- **Format**: `"GZIP:<Base64-encoded-compressed-data>"`
- **Compressed Content**: JSON array of `HistoricalPrice` objects
- **Compression Method**: GZip compression

## C# Example: Reading and Decompressing

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.Models;
using Quantra.Utilities;
using Newtonsoft.Json;

// Example 1: Read data for a specific symbol
public async Task<List<HistoricalPrice>> ReadStockData(string symbol)
{
    var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
    optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

    using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
    {
        // Get the latest cache entry for the symbol
        var cacheEntry = await dbContext.StockDataCache
            .Where(c => c.Symbol == symbol)
            .OrderByDescending(c => c.CachedAt)
            .FirstOrDefaultAsync();

        if (cacheEntry == null)
        {
            Console.WriteLine($"No cached data found for {symbol}");
            return new List<HistoricalPrice>();
        }

        // Decompress and deserialize
        var storedData = cacheEntry.Data;
        string jsonData;

        // Check if data is compressed
        if (CompressionHelper.IsCompressed(storedData))
        {
            jsonData = CompressionHelper.DecompressString(storedData);
        }
        else
        {
            jsonData = storedData;
        }

        // Deserialize from JSON
        var prices = JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);

        Console.WriteLine($"Found {prices?.Count ?? 0} price records for {symbol}");
        Console.WriteLine($"Date range: {prices?.FirstOrDefault()?.Date} to {prices?.LastOrDefault()?.Date}");

        return prices ?? new List<HistoricalPrice>();
    }
}

// Example 2: Query all cached symbols with their data
public async Task<Dictionary<string, List<HistoricalPrice>>> ReadAllCachedData()
{
    var result = new Dictionary<string, List<HistoricalPrice>>();
    
    var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
    optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

    using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
    {
        // Get all cache entries
        var allEntries = await dbContext.StockDataCache
            .OrderBy(c => c.Symbol)
            .ThenByDescending(c => c.CachedAt)
            .ToListAsync();

        // Group by symbol and take the latest entry for each
        var latestEntries = allEntries
            .GroupBy(e => e.Symbol)
            .Select(g => g.First())
            .ToList();

        foreach (var entry in latestEntries)
        {
            var storedData = entry.Data;
            string jsonData;

            if (CompressionHelper.IsCompressed(storedData))
            {
                jsonData = CompressionHelper.DecompressString(storedData);
            }
            else
            {
                jsonData = storedData;
            }

            var prices = JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
            if (prices != null)
            {
                result[entry.Symbol] = prices;
            }
        }

        Console.WriteLine($"Loaded data for {result.Count} symbols");
    }

    return result;
}

// Example 3: Query specific date range for a symbol
public async Task<List<HistoricalPrice>> ReadStockDataForDateRange(
    string symbol, 
    DateTime startDate, 
    DateTime endDate)
{
    var allData = await ReadStockData(symbol);

    // Filter by date range
    var filteredData = allData
        .Where(p => p.Date >= startDate && p.Date <= endDate)
        .OrderBy(p => p.Date)
        .ToList();

    Console.WriteLine($"Found {filteredData.Count} records for {symbol} between {startDate:yyyy-MM-dd} and {endDate:yyyy-MM-dd}");

    return filteredData;
}

// Example 4: Get summary statistics
public async Task<PriceSummary> GetPriceSummary(string symbol)
{
    var prices = await ReadStockData(symbol);

    if (prices == null || prices.Count == 0)
    {
        return null;
    }

    return new PriceSummary
    {
        Symbol = symbol,
        DataPoints = prices.Count,
        StartDate = prices.First().Date,
        EndDate = prices.Last().Date,
        HighestPrice = prices.Max(p => p.High),
        LowestPrice = prices.Min(p => p.Low),
        AverageClose = prices.Average(p => p.Close),
        TotalVolume = prices.Sum(p => p.Volume),
        LatestPrice = prices.Last().Close,
        PriceChange = prices.Last().Close - prices.First().Close,
        PriceChangePercent = ((prices.Last().Close - prices.First().Close) / prices.First().Close) * 100
    };
}

public class PriceSummary
{
    public string Symbol { get; set; }
    public int DataPoints { get; set; }
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public double HighestPrice { get; set; }
    public double LowestPrice { get; set; }
    public double AverageClose { get; set; }
    public long TotalVolume { get; set; }
    public double LatestPrice { get; set; }
    public double PriceChange { get; set; }
    public double PriceChangePercent { get; set; }
}
```

## SQL Example: Direct Query (Not Recommended)

If you need to query directly in SQL (not recommended due to compression), you would need to:

```sql
-- View the raw compressed data (not human-readable)
SELECT TOP 10
    Symbol,
    TimeRange,
    Interval,
    LEFT(Data, 50) + '...' AS DataPreview,
    CacheTime,
    LEN(Data) AS DataSize
FROM [StockDataCache]
ORDER BY CacheTime DESC

-- Get cache metadata
SELECT 
    Symbol,
    COUNT(*) AS CacheEntries,
    MAX(CacheTime) AS LatestCache,
    SUM(LEN(Data)) / 1024 / 1024 AS TotalSizeMB
FROM [StockDataCache]
GROUP BY Symbol
ORDER BY Symbol

-- Find symbols with data cached in the last 24 hours
SELECT 
    Symbol,
    TimeRange,
    Interval,
    CacheTime,
    DATEDIFF(MINUTE, CacheTime, GETDATE()) AS MinutesOld
FROM [StockDataCache]
WHERE CacheTime > DATEADD(HOUR, -24, GETDATE())
ORDER BY CacheTime DESC
```

## Decompression Logic

The `CompressionHelper` class handles compression/decompression:

```csharp
// Check if data is compressed
bool isCompressed = CompressionHelper.IsCompressed(data);
// Returns true if data starts with "GZIP:"

// Decompress data
string jsonData = CompressionHelper.DecompressString(compressedData);
// Steps:
// 1. Remove "GZIP:" prefix
// 2. Convert from Base64 to byte array
// 3. Decompress using GZipStream
// 4. Convert back to UTF-8 string
```

## HistoricalPrice Model

```csharp
public class HistoricalPrice
{
    public DateTime Date { get; set; }      // Trading date
    public double Open { get; set; }         // Opening price
    public double High { get; set; }         // Highest price
    public double Low { get; set; }          // Lowest price
    public double Close { get; set; }        // Closing price
    public long Volume { get; set; }         // Trading volume
    public double AdjClose { get; set; }     // Adjusted closing price
}
```

## Common Use Cases

### 1. Load data for charting
```csharp
var prices = await ReadStockData("AAPL");
var dates = prices.Select(p => p.Date).ToList();
var closePrices = prices.Select(p => p.Close).ToList();
```

### 2. Calculate technical indicators
```csharp
var prices = await ReadStockData("MSFT");
var sma20 = prices.TakeLast(20).Average(p => p.Close);
var rsi = CalculateRSI(prices, 14);
```

### 3. Backtesting strategies
```csharp
var prices = await ReadStockDataForDateRange("TSLA", 
    new DateTime(2023, 1, 1), 
    new DateTime(2023, 12, 31));
// Run backtest logic
```

### 4. Check cache freshness
```csharp
var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
{
    var cacheEntry = await dbContext.StockDataCache
        .Where(c => c.Symbol == "AAPL")
        .OrderByDescending(c => c.CachedAt)
        .FirstOrDefaultAsync();

    if (cacheEntry != null)
    {
        var age = DateTime.Now - cacheEntry.CachedAt;
        Console.WriteLine($"Cache is {age.TotalMinutes:F0} minutes old");
        
        if (age.TotalMinutes > 15)
        {
            Console.WriteLine("Cache is stale, consider refreshing");
        }
    }
}
```

## Performance Tips

1. **Use AsNoTracking()** for read-only queries to improve performance
2. **Cache decompressed data** in memory if you need to access it multiple times
3. **Filter dates after decompression** - the database can't filter compressed JSON
4. **Use pagination** when loading all symbols to avoid memory issues
5. **Consider parallel processing** when loading multiple symbols

## Related Files

- **Entity Definition**: `Quantra.DAL/Data/Entities/StockEntities.cs`
- **DbContext**: `Quantra.DAL/Data/QuantraDbContext.cs`
- **Compression Utility**: `Quantra/Utilities/CompressionHelper.cs`
- **Service Layer**: `Quantra.DAL/Services/StockDataCacheService.cs`
- **Model**: `Quantra.DAL/Models/HistoricalPrice.cs`

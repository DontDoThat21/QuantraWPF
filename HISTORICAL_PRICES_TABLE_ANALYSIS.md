# Historical Prices Table Analysis & Fix

## Problem
The `PredictionAnalysisRepository.cs` was using raw SQL queries to query a non-existent `HistoricalPrices` table:

```sql
SELECT Date, Open, High, Low, Close, Volume, AdjClose 
FROM HistoricalPrices 
WHERE Symbol = {symbol} 
ORDER BY Date ASC
```

## Analysis Results

### Tables That DON'T Exist:
- ? **`HistoricalPrices`** - Never created in EF Core schema
- ? **`QuoteDataCache`** - Mentioned in comments but not implemented as a DbSet

### Tables That DO Exist:
- ? **`StockDataCache`** - Stores historical price data as compressed JSON
- ? **`QuoteData`** - Not defined as a separate table (it's a model)
- ? **`StockSymbols`** - Stores stock symbol metadata

### How Historical Data is ACTUALLY Stored

Historical price data is stored in the `StockDataCache` table with this structure:

```csharp
public class StockDataCache
{
    public int Id { get; set; }
    public string Symbol { get; set; }
    public string TimeRange { get; set; }
    public string Interval { get; set; }
    public string Data { get; set; }  // Compressed JSON of List<HistoricalPrice>
    public DateTime CachedAt { get; set; }
    public DateTime? ExpiresAt { get; set; }
}
```

The `Data` column contains:
1. A JSON-serialized `List<HistoricalPrice>`
2. Compressed using `CompressionHelper.CompressString()`
3. Can be decompressed and deserialized back to the original list

## Solution Implemented

### Changed From (Raw SQL):
```csharp
var prices = await _context.Database
    .SqlQuery<HistoricalPrice>($@"
         SELECT Date, Open, High, Low, Close, Volume, AdjClose 
         FROM HistoricalPrices 
         WHERE Symbol = {symbol} 
         ORDER BY Date ASC")
    .ToListAsync();
```

### Changed To (Entity Framework):
```csharp
// Query StockDataCache for the most recent cached data for this symbol
var cacheEntry = await _context.StockDataCache
    .AsNoTracking()
    .Where(c => c.Symbol == symbol)
    .OrderByDescending(c => c.CachedAt)
    .FirstOrDefaultAsync();

if (cacheEntry == null)
{
    return new List<HistoricalPrice>();
}

// Deserialize the cached data
var storedData = cacheEntry.Data;
string jsonData;

// Check if data is compressed and decompress if needed
if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
{
    jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
}
else
{
    jsonData = storedData;
}

// Deserialize from JSON to HistoricalPrice list
var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);

// Sort by date ascending as expected
return prices?.OrderBy(p => p.Date).ToList() ?? new List<HistoricalPrice>();
```

## Benefits of This Approach

1. ? **Uses actual existing tables** - No more SQL errors
2. ? **Leverages EF Core** - Type-safe queries, better maintainability
3. ? **Handles compression** - Properly decompresses cached data
4. ? **Consistent with other services** - Matches `StockDataCacheService` patterns
5. ? **Better error handling** - Returns empty list gracefully if no data found

## Database Schema Comparison

### What You Have (StockDataCache):
| Column | Type | Description |
|--------|------|-------------|
| Id | int | Primary key |
| Symbol | string(20) | Stock symbol |
| TimeRange | string(50) | Range like "1mo", "1y" |
| Interval | string(50) | Interval like "1d", "1h" |
| **Data** | string | **Compressed JSON of List&lt;HistoricalPrice&gt;** |
| CachedAt | DateTime | Cache timestamp |
| ExpiresAt | DateTime? | Expiration time |

### What You DON'T Have (HistoricalPrices):
This table was never created. The `HistoricalPrice` class is a **DTO/Model**, not an **Entity**.

## Recommendation for Future

If you need to query historical prices more efficiently (without deserializing entire cache), consider:

1. **Option A**: Keep current approach - Simple, works well for cached data
2. **Option B**: Create a dedicated `HistoricalPrices` entity table with individual price records
   - Pro: More granular queries, easier filtering by date ranges
   - Con: More storage, more complex caching logic

For now, **Option A** (current fix) is recommended as it:
- Works with existing infrastructure
- Minimal code changes
- Maintains backward compatibility
- Uses established caching patterns

## Related Files
- ? Modified: `Quantra/Repositories/PredictionAnalysisRepository.cs`
- ?? Reference: `Quantra.DAL/Services/StockDataCacheService.cs` (similar pattern)
- ?? Reference: `Quantra.DAL/Data/Entities/StockEntities.cs` (entity definitions)
- ?? Reference: `Quantra.DAL/Data/QuantraDbContext.cs` (DbContext with all DbSets)

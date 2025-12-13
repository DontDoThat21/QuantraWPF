# StockExplorerData Table Implementation

## Overview
This document outlines the implementation of the `StockExplorerData` table for persistent storage of stock data displayed in the StockExplorer view, separate from the `StockDataCache` table used for predictions.

## Changes Made

### 1. Entity Creation
**File:** `Quantra.DAL\Data\Entities\StockEntities.cs`

Created `StockExplorerDataEntity` with the following properties:
- `Id` (Primary Key)
- `Symbol` (Required, max 20 chars)
- `Name` (max 500 chars)
- `Price`, `Change`, `ChangePercent`
- `DayHigh`, `DayLow`, `MarketCap`, `Volume`
- `Sector` (max 200 chars)
- `RSI`, `PERatio`, `VWAP` (Technical indicators)
- `Date`, `LastUpdated`, `LastAccessed`, `Timestamp`, `CacheTime`

Also created `StockConfigurationEntity` for predefined stock configurations.

### 2. Database Migrations
**Files:** 
- `Quantra.DAL\Migrations\CreateStockExplorerDataTable.sql`
- `Quantra.DAL\Migrations\CreateStockConfigurationsTable.sql`

Created SQL migration scripts with:
- Table creation with proper constraints
- Unique constraint on `Symbol`
- Indexes on `Symbol` and `LastUpdated` for performance
- Conditional creation (IF NOT EXISTS)

### 3. Service Layer
**File:** `Quantra.DAL\Services\StockExplorerDataService.cs`

The service was already implemented with the following methods:
- `SaveStockDataAsync()` - Saves/updates individual stock data
- `SaveStockDataFromQuoteAsync()` - Saves from QuoteData object
- `SaveStockDataBatchAsync()` - Batch saves for multiple stocks
- `GetStockDataAsync()` - Retrieves stock data by symbol
- `GetAllStockDataAsync()` - Gets all stock data
- `HasStockDataAsync()` - Checks existence
- `GetAllSymbolsWithDataAsync()` - Gets all symbols with data
- `DeleteStockDataAsync()` - Deletes stock data
- `ClearAllDataAsync()` - Clears all data

### 4. DbContext Registration
**File:** `Quantra.DAL\Data\QuantraDbContext.cs`

The `StockExplorerData` DbSet was already registered in the DbContext:
```csharp
public DbSet<StockExplorerDataEntity> StockExplorerData { get; set; }
```

### 5. Service Registration
**File:** `Quantra\Extensions\ServiceCollectionExtensions.cs`

The service was already registered in DI (lines 142-148):
```csharp
services.AddSingleton<StockExplorerDataService>(sp =>
{
    var dbContextFactory = sp.GetRequiredService<IDbContextFactory<QuantraDbContext>>();
    var loggingService = sp.GetRequiredService<LoggingService>();
    return new StockExplorerDataService(dbContextFactory, loggingService);
});
```

### 6. StockExplorer Integration

#### A. Service Injection
**File:** `Quantra\Views\StockExplorer\StockExplorer.xaml.cs`

The service was already injected in the constructor (line 44, 807):
```csharp
private readonly StockExplorerDataService _stockExplorerDataService;
```

#### B. First Load - Batch Save
**File:** `Quantra\Views\StockExplorer\StockExplorer.xaml.cs` (lines 1060-1072)

Data is already saved on first load for all selection modes:
```csharp
// Save loaded stock data to StockExplorerData table in batch
try
{
    if (stockList != null && stockList.Any())
    {
        await _stockExplorerDataService.SaveStockDataBatchAsync(stockList);
        _loggingService?.Log("Info", $"Saved {stockList.Count} stocks to StockExplorerData table for mode {mode}");
    }
}
catch (Exception saveEx)
{
    _loggingService?.Log("Warning", $"Failed to save stocks to StockExplorerData table", saveEx.ToString());
}
```

#### C. Individual Symbol Load
**File:** `Quantra\Views\StockExplorer\StockExplorer.xaml.cs` (around lines 1456-1470)

Added save call when loading individual symbols:
```csharp
await _cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);

// Save loaded stock data to StockExplorerData table
await _stockExplorerDataService.SaveStockDataFromQuoteAsync(quoteData).ConfigureAwait(false);
```

#### D. Individual Symbol Refresh
**File:** `Quantra\Views\StockExplorer\StockExplorer.xaml.cs` (around lines 1544-1558)

Added save call when refreshing individual symbols:
```csharp
await _cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);

// Save refreshed stock data to StockExplorerData table
await _stockExplorerDataService.SaveStockDataFromQuoteAsync(quoteData).ConfigureAwait(false);
```

#### E. Auto-Refresh Update
**File:** `Quantra\Views\StockExplorer\StockExplorer.UIEventHandlers.cs` (around lines 1835-1850)

Added batch save in auto-refresh functionality:
```csharp
// Update last refresh time
_lastAutoRefreshTime = DateTime.Now;

// Save refreshed stock data to StockExplorerData table in batch
try
{
    if (visibleStocks != null && visibleStocks.Any())
    {
        await _stockExplorerDataService.SaveStockDataBatchAsync(visibleStocks);
        _loggingService?.Log("Info", $"Saved {visibleStocks.Count} refreshed stocks to StockExplorerData table");
    }
}
catch (Exception saveEx)
{
    _loggingService?.Log("Warning", "Failed to save refreshed stock data to database", saveEx.ToString());
}
```

## Database Table Structure

### StockExplorerData Table
```sql
CREATE TABLE [dbo].[StockExplorerData] (
    [Id] INT IDENTITY(1,1) PRIMARY KEY,
    [Symbol] NVARCHAR(20) NOT NULL,
    [Name] NVARCHAR(500) NULL,
    [Price] FLOAT NOT NULL DEFAULT 0,
    [Change] FLOAT NOT NULL DEFAULT 0,
    [ChangePercent] FLOAT NOT NULL DEFAULT 0,
    [DayHigh] FLOAT NOT NULL DEFAULT 0,
    [DayLow] FLOAT NOT NULL DEFAULT 0,
    [MarketCap] FLOAT NOT NULL DEFAULT 0,
    [Volume] FLOAT NOT NULL DEFAULT 0,
    [Sector] NVARCHAR(200) NULL,
    [RSI] FLOAT NOT NULL DEFAULT 0,
    [PERatio] FLOAT NOT NULL DEFAULT 0,
    [VWAP] FLOAT NOT NULL DEFAULT 0,
    [Date] DATETIME NOT NULL DEFAULT GETDATE(),
    [LastUpdated] DATETIME NOT NULL DEFAULT GETDATE(),
    [LastAccessed] DATETIME NOT NULL DEFAULT GETDATE(),
    [Timestamp] DATETIME NOT NULL DEFAULT GETDATE(),
    [CacheTime] DATETIME NULL,
    CONSTRAINT [UQ_StockExplorerData_Symbol] UNIQUE ([Symbol])
);
```

**Indexes:**
- `IX_StockExplorerData_Symbol` - For fast symbol lookups
- `IX_StockExplorerData_LastUpdated` - For auto-refresh queries

## Data Flow

1. **First Load**
   - User selects a symbol selection mode (All Database, High Volume, RSI Oversold, etc.)
   - StockExplorer loads stocks from Alpha Vantage API or cache
   - Batch save to `StockExplorerData` table via `SaveStockDataBatchAsync()`

2. **Individual Symbol Selection**
   - User types and selects a symbol in Individual Asset mode
   - StockExplorer loads data from API or cache
   - Save to `StockExplorerData` table via `SaveStockDataFromQuoteAsync()`

3. **Manual Refresh**
   - User clicks Refresh button for individual symbol
   - StockExplorer refreshes data from API
   - Save updated data to `StockExplorerData` table

4. **Auto-Refresh**
   - Timer triggers periodic refresh of visible stocks
   - StockExplorer updates all stocks on current page
   - Batch save updated data to `StockExplorerData` table
   - Status shows last refresh time and success/error counts

## Separation from StockDataCache

The `StockExplorerData` table is **separate** from `StockDataCache`:

- **StockDataCache**: Used for prediction model inputs, stores serialized OHLCV data
- **StockExplorerData**: Used for StockExplorer view display, stores flat denormalized data

This separation ensures:
- Clean data isolation between UI and ML operations
- Optimized schema for each use case
- No conflicts between prediction cache and UI cache
- Independent update cycles

## Running Migrations

To apply the migrations:

1. Ensure SQL Server is running
2. Run the migration scripts in SQL Server Management Studio or via EF Core:
   ```bash
   sqlcmd -S (localdb)\MSSQLLocalDB -d QuantraDatabase -i "Quantra.DAL\Migrations\CreateStockExplorerDataTable.sql"
   sqlcmd -S (localdb)\MSSQLLocalDB -d QuantraDatabase -i "Quantra.DAL\Migrations\CreateStockConfigurationsTable.sql"
   ```

Or let Entity Framework create the tables on next application run (DbContext.Initialize() will handle it).

## Testing

To verify the implementation:

1. Launch the application
2. Navigate to StockExplorer
3. Select different symbol modes and verify data loads
4. Check that data is persisted in SQL Server:
   ```sql
   SELECT * FROM StockExplorerData ORDER BY LastUpdated DESC;
   ```
5. Enable Auto-Refresh and verify updates:
   ```sql
   SELECT Symbol, Price, LastUpdated FROM StockExplorerData ORDER BY LastUpdated DESC;
   ```

## Benefits

1. **Persistence**: Stock data survives application restarts
2. **Performance**: Reduces API calls by checking DB first
3. **History**: Tracks when data was last updated/accessed
4. **Separation**: Clean isolation from prediction cache
5. **Scalability**: Batch operations for efficient updates
6. **Auditing**: Full timestamp tracking for debugging

## Future Enhancements

- Add historical snapshots for price tracking over time
- Implement data retention policies (e.g., purge old data)
- Add analytics queries for most-viewed symbols
- Implement change tracking for audit trails
- Add caching layer for frequently accessed symbols

# Paper Trading Session Database Implementation

## Overview

This implementation adds database persistence for paper trading sessions, tracking when sessions start/stop and storing key performance metrics.

## Changes Made

### 1. New Entity: `PaperTradingSessionEntity`

**File:** `Quantra.DAL/Data/Entities/PaperTradingSessionEntity.cs`

A new entity to track paper trading sessions with the following key properties:
- `SessionId` - Unique identifier for each session
- `StartTime`/`EndTime` - Session duration
- `InitialBalance`/`FinalBalance` - Cash tracking
- `FinalPortfolioValue` - Total portfolio value including positions
- `RealizedPnL`/`UnrealizedPnL` - Profit/loss tracking
- `TradeCount`, `WinningTrades`, `LosingTrades` - Trade statistics
- `WinRate` - Calculated win rate percentage
- `Status` - Session status (Active, Completed, Reset)

### 2. New Service: `PaperTradingSessionService`

**File:** `Quantra.DAL/Services/PaperTradingSessionService.cs`

Service layer for managing paper trading sessions:
- `StartSessionAsync()` - Creates a new session when trading starts
- `StopSessionAsync()` - Finalizes a session when trading stops
- `UpdateSessionAsync()` - Updates session stats during trading
- `GetActiveSessionIdAsync()` - Retrieves the current active session
- `ResetSessionAsync()` - Resets a session and creates a new one

### 3. Updated: `QuantraDbContext`

**File:** `Quantra.DAL/Data/QuantraDbContext.cs`

Added DbSet for the new entity:
```csharp
public DbSet<PaperTradingSessionEntity> PaperTradingSessions { get; set; }
```

### 4. Updated: `PaperTradingViewModel`

**File:** `Quantra/ViewModels/PaperTradingViewModel.cs`

Modified to integrate with the session service:

#### Constructor Changes
- Initializes `PaperTradingSessionService` instance
- Stores current `_currentSessionId`

#### `ToggleEngine()` Method
**Before:**
```csharp
public void ToggleEngine()
{
    if (IsEngineRunning)
    {
        _tradingEngine.Stop();
        IsEngineRunning = false;
    }
    else
    {
        _tradingEngine.Start();
        IsEngineRunning = true;
    }
}
```

**After:**
```csharp
public async void ToggleEngine()
{
    if (IsEngineRunning)
    {
        _tradingEngine.Stop();
        IsEngineRunning = false;
        
        // Save session to database
        if (_currentSessionId.HasValue && _portfolioManager != null)
        {
            await _sessionService.StopSessionAsync(
                _currentSessionId.Value,
                _portfolioManager.CashBalance,
                _portfolioManager.TotalValue,
                _portfolioManager.RealizedPnL,
                _portfolioManager.UnrealizedPnL
            );
            _currentSessionId = null;
        }
    }
    else
    {
        _tradingEngine.Start();
        IsEngineRunning = true;
        
        // Create new session in database
        if (_portfolioManager != null)
        {
            _currentSessionId = await _sessionService.StartSessionAsync(
                _portfolioManager.CashBalance
            );
        }
    }
}
```

#### Event Handler Changes
- `OnOrderFilled()` - Now updates the session in the database after each fill
- `OnPortfolioChanged()` - Updates session when portfolio changes
- Added `UpdateSessionInDatabaseAsync()` - Helper method to sync current session stats to database

#### `ResetAccount()` Method
- Now calls `_sessionService.ResetSessionAsync()` to properly track account resets
- Creates a new session with the reset balance

## Database Schema

The new `PaperTradingSessions` table will be created automatically by EF Core on next database initialization:

```sql
CREATE TABLE [PaperTradingSessions] (
    [Id] INT IDENTITY(1,1) PRIMARY KEY,
    [SessionId] UNIQUEIDENTIFIER NOT NULL,
    [UserId] INT NULL,
    [StartTime] DATETIME2 NOT NULL,
    [EndTime] DATETIME2 NULL,
    [InitialBalance] DECIMAL(18,2) NOT NULL,
    [FinalBalance] DECIMAL(18,2) NULL,
    [FinalPortfolioValue] DECIMAL(18,2) NULL,
    [TotalPnL] DECIMAL(18,2) NULL,
    [RealizedPnL] DECIMAL(18,2) NULL,
    [UnrealizedPnL] DECIMAL(18,2) NULL,
    [TradeCount] INT NOT NULL,
    [WinningTrades] INT NOT NULL,
    [LosingTrades] INT NOT NULL,
    [WinRate] DECIMAL(5,2) NULL,
    [Status] NVARCHAR(20) NOT NULL,
    [Notes] NVARCHAR(1000) NULL,
    [CreatedAt] DATETIME2 NOT NULL,
    [UpdatedAt] DATETIME2 NOT NULL
);
```

## Usage Flow

### Starting a Trading Session

1. User clicks "Start" button
2. `ToggleEngine()` is called
3. Trading engine starts
4. A new record is created in `PaperTradingSessions` table with:
   - `SessionId` = New GUID
   - `StartTime` = Current time
   - `InitialBalance` = Current portfolio balance ($100,000 default)
   - `Status` = "Active"

### During Trading

- Each time an order is filled or portfolio changes:
  - `UpdateSessionInDatabaseAsync()` is called
  - Session record is updated with current:
    - `FinalBalance`, `FinalPortfolioValue`
    - `RealizedPnL`, `UnrealizedPnL`, `TotalPnL`
    - `TradeCount`, `WinningTrades`, `LosingTrades`
    - `WinRate` (calculated automatically)

### Stopping a Trading Session

1. User clicks "Stop" button
2. `ToggleEngine()` is called
3. Trading engine stops
4. Session record is finalized with:
   - `EndTime` = Current time
   - Final values for all metrics
   - `Status` = "Completed"

### Resetting Account

1. User clicks "Reset Account" and confirms
2. `ResetAccount()` is called
3. Current session is marked with `Status` = "Reset"
4. New session is created with `InitialBalance` = $100,000

## Benefits

1. **Historical Tracking** - All trading sessions are permanently recorded
2. **Performance Analysis** - Can query past sessions to analyze trading performance
3. **Audit Trail** - Complete record of when paper trading occurred and results
4. **Statistics** - Automatically calculates win rate and P&L metrics
5. **Session Continuity** - Can track active sessions across application restarts

## Future Enhancements

Possible improvements for future iterations:
- Associate sessions with user accounts when authentication is added
- Create a UI to view past session history
- Add charts/graphs showing session performance over time
- Export session data for external analysis
- Add more advanced metrics (Sharpe ratio, max drawdown, etc.)
- Link individual orders to their parent session via foreign key

## Testing

To test the implementation:

1. Run the application and open the Paper Trading control
2. Click "Start" to begin a trading session
3. Place some orders and observe portfolio changes
4. Click "Stop" to end the session
5. Query the database to verify the `PaperTradingSessions` table has the session record:

```sql
SELECT TOP 10 * FROM PaperTradingSessions ORDER BY StartTime DESC;
```

## Notes

- The table is created automatically by EF Core when the DbContext initializes
- Sessions are updated asynchronously to avoid blocking the UI
- Error handling is in place to log any database issues without crashing the app
- The session service uses the existing `LoggingService` for consistent logging

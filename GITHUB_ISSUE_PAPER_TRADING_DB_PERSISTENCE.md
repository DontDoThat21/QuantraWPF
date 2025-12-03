# GitHub Issue: Persist Paper Trading Positions and Orders to Database

## Issue Title
**[Feature] Add Database Persistence for Paper Trading Open Positions and Orders**

## Labels
`enhancement`, `database`, `paper-trading`, `persistence`

## Priority
**High** - Critical for maintaining paper trading state across sessions

---

## ?? Problem Statement

Currently, the Paper Trading feature in Quantra stores positions and orders **only in memory** within the `TradingEngine` and `PortfolioManager` classes. This means:

1. ? **All positions are lost** when the application closes or crashes
2. ? **Orders are not persisted** beyond the current session
3. ? **Historical position tracking** is not available
4. ? **Cannot restore session state** when resuming a paper trading session
5. ? **No audit trail** for positions and orders over time

The `PaperTradingSessionEntity` table exists but only stores **session-level aggregated statistics** (cash balance, P&L, trade counts), not the detailed positions and orders that make up the session.

---

## ?? Desired Solution

Implement full database persistence for:
1. **Open Positions** - Track all current positions with entry prices, quantities, and P&L
2. **Orders** - Track all order history (submitted, filled, cancelled, rejected)
3. **Session Restoration** - Ability to load positions/orders when resuming a session

---

## ?? Current Database Schema Analysis

### Existing Tables

| Table Name | Purpose | Status |
|------------|---------|--------|
| `PaperTradingSessions` | Session metadata & aggregated stats | ? Exists |
| `OrderHistory` | Generic order records (legacy) | ?? Exists but not integrated with Paper Trading |
| `TradeRecords` | Executed trade records (legacy) | ?? Exists but not integrated with Paper Trading |

### Issues with Existing Tables

**`OrderHistory` Table:**
- ? Uses `double` for prices (should be `decimal` for financial accuracy)
- ? Missing key fields: `OrderState`, `FilledQuantity`, `AvgFillPrice`, `SessionId`
- ? No relationship to `PaperTradingSessions`
- ? Not integrated with the TradingEngine `Order` class

**`TradeRecords` Table:**
- ? Not designed for positions - focuses on individual trades
- ? Missing position-specific fields: `AverageCost`, `UnrealizedPnL`, `RealizedPnL`
- ? No session tracking

---

## ??? Proposed Database Schema Changes

### 1. New Table: `PaperTradingPositions`

This table stores **open positions** for each paper trading session.

```sql
CREATE TABLE [dbo].[PaperTradingPositions] (
    [Id] INT PRIMARY KEY IDENTITY(1,1),
    [SessionId] UNIQUEIDENTIFIER NOT NULL,
    [Symbol] NVARCHAR(20) NOT NULL,
    [Quantity] INT NOT NULL,                    -- Positive = long, Negative = short
    [AverageCost] DECIMAL(18, 4) NOT NULL,
    [CurrentPrice] DECIMAL(18, 4) NOT NULL,
    [MarketValue] DECIMAL(18, 4) NOT NULL,
    [UnrealizedPnL] DECIMAL(18, 4) NOT NULL,
    [RealizedPnL] DECIMAL(18, 4) NOT NULL,
    [OpenedTime] DATETIME2 NOT NULL,
    [LastUpdateTime] DATETIME2 NOT NULL,
    [AssetType] NVARCHAR(20) NOT NULL DEFAULT 'Stock',  -- Stock, ETF, Option
    [IsClosed] BIT NOT NULL DEFAULT 0,          -- FALSE = open, TRUE = closed
    [ClosedTime] DATETIME2 NULL,
    [CreatedAt] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    [UpdatedAt] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT [FK_PaperTradingPositions_Sessions] 
        FOREIGN KEY ([SessionId]) 
        REFERENCES [dbo].[PaperTradingSessions]([SessionId])
        ON DELETE CASCADE,
    
    INDEX [IX_PaperTradingPositions_SessionId] ([SessionId]),
    INDEX [IX_PaperTradingPositions_Symbol] ([Symbol]),
    INDEX [IX_PaperTradingPositions_IsClosed] ([IsClosed])
)
```

**Key Features:**
- ? Tracks both open and closed positions (for history)
- ? Uses `DECIMAL(18,4)` for financial precision
- ? Links to session via `SessionId` foreign key
- ? Supports cascade delete when session is deleted
- ? Indexed for fast queries on session and symbol

---

### 2. New Table: `PaperTradingOrders`

This table stores **all orders** (pending, filled, cancelled, rejected) for each session.

```sql
CREATE TABLE [dbo].[PaperTradingOrders] (
    [Id] INT PRIMARY KEY IDENTITY(1,1),
    [OrderId] UNIQUEIDENTIFIER NOT NULL UNIQUE,  -- The Order.Id from TradingEngine
    [SessionId] UNIQUEIDENTIFIER NOT NULL,
    [Symbol] NVARCHAR(20) NOT NULL,
    [OrderType] NVARCHAR(20) NOT NULL,           -- Market, Limit, Stop, StopLimit, TrailingStop
    [Side] NVARCHAR(10) NOT NULL,                -- Buy, Sell
    [State] NVARCHAR(20) NOT NULL,               -- Pending, Submitted, PartiallyFilled, Filled, Cancelled, Rejected, Expired
    [Quantity] INT NOT NULL,
    [FilledQuantity] INT NOT NULL DEFAULT 0,
    [RemainingQuantity] INT NOT NULL,
    [LimitPrice] DECIMAL(18, 4) NULL,
    [StopPrice] DECIMAL(18, 4) NULL,
    [AvgFillPrice] DECIMAL(18, 4) NULL,
    [TimeInForce] NVARCHAR(10) NOT NULL DEFAULT 'Day',  -- Day, GTC, IOC, FOK
    [AssetType] NVARCHAR(20) NOT NULL DEFAULT 'Stock',
    [CreatedTime] DATETIME2 NOT NULL,
    [SubmittedTime] DATETIME2 NULL,
    [FilledTime] DATETIME2 NULL,
    [ExpirationTime] DATETIME2 NULL,
    [RejectReason] NVARCHAR(500) NULL,
    [Notes] NVARCHAR(1000) NULL,
    [CreatedAt] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    [UpdatedAt] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT [FK_PaperTradingOrders_Sessions] 
        FOREIGN KEY ([SessionId]) 
        REFERENCES [dbo].[PaperTradingSessions]([SessionId])
        ON DELETE CASCADE,
    
    INDEX [IX_PaperTradingOrders_SessionId] ([SessionId]),
    INDEX [IX_PaperTradingOrders_OrderId] ([OrderId]),
    INDEX [IX_PaperTradingOrders_Symbol] ([Symbol]),
    INDEX [IX_PaperTradingOrders_State] ([State]),
    INDEX [IX_PaperTradingOrders_CreatedTime] ([CreatedTime] DESC)
)
```

**Key Features:**
- ? Tracks complete order lifecycle
- ? Stores `OrderId` (Guid) to match with in-memory `Order` objects
- ? Tracks fill progress (`FilledQuantity`, `RemainingQuantity`, `AvgFillPrice`)
- ? Supports all order types and states
- ? Links to session with cascade delete
- ? Optimized indexes for common queries

---

### 3. New Table: `PaperTradingFills` (Optional Enhancement)

This table stores **individual fill records** for detailed order execution tracking.

```sql
CREATE TABLE [dbo].[PaperTradingFills] (
    [Id] INT PRIMARY KEY IDENTITY(1,1),
    [OrderId] UNIQUEIDENTIFIER NOT NULL,
    [SessionId] UNIQUEIDENTIFIER NOT NULL,
    [Symbol] NVARCHAR(20) NOT NULL,
    [Side] NVARCHAR(10) NOT NULL,               -- Buy, Sell
    [Quantity] INT NOT NULL,
    [Price] DECIMAL(18, 4) NOT NULL,
    [Commission] DECIMAL(18, 4) NOT NULL DEFAULT 0,
    [Slippage] DECIMAL(18, 4) NOT NULL DEFAULT 0,
    [FillTime] DATETIME2 NOT NULL,
    [CreatedAt] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT [FK_PaperTradingFills_Orders] 
        FOREIGN KEY ([OrderId]) 
        REFERENCES [dbo].[PaperTradingOrders]([OrderId])
        ON DELETE CASCADE,
    
    CONSTRAINT [FK_PaperTradingFills_Sessions] 
        FOREIGN KEY ([SessionId]) 
        REFERENCES [dbo].[PaperTradingSessions]([SessionId])
        ON DELETE CASCADE,
    
    INDEX [IX_PaperTradingFills_OrderId] ([OrderId]),
    INDEX [IX_PaperTradingFills_SessionId] ([SessionId]),
    INDEX [IX_PaperTradingFills_FillTime] ([FillTime] DESC)
)
```

**Purpose:**
- Track individual partial fills for large orders
- Detailed execution audit trail
- Calculate average fill prices accurately

---

## ?? Required Code Changes

### 1. Create Entity Classes

**File: `Quantra.DAL/Data/Entities/PaperTradingPositionEntity.cs`**

```csharp
using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    [Table("PaperTradingPositions")]
    public class PaperTradingPositionEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public Guid SessionId { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        public int Quantity { get; set; }

        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal AverageCost { get; set; }

        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal CurrentPrice { get; set; }

        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal MarketValue { get; set; }

        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal UnrealizedPnL { get; set; }

        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal RealizedPnL { get; set; }

        [Required]
        public DateTime OpenedTime { get; set; }

        [Required]
        public DateTime LastUpdateTime { get; set; }

        [Required]
        [MaxLength(20)]
        public string AssetType { get; set; }

        [Required]
        public bool IsClosed { get; set; }

        public DateTime? ClosedTime { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; }

        [Required]
        public DateTime UpdatedAt { get; set; }

        // Navigation property
        [ForeignKey(nameof(SessionId))]
        public PaperTradingSessionEntity Session { get; set; }
    }
}
```

**File: `Quantra.DAL/Data/Entities/PaperTradingOrderEntity.cs`**

```csharp
using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    [Table("PaperTradingOrders")]
    public class PaperTradingOrderEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public Guid OrderId { get; set; }

        [Required]
        public Guid SessionId { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(20)]
        public string OrderType { get; set; }

        [Required]
        [MaxLength(10)]
        public string Side { get; set; }

        [Required]
        [MaxLength(20)]
        public string State { get; set; }

        [Required]
        public int Quantity { get; set; }

        [Required]
        public int FilledQuantity { get; set; }

        [Required]
        public int RemainingQuantity { get; set; }

        [Column(TypeName = "decimal(18,4)")]
        public decimal? LimitPrice { get; set; }

        [Column(TypeName = "decimal(18,4)")]
        public decimal? StopPrice { get; set; }

        [Column(TypeName = "decimal(18,4)")]
        public decimal? AvgFillPrice { get; set; }

        [Required]
        [MaxLength(10)]
        public string TimeInForce { get; set; }

        [Required]
        [MaxLength(20)]
        public string AssetType { get; set; }

        [Required]
        public DateTime CreatedTime { get; set; }

        public DateTime? SubmittedTime { get; set; }

        public DateTime? FilledTime { get; set; }

        public DateTime? ExpirationTime { get; set; }

        [MaxLength(500)]
        public string RejectReason { get; set; }

        [MaxLength(1000)]
        public string Notes { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; }

        [Required]
        public DateTime UpdatedAt { get; set; }

        // Navigation property
        [ForeignKey(nameof(SessionId))]
        public PaperTradingSessionEntity Session { get; set; }
    }
}
```

---

### 2. Update DbContext

**File: `Quantra.DAL/Data/QuantraDbContext.cs`**

Add the new DbSets:

```csharp
public DbSet<PaperTradingPositionEntity> PaperTradingPositions { get; set; }
public DbSet<PaperTradingOrderEntity> PaperTradingOrders { get; set; }
public DbSet<PaperTradingFillEntity> PaperTradingFills { get; set; }  // Optional
```

---

### 3. Create Service Class

**File: `Quantra.DAL/Services/PaperTradingPersistenceService.cs`**

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;

namespace Quantra.DAL.Services
{
    public class PaperTradingPersistenceService
    {
        private readonly string _connectionString;
        private readonly LoggingService _loggingService;

        public PaperTradingPersistenceService(string connectionString, LoggingService loggingService)
        {
            _connectionString = connectionString;
            _loggingService = loggingService;
        }

        private QuantraDbContext CreateDbContext()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(_connectionString);
            return new QuantraDbContext(optionsBuilder.Options);
        }

        // Position Methods
        public async Task SavePositionAsync(Guid sessionId, TradingPosition position);
        public async Task UpdatePositionAsync(Guid sessionId, TradingPosition position);
        public async Task ClosePositionAsync(Guid sessionId, string symbol);
        public async Task<List<TradingPosition>> LoadPositionsAsync(Guid sessionId);

        // Order Methods
        public async Task SaveOrderAsync(Guid sessionId, Order order);
        public async Task UpdateOrderAsync(Guid sessionId, Order order);
        public async Task<List<Order>> LoadOrdersAsync(Guid sessionId);
        public async Task<List<Order>> LoadActiveOrdersAsync(Guid sessionId);

        // Fill Methods (Optional)
        public async Task SaveFillAsync(Guid sessionId, OrderFill fill);
        public async Task<List<OrderFill>> LoadFillsAsync(Guid sessionId);
    }
}
```

---

### 4. Update PaperTradingViewModel

**Changes needed in `Quantra/ViewModels/PaperTradingViewModel.cs`:**

```csharp
private PaperTradingPersistenceService _persistenceService;

public PaperTradingViewModel(IAlphaVantageService alphaVantageService)
{
    // ... existing code ...
    
    _persistenceService = new PaperTradingPersistenceService(
        ConnectionHelper.ConnectionString, 
        new LoggingService());
}

// Add method to restore session
public async Task RestoreSessionAsync(Guid sessionId)
{
    // Load positions from database
    var positions = await _persistenceService.LoadPositionsAsync(sessionId);
    foreach (var position in positions)
    {
        _portfolioManager.RestorePosition(position);
        Positions.Add(position);
    }

    // Load orders from database
    var orders = await _persistenceService.LoadOrdersAsync(sessionId);
    foreach (var order in orders)
    {
        _tradingEngine.RestoreOrder(order);
        Orders.Add(order);
    }

    RefreshPortfolio();
}

// Update event handlers to persist changes
private async void OnOrderFilled(object sender, OrderFilledEventArgs e)
{
    RefreshPortfolio();
    RefreshPositions();
    RefreshOrders();

    // Save order and position to database
    if (_currentSessionId.HasValue)
    {
        await _persistenceService.UpdateOrderAsync(_currentSessionId.Value, e.Order);
        
        var position = _portfolioManager.GetPosition(e.Order.Symbol);
        if (position != null)
        {
            await _persistenceService.SavePositionAsync(_currentSessionId.Value, position);
        }
    }

    await UpdateSessionInDatabaseAsync();
}

private void OnOrderStateChanged(object sender, OrderStateChangedEventArgs e)
{
    RefreshOrders();

    // Persist order state change
    if (_currentSessionId.HasValue)
    {
        _ = _persistenceService.UpdateOrderAsync(_currentSessionId.Value, e.Order);
    }
}
```

---

## ?? Implementation Checklist

### Phase 1: Database Schema (Day 1)
- [ ] Create `PaperTradingPositions` table migration
- [ ] Create `PaperTradingOrders` table migration
- [ ] Create `PaperTradingFills` table migration (optional)
- [ ] Add foreign key constraints
- [ ] Add indexes for performance
- [ ] Test migration on local database

### Phase 2: Entity Models (Day 1)
- [ ] Create `PaperTradingPositionEntity.cs`
- [ ] Create `PaperTradingOrderEntity.cs`
- [ ] Create `PaperTradingFillEntity.cs` (optional)
- [ ] Update `QuantraDbContext.cs` with new DbSets
- [ ] Add navigation properties to `PaperTradingSessionEntity`
- [ ] Test EF Core mappings

### Phase 3: Service Layer (Day 2)
- [ ] Create `PaperTradingPersistenceService.cs`
- [ ] Implement position CRUD operations
- [ ] Implement order CRUD operations
- [ ] Implement fill tracking (optional)
- [ ] Add error handling and logging
- [ ] Write unit tests for service methods

### Phase 4: ViewModel Integration (Day 2-3)
- [ ] Add `_persistenceService` to `PaperTradingViewModel`
- [ ] Implement `RestoreSessionAsync()` method
- [ ] Update `OnOrderFilled()` to persist orders
- [ ] Update `OnOrderStateChanged()` to persist state
- [ ] Update `OnPortfolioChanged()` to persist positions
- [ ] Add position loading on session start
- [ ] Test session restoration flow

### Phase 5: UI Updates (Day 3)
- [ ] Add "Resume Session" option in UI
- [ ] Show loading indicator when restoring session
- [ ] Display session restoration status
- [ ] Test full workflow from UI

### Phase 6: Testing & Validation (Day 4)
- [ ] Test session persistence across app restarts
- [ ] Verify position accuracy after restoration
- [ ] Verify order state preservation
- [ ] Test with multiple simultaneous positions
- [ ] Test edge cases (app crash, partial fills)
- [ ] Performance testing with large order history

### Phase 7: Documentation (Day 4)
- [ ] Update user documentation
- [ ] Add code comments for new methods
- [ ] Document database schema changes
- [ ] Create migration guide

---

## ?? Acceptance Criteria

? **Must Have:**
1. Positions persist to database and restore correctly on session resume
2. Orders persist with full state history (pending ? filled ? etc.)
3. Session can be completely restored after app restart
4. No data loss on application crash
5. Performance remains acceptable (<500ms for typical restore)
6. All financial calculations use `decimal` precision

? **Nice to Have:**
7. Fill-level tracking for detailed execution history
8. Historical position snapshots for performance charting
9. Ability to export positions/orders to CSV
10. Database cleanup for old closed sessions

---

## ?? Testing Scenarios

1. **Basic Persistence**
   - Start session, place order, close app
   - Reopen app, verify order exists

2. **Position Restoration**
   - Open position, close app
   - Reopen app, verify position with correct P&L

3. **Order State Transitions**
   - Place limit order, close app before fill
   - Reopen app, order should still be pending
   - Fill order, verify state update in DB

4. **Crash Recovery**
   - Kill app process during active session
   - Restart app, verify no data corruption

5. **Multiple Sessions**
   - Create multiple sessions
   - Verify positions are session-isolated
   - Delete old session, verify cascade delete

---

## ?? Performance Considerations

- **Batch Updates**: Use batching for multiple position/order updates
- **Async Operations**: All DB operations should be async
- **Connection Pooling**: Reuse DbContext connections
- **Indexes**: Ensure queries on `SessionId`, `Symbol`, `State` are indexed
- **Lazy Loading**: Only load positions/orders when needed

---

## ?? Related Issues

- #XXX - Paper Trading Real-Time Price Updates
- #XXX - Paper Trading Session History & Analytics
- #XXX - Export Paper Trading Results

---

## ?? Implementation Notes

1. **Decimal Precision**: All financial fields MUST use `decimal(18,4)` not `double`
2. **UTC Times**: All timestamps should be stored in UTC
3. **Cascade Deletes**: Use `ON DELETE CASCADE` for session cleanup
4. **Soft Deletes**: Consider soft deletes for positions (set `IsClosed = true`)
5. **Idempotency**: Save operations should be idempotent (check if exists before insert)
6. **Transaction Safety**: Wrap multi-step operations in transactions

---

## ?? Future Enhancements

1. **Position History Snapshots** - Track position value over time for charts
2. **Order Modification History** - Track when orders are modified
3. **Commission Tracking** - Detailed fee calculation and tracking
4. **Multi-User Support** - Add UserId foreign key for user-specific sessions
5. **Cloud Sync** - Sync sessions across devices
6. **Performance Reports** - Generate detailed performance analytics from historical data

---

## ?? References

- Current TradingEngine implementation: `Quantra.DAL/TradingEngine/Core/TradingEngine.cs`
- Current Position model: `Quantra.DAL/TradingEngine/Positions/TradingPosition.cs`
- Current Order model: `Quantra.DAL/TradingEngine/Orders/Order.cs`
- Session service: `Quantra.DAL/Services/PaperTradingSessionService.cs`

---

**Estimated Development Time:** 4-5 days
**Priority:** High
**Complexity:** Medium-High
**Dependencies:** EF Core, SQL Server

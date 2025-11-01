# DatabaseMonolith to EF Core DbContext Migration Guide

## Overview

The `DatabaseMonolith` has been migrated to use **Entity Framework Core** with a `DbContext` pattern while maintaining backward compatibility. This allows for gradual migration of existing code.

## Architecture

### New Structure

```
Quantra.DAL/
??? Data/
?   ??? QuantraDbContext.cs       # Main DbContext
?   ??? QuantraDbContextExtensions.cs    # DI configuration
?   ??? Entities/             # Entity models
?   ?   ??? ConfigurationEntities.cs
?   ?   ??? StockEntities.cs
? ?   ??? TradingEntities.cs
?   ?   ??? PredictionEntities.cs
?   ?   ??? AnalystEntities.cs
?   ??? Configurations/          # EF Core configurations
?   ?   ??? EntityConfigurations.cs
?   ??? Repositories/   # Repository pattern
?       ??? GenericRepository.cs
??? DatabaseMonolith.cs         # Backward compatibility facade
```

## Benefits of EF Core Migration

? **Type Safety** - Strongly typed LINQ queries instead of raw SQL
? **Change Tracking** - Automatic tracking of entity changes
? **Migrations** - Database schema version control
? **Repository Pattern** - Clean separation of concerns
? **Dependency Injection** - Proper DI container integration
? **Testing** - Easy to mock and test
? **Performance** - Connection pooling, query caching, compiled queries

## Setup in Application Startup

### Option 1: Using Dependency Injection (Recommended for new code)

```csharp
// In your application startup (e.g., App.xaml.cs or Startup.cs)
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Data;

var services = new ServiceCollection();

// Add Quantra database services
services.AddQuantraDatabase();

// Add other services
services.AddSingleton<SettingsService>();
// ... other services

var serviceProvider = services.BuildServiceProvider();

// Initialize database
serviceProvider.InitializeQuantraDatabase();

// Use DbContext via DI
var dbContext = serviceProvider.GetRequiredService<QuantraDbContext>();
```

### Option 2: Direct Usage (for migration period)

```csharp
// DatabaseMonolith now uses DbContext internally
var settingsService = new SettingsService();
var database = new DatabaseMonolith(settingsService);

// Works exactly as before - backward compatible
database.Log("Info", "Application started");
```

## Migration Examples

### Example 1: Logging

**Old Way (DatabaseMonolith):**
```csharp
DatabaseMonolith.Log("Info", "User action performed", "Details here");
```

**New Way (EF Core):**
```csharp
// Via DI
public class MyService
{
 private readonly ILogRepository _logRepo;

    public MyService(ILogRepository logRepo)
 {
      _logRepo = logRepo;
    }

    public async Task LogAction()
    {
        var log = new LogEntry
        {
            Level = "Info",
         Message = "User action performed",
            Details = "Details here",
    Timestamp = DateTime.Now
    };
    await _logRepo.AddAsync(log);
    }
}
```

### Example 2: Stock Symbol Queries

**Old Way:**
```csharp
var symbols = DatabaseMonolith.GetAllStockSymbols();
var symbol = DatabaseMonolith.GetStockSymbol("AAPL");
```

**New Way:**
```csharp
public class StockService
{
    private readonly IStockSymbolRepository _stockRepo;

    public StockService(IStockSymbolRepository stockRepo)
    {
        _stockRepo = stockRepo;
    }

    public async Task<IEnumerable<StockSymbolEntity>> GetAllSymbols()
    {
        return await _stockRepo.GetAllAsync();
    }

    public async Task<StockSymbolEntity> GetSymbol(string symbol)
    {
        return await _stockRepo.GetBySymbolAsync(symbol);
}

    // LINQ queries are now possible!
    public async Task<IEnumerable<StockSymbolEntity>> GetTechStocks()
  {
        return await _stockRepo.FindAsync(s => s.Sector == "Technology");
    }
}
```

### Example 3: Trading Rules

**Old Way:**
```csharp
var rules = DatabaseMonolith.GetTradingRules("AAPL");
DatabaseMonolith.SaveTradingRule(rule);
```

**New Way:**
```csharp
public class TradingRuleService
{
    private readonly ITradingRuleRepository _ruleRepo;

  public TradingRuleService(ITradingRuleRepository ruleRepo)
    {
        _ruleRepo = ruleRepo;
    }

    public async Task<IEnumerable<TradingRuleEntity>> GetActiveRules(string symbol)
    {
      return await _ruleRepo.GetActiveRulesAsync(symbol);
    }

    public async Task SaveRule(TradingRuleEntity rule)
    {
        if (rule.Id == 0)
        {
        await _ruleRepo.AddAsync(rule);
}
        else
        {
   await _ruleRepo.UpdateAsync(rule);
        }
    }
}
```

### Example 4: Complex Queries with LINQ

**New capability - Not possible with DatabaseMonolith:**

```csharp
public class AnalyticsService
{
    private readonly QuantraDbContext _context;

    public AnalyticsService(QuantraDbContext context)
    {
        _context = context;
    }

    // Complex query with joins and grouping
  public async Task<Dictionary<string, int>> GetOrderStatsBySymbol()
    {
   return await _context.OrderHistory
   .Where(o => o.Timestamp > DateTime.Now.AddDays(-30))
  .GroupBy(o => o.Symbol)
  .Select(g => new { Symbol = g.Key, Count = g.Count() })
          .ToDictionaryAsync(x => x.Symbol, x => x.Count);
    }

    // Query with navigation properties
    public async Task<IEnumerable<StockPredictionEntity>> GetPredictionsWithIndicators()
    {
      return await _context.StockPredictions
    .Include(p => p.Indicators)
            .Where(p => p.Confidence > 0.7)
            .OrderByDescending(p => p.CreatedDate)
 .Take(10)
            .ToListAsync();
    }

    // Aggregate functions
    public async Task<double> GetAverageConfidence(string symbol)
    {
        return await _context.StockPredictions
  .Where(p => p.Symbol == symbol)
          .AverageAsync(p => p.Confidence);
    }
}
```

## Database Migrations

### Creating a Migration

```bash
# Navigate to the DAL project directory
cd Quantra.DAL

# Add a migration
dotnet ef migrations add InitialCreate

# Apply migrations
dotnet ef database update
```

### Programmatic Migrations

```csharp
// In application startup
using var scope = serviceProvider.CreateScope();
var context = scope.ServiceProvider.GetRequiredService<QuantraDbContext>();

// Apply any pending migrations
await context.Database.MigrateAsync();
```

## Testing

### Unit Testing with In-Memory Database

```csharp
using Microsoft.EntityFrameworkCore;
using Xunit;

public class StockServiceTests
{
    private QuantraDbContext GetInMemoryContext()
  {
        var options = new DbContextOptionsBuilder<QuantraDbContext>()
 .UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString())
            .Options;

        return new QuantraDbContext(options);
    }

    [Fact]
    public async Task CanAddStockSymbol()
    {
        // Arrange
  var context = GetInMemoryContext();
        var repository = new StockSymbolRepository(context);
        var symbol = new StockSymbolEntity
        {
     Symbol = "TEST",
     Name = "Test Company",
    Sector = "Technology"
        };

        // Act
      await repository.AddAsync(symbol);

        // Assert
      var result = await repository.GetBySymbolAsync("TEST");
        Assert.NotNull(result);
     Assert.Equal("Test Company", result.Name);
    }
}
```

## Performance Optimization

### Query Optimization

```csharp
// Use AsNoTracking for read-only queries
var symbols = await _context.StockSymbols
    .AsNoTracking()
    .ToListAsync();

// Use pagination
var page = await _context.OrderHistory
  .OrderByDescending(o => o.Timestamp)
    .Skip((pageNumber - 1) * pageSize)
  .Take(pageSize)
 .ToListAsync();

// Batch operations
var logsToDelete = await _context.Logs
    .Where(l => l.Timestamp < cutoffDate)
    .ToListAsync();
_context.Logs.RemoveRange(logsToDelete);
await _context.SaveChangesAsync();
```

### Compiled Queries (for frequently used queries)

```csharp
private static readonly Func<QuantraDbContext, string, Task<StockSymbolEntity>> 
    GetStockBySymbolCompiled = 
        EF.CompileAsyncQuery((QuantraDbContext context, string symbol) =>
            context.StockSymbols.FirstOrDefault(s => s.Symbol == symbol));

// Use it
var stock = await GetStockBySymbolCompiled(_context, "AAPL");
```

## Migration Timeline

### Phase 1: Setup (Current)
- ? EF Core infrastructure in place
- ? DatabaseMonolith uses DbContext internally
- ? Backward compatibility maintained

### Phase 2: Gradual Migration (Next 2-4 weeks)
- Migrate new features to use DbContext directly
- Refactor high-traffic code paths to use repositories
- Update services one at a time

### Phase 3: Full Migration (1-2 months)
- All new code uses DbContext/Repositories
- Legacy DatabaseMonolith methods marked as [Obsolete]
- Comprehensive test coverage

### Phase 4: Cleanup (Future)
- Remove DatabaseMonolith wrapper
- Pure EF Core implementation

## Best Practices

1. **Always use async methods** - `SaveChangesAsync()`, `ToListAsync()`, etc.
2. **Use AsNoTracking** for read-only queries
3. **Dispose DbContext properly** - Use DI scopes or `using` statements
4. **Use repositories** for common operations
5. **Keep entities simple** - Avoid business logic in entities
6. **Use migrations** for schema changes
7. **Test with in-memory database** for unit tests
8. **Monitor query performance** with logging

## Troubleshooting

### Issue: "No DbContext registered"
**Solution:** Make sure you've called `services.AddQuantraDatabase()` in your startup

### Issue: Migration errors
**Solution:** Delete Quantra.db and let EF Core recreate with `Database.EnsureCreated()`

### Issue: Slow queries
**Solution:** Use `.AsNoTracking()`, add indexes via entity configuration, or use compiled queries

## Resources

- [EF Core Documentation](https://docs.microsoft.com/ef/core/)
- [Repository Pattern](https://docs.microsoft.com/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/infrastructure-persistence-layer-design)
- [SQLite Provider](https://docs.microsoft.com/ef/core/providers/sqlite/)

## Support

For questions or issues with the migration, contact the development team or create an issue in the repository.

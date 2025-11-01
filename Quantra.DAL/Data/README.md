# Quantra Data Layer - EF Core Implementation

## Overview

The Quantra Data Access Layer has been modernized with **Entity Framework Core 9.0**, replacing the monolithic database pattern with a proper ORM-based approach.

## Quick Start

### 1. Install Required Packages

Already included in `Quantra.DAL.csproj`:
- `Microsoft.EntityFrameworkCore` (9.0.0)
- `Microsoft.EntityFrameworkCore.Sqlite` (9.0.0)
- `Microsoft.EntityFrameworkCore.Tools` (9.0.0)
- `Microsoft.EntityFrameworkCore.Design` (9.0.0)

### 2. Configure in Your Application

```csharp
// In App.xaml.cs or Startup.cs
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Data;

public partial class App : Application
{
    public IServiceProvider ServiceProvider { get; private set; }

    protected override void OnStartup(StartupEventArgs e)
  {
        base.OnStartup(e);

        var services = new ServiceCollection();

     // Add Quantra database services
        services.AddQuantraDatabase();

 // Add your other services
        services.AddSingleton<SettingsService>();
        services.AddTransient<ModernLoggingService>();
 services.AddTransient<ModernStockService>();

        // Build service provider
        ServiceProvider = services.BuildServiceProvider();

        // Initialize database
        ServiceProvider.InitializeQuantraDatabase();

     // Show main window
   var mainWindow = ServiceProvider.GetRequiredService<MainWindow>();
        mainWindow.Show();
}
}
```

### 3. Use in Your ViewModels/Services

```csharp
public class MyViewModel
{
 private readonly ModernLoggingService _loggingService;
    private readonly IStockSymbolRepository _stockRepo;

 public MyViewModel(
   ModernLoggingService loggingService,
        IStockSymbolRepository stockRepo)
    {
        _loggingService = loggingService;
 _stockRepo = stockRepo;
    }

    public async Task LoadStocksAsync()
    {
        await _loggingService.LogAsync("Info", "Loading stocks");
  var stocks = await _stockRepo.GetAllAsync();
        // Use stocks...
    }
}
```

## Architecture

### Components

1. **QuantraDbContext** - Main EF Core DbContext
   - Manages database connection and change tracking
   - Provides DbSet<T> for each entity
   - Handles migrations and initialization

2. **Entity Models** - POCOs representing database tables
   - Located in `Data/Entities/`
- Separated by domain (Configuration, Stock, Trading, etc.)
   - Properly annotated with data annotations

3. **Repositories** - Data access abstraction
   - Generic repository for common operations
 - Specialized repositories for complex queries
   - Async-first API

4. **Services** - Business logic layer
 - `ModernLoggingService`, `ModernStockService`, etc.
   - Examples of how to use repositories
   - Can be used as templates for new services

## Key Features

### ?? Performance
- Connection pooling
- Query caching
- Compiled queries support
- Batch operations
- AsNoTracking for read-only queries

### ?? Type Safety
- Strongly typed LINQ queries
- Compile-time checking
- IntelliSense support

### ?? Testability
- Easy mocking with interfaces
- In-memory database for unit tests
- Separation of concerns

### ?? Query Power
```csharp
// Complex queries made easy
var techStocks = await _context.StockSymbols
.Where(s => s.Sector == "Technology")
    .OrderBy(s => s.Symbol)
 .ToListAsync();

// Aggregations
var stats = await _context.OrderHistory
    .GroupBy(o => o.Symbol)
 .Select(g => new
    {
    Symbol = g.Key,
   TotalOrders = g.Count(),
   AveragePrice = g.Average(o => o.Price)
    })
    .ToListAsync();

// Joins with navigation properties
var predictions = await _context.StockPredictions
    .Include(p => p.Indicators)
.Where(p => p.Confidence > 0.7)
    .ToListAsync();
```

## Entity Relationship Diagram

```
???????????????????
?  StockSymbols   ?
???????????????????
         ?
         ????????????????
 ?              ?
??????????????????? ??????????????????
? StockDataCache  ? ? OrderHistory   ?
??????????????????? ??????????????????

????????????????????
? StockPredictions ?
????????????????????
         ?
   ? 1:N
     ?
?????????????????????????
? PredictionIndicators  ?
?????????????????????????

????????????????????
? AnalystRatings   ?
????????????????????

????????????????????
? TradingRules     ?
????????????????????
```

## Available Repositories

### ILogRepository
```csharp
Task<IEnumerable<LogEntry>> GetLogsByLevelAsync(string level, int count = 100);
Task DeleteOldLogsAsync(DateTime olderThan);
```

### IStockSymbolRepository
```csharp
Task<StockSymbolEntity> GetBySymbolAsync(string symbol);
Task<IEnumerable<StockSymbolEntity>> SearchSymbolsAsync(string searchTerm, int limit = 100);
Task<bool> IsSymbolCacheValidAsync(int maxAgeDays = 7);
```

### ITradingRuleRepository
```csharp
Task<IEnumerable<TradingRuleEntity>> GetActiveRulesAsync(string symbol = null);
```

### Generic IRepository<T>
```csharp
Task<T> GetByIdAsync(object id);
Task<IEnumerable<T>> GetAllAsync();
Task<T> AddAsync(T entity);
Task UpdateAsync(T entity);
Task DeleteAsync(T entity);
Task<IEnumerable<T>> FindAsync(Expression<Func<T, bool>> predicate);
```

## Database Migrations

### Add a Migration
```bash
cd Quantra.DAL
dotnet ef migrations add MigrationName
```

### Apply Migrations
```bash
dotnet ef database update
```

### Revert Migration
```bash
dotnet ef database update PreviousMigrationName
```

### Remove Last Migration
```bash
dotnet ef migrations remove
```

## Testing

### Unit Test Example
```csharp
using Microsoft.EntityFrameworkCore;
using Xunit;

public class StockRepositoryTests
{
    private QuantraDbContext CreateInMemoryContext()
    {
        var options = new DbContextOptionsBuilder<QuantraDbContext>()
   .UseInMemoryDatabase(Guid.NewGuid().ToString())
     .Options;
        return new QuantraDbContext(options);
 }

    [Fact]
    public async Task CanAddAndRetrieveStock()
  {
    // Arrange
 using var context = CreateInMemoryContext();
 var repo = new StockSymbolRepository(context);
   var stock = new StockSymbolEntity { Symbol = "TEST", Name = "Test" };

   // Act
     await repo.AddAsync(stock);
        var retrieved = await repo.GetBySymbolAsync("TEST");

        // Assert
   Assert.NotNull(retrieved);
   Assert.Equal("Test", retrieved.Name);
    }
}
```

## Best Practices

### ? Do

```csharp
// Use async/await
var stocks = await _context.StockSymbols.ToListAsync();

// Use AsNoTracking for read-only queries
var data = await _context.Logs.AsNoTracking().ToListAsync();

// Dispose contexts properly (or use DI)
using (var context = new QuantraDbContext(options))
{
 // Work with context
}

// Use repositories for common operations
await _stockRepo.AddAsync(newStock);

// Use LINQ for complex queries
var filtered = await _context.StockSymbols
    .Where(s => s.LastUpdated > DateTime.Now.AddDays(-7))
 .ToListAsync();
```

### ? Don't

```csharp
// Don't use blocking calls
var stocks = _context.StockSymbols.ToList(); // Use ToListAsync()

// Don't forget to save changes
_context.StockSymbols.Add(stock);
// Missing: await _context.SaveChangesAsync();

// Don't load unnecessary data
var all = _context.StockSymbols.ToList(); // Loads everything!
var one = all.First(s => s.Symbol == "AAPL"); // Use FindAsync instead

// Don't create DbContext without disposing
var context = new QuantraDbContext(options);
// Work...
// Never disposed - memory leak!
```

## Performance Tips

1. **Use AsNoTracking** for read-only queries
2. **Batch operations** when possible (AddRange, RemoveRange)
3. **Paginate** large result sets
4. **Use projections** to select only needed columns
5. **Cache** frequently accessed data
6. **Use compiled queries** for repeated queries
7. **Monitor** query performance with logging

## Migration from DatabaseMonolith

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

## Troubleshooting

### "A connection was not closed"
**Solution:** Ensure DbContext is properly disposed. Use DI or `using` statements.

### "No tracking information found"
**Solution:** Don't use AsNoTracking when you need to update entities.

### "The instance of entity type cannot be tracked"
**Solution:** You may have duplicate entities. Use `Update()` method or detach first.

## Resources

- [EF Core Documentation](https://docs.microsoft.com/ef/core/)
- [SQLite Provider](https://docs.microsoft.com/ef/core/providers/sqlite/)
- [Quantra Migration Guide](MIGRATION_GUIDE.md)

## Support

For questions or issues, contact the development team or create an issue.

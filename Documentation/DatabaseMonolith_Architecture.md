# DatabaseMonolith Architecture Documentation

## Overview

The `DatabaseMonolith` class serves as the central database access layer for the Quantra algorithmic trading platform. This document provides comprehensive architectural details, design decisions, and usage patterns for this critical component.

## Design Philosophy

### Why Monolithic?

The DatabaseMonolith class adopts a monolithic design pattern for the following reasons:

1. **Simplified Dependency Management**: Single point of database access eliminates complex dependency injection scenarios
2. **Consistent Connection Handling**: Unified connection string management and database configuration
3. **Centralized Schema Management**: All database schema evolution happens in one place
4. **Performance Optimization**: Single class allows for optimized connection pooling and caching strategies
5. **Easier Testing**: Mock/stub creation is simplified with a single database interface

### Trade-offs

**Benefits:**
- Simple to understand and maintain
- No complex object-relational mapping overhead
- Direct SQL control for performance optimization
- Centralized logging and error handling
- Consistent transaction management

**Drawbacks:**
- Large class size (~3400 lines) can be intimidating
- Not following separation of concerns principles strictly
- Potential for merge conflicts in team environments
- Testing requires mocking the entire database layer

## Architecture Components

### Core Responsibilities

```
DatabaseMonolith
├── Connection Management
│   ├── SQLite Connection Factory
│   ├── WAL Mode Configuration
│   └── Connection String Management
├── Schema Management
│   ├── Database Initialization
│   ├── Table Creation
│   ├── Schema Migration
│   └── Column Addition/Migration
├── Data Access Layers
│   ├── Logging System
│   ├── User Settings & Preferences
│   ├── Trading Data (Orders, History, Rules)
│   ├── Market Data (Quotes, Charts, Analyst Ratings)
│   ├── API Usage Tracking
│   └── Configuration Persistence
└── Utility Functions
    ├── Caching Operations
    ├── Error Handling
    └── Data Validation
```

### Database Schema

The DatabaseMonolith manages multiple tables:

#### Core Application Tables
- **Logs**: Application logging with levels, messages, and context
- **UserAppSettings**: Tab configurations, layouts, and UI state
- **UserCredentials**: Saved login credentials (consider encryption)
- **Settings**: Application preferences and configuration
- **UserPreferences**: Key-value preference storage

#### Trading Tables
- **OrderHistory**: Completed trade records
- **TradingRules**: Automated trading rule definitions
- **TradeRecords**: Detailed execution records

#### Market Data Tables
- **StockSymbols**: Symbol information and caching
- **StockDataCache**: Historical price data with compression
- **AnalystRatings**: Analyst recommendation tracking
- **ConsensusHistory**: Aggregated analyst consensus over time
- **AlphaVantageApiUsage**: API usage tracking for rate limiting

## Key Design Patterns

### Factory Pattern
```csharp
public static SQLiteConnection GetConnection()
{
    if (!initialized)
        Initialize();
    return new SQLiteConnection(ConnectionString);
}
```

### Template Method Pattern
Many methods follow this pattern:
1. Get connection
2. Open connection
3. Ensure table exists
4. Execute operation
5. Handle errors
6. Log results

### Strategy Pattern (Error Handling)
Different error handling strategies based on operation type:
- Database errors: Log and fallback to console
- Schema errors: Auto-migration when possible
- Data errors: Validation and graceful degradation

## Performance Considerations

### Connection Management
- Each method creates its own connection (no connection pooling at this level)
- WAL mode enabled for better concurrency
- Busy timeout set to 30 seconds
- Journal mode optimized for performance

### Data Access Optimization
- Parameterized queries prevent SQL injection
- Compressed data storage for large datasets (StockDataCache)
- Indexed tables for frequently queried data
- Upsert operations to minimize database round trips

### Caching Strategy
- In-memory API key storage
- Compressed JSON for market data
- Timestamp-based cache validation
- User preference caching

## Error Handling Strategy

### Resilient Logging
```csharp
catch (Exception ex)
{
    // If database logging fails, write to console at minimum
    Console.WriteLine($"Failed to log: {level} - {message}");
    Console.WriteLine($"Error: {ex.Message}");
}
```

### Schema Evolution
- Automatic column addition when needed
- Migration from old schema formats (LogLevel → Level)
- Table creation on first access
- Graceful handling of missing tables

### Data Validation
- Null checks on all inputs
- Default value handling
- Type validation before database operations
- Graceful degradation when data is invalid

## Security Considerations

### Current Implementation
- Parameterized queries prevent SQL injection
- Local SQLite database (no network exposure)
- Application-level access control

### Recommendations for Production
1. **Encrypt stored credentials**: Currently stored in plain text
2. **Add database encryption**: SQLite supports encryption extensions
3. **Implement connection security**: For multi-user scenarios
4. **Add audit logging**: Track all database access
5. **Regular backup strategy**: Critical for trading data

## Threading and Concurrency

### Current Approach
- Thread-safe through connection-per-operation pattern
- WAL mode reduces lock contention
- No shared state between operations
- Automatic retry logic for busy database scenarios

### Scalability Considerations
- SQLite suitable for single-application scenarios
- Consider PostgreSQL/SQL Server for multi-user deployments
- Connection pooling would improve performance under load
- Async/await pattern could improve UI responsiveness

## Extension Points

### Adding New Tables
1. Create table in `EnsureDatabaseAndTables()` or dedicated method
2. Add CRUD operations following existing patterns
3. Include error handling and logging
4. Add appropriate indexing for performance

### Schema Migration
Follow the pattern established in `EnsureLogsTableHasLevelColumn()`:
1. Check current schema
2. Create backup if needed
3. Perform migration
4. Validate results
5. Log migration completion

## Testing Strategy

### Current Tests
- Column migration testing (`LoggingTests.cs`)
- Error handling validation
- Schema evolution verification

### Recommended Additional Tests
1. **Performance tests**: Large dataset operations
2. **Concurrency tests**: Multiple connection scenarios  
3. **Data integrity tests**: Transaction rollback scenarios
4. **Migration tests**: All schema evolution paths
5. **Error recovery tests**: Database corruption scenarios

## Usage Examples

### Basic Logging
```csharp
// Simple logging
DatabaseMonolith.Log("Info", "Application started");

// Error with context
try {
    // risky operation
} catch (Exception ex) {
    DatabaseMonolith.LogErrorWithContext(ex, "Operation failed");
}
```

### Trading Operations
```csharp
// Save completed trade
var order = new OrderModel {
    Symbol = "AAPL",
    OrderType = "BUY", 
    Quantity = 100,
    Price = 150.50
};
DatabaseMonolith.AddOrderToHistory(order);

// Manage trading rules
var rule = new TradingRule {
    Name = "RSI Oversold",
    Symbol = "AAPL",
    Conditions = new List<string> { "RSI < 30" }
};
DatabaseMonolith.SaveTradingRule(rule);
```

### Settings Management
```csharp
// Load/save user preferences
var settings = DatabaseMonolith.GetUserSettings();
settings.EnableDarkMode = true;
DatabaseMonolith.SaveUserSettings(settings);
```

## Future Enhancements

### Short Term
1. Add async/await support for better UI responsiveness
2. Implement connection pooling for performance
3. Add more comprehensive error recovery
4. Expand caching mechanisms

### Long Term
1. Consider decomposition into specialized DAOs
2. Add database encryption for sensitive data
3. Implement audit trail functionality
4. Support for multiple database backends

## Conclusion

The DatabaseMonolith class effectively serves as a centralized database layer for the Quantra platform. While it violates some separation of concerns principles, it provides excellent practical benefits in terms of simplicity, performance, and maintainability for a single-user trading application.

The monolithic design is appropriate for the current scope but should be reevaluated if the application evolves toward multi-user or distributed scenarios.
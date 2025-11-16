# Tab Configuration and Trading Rule Service Migration

## Summary

Successfully migrated tab configuration, control placement, and trading rule deletion methods from `DatabaseMonolith` to dedicated services. This is part of the ongoing refactoring to move away from the monolithic database pattern to a proper service-oriented architecture using Entity Framework Core.

## Changes Made

### 1. Created New Service: `TabConfigurationService.cs`

**Location:** `Quantra.DAL/Services/TabConfigurationService.cs`

This service provides the following methods:
- `LoadControlsConfig(string tabName)` - Loads controls configuration for a specific tab
- `SaveControlsConfig(string tabName, string controlsConfig)` - Saves controls configuration
- `AddCustomControlWithSpans(...)` - Adds a custom control with span configuration
- `UpdateControlPosition(...)` - Updates the position of a control
- `RemoveControl(string tabName, int controlIndex)` - Removes a control from a tab

### 2. Updated `DatabaseMonolith.cs`

Added static facade methods to maintain backward compatibility:
- `LoadGridConfig(string tabName)` - Returns grid dimensions (rows, columns)
- `LoadControlsConfig(string tabName)` - Static wrapper
- `SaveControlsConfig(string tabName, string controlsConfig)` - Static wrapper
- `AddCustomControlWithSpans(...)` - Static wrapper
- `UpdateControlPosition(...)` - Static wrapper
- `RemoveControl(string tabName, int controlIndex)` - Static wrapper
- `LoadCardPositions()` - Stub for future implementation
- `SaveCardPositions(string cardPositions)` - Stub for future implementation
- `DeleteRule(int ruleId)` - Static wrapper for TradingRuleService.DeleteTradingRuleAsync

All methods are marked as `[Obsolete]` to encourage migration to dependency injection.

### 3. Updated `TabManager.cs`

Modified `LoadTabControls` method to use `DatabaseMonolith.LoadControlsConfig` and `DatabaseMonolith.LoadGridConfig` methods.

### 4. Updated Logging in DeleteRule

The `DeleteRule` method now uses `LoggingService.LogErrorWithContext` instead of the old `DatabaseMonolith.LogErrorWithContext`, providing better error tracking and context.

## Migration Path for Consuming Code

### Before (using DatabaseMonolith):
```csharp
DatabaseMonolith.AddCustomControlWithSpans(tabName, controlType, row, column, rowSpan, columnSpan);
DatabaseMonolith.UpdateControlPosition(tabName, controlIndex, row, column, rowSpan, columnSpan);
var config = DatabaseMonolith.LoadControlsConfig(tabName);
var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);
DatabaseMonolith.DeleteRule(ruleId);
```

### After (using dependency injection):
```csharp
public class MyClass
{
    private readonly TabConfigurationService _tabConfigService;
    private readonly ITradingRuleService _tradingRuleService;
    
    public MyClass(TabConfigurationService tabConfigService, ITradingRuleService tradingRuleService)
    {
        _tabConfigService = tabConfigService;
        _tradingRuleService = tradingRuleService;
    }
    
    public async Task MyMethod()
    {
        _tabConfigService.AddCustomControlWithSpans(tabName, controlType, row, column, rowSpan, columnSpan);
        _tabConfigService.UpdateControlPosition(tabName, controlIndex, row, column, rowSpan, columnSpan);
        var config = _tabConfigService.LoadControlsConfig(tabName);
        
        // For grid config, you'll need to query the database directly or extend the service
        // var gridConfig = _tabConfigService.LoadGridConfig(tabName);
        
        // Trading rule deletion
        await _tradingRuleService.DeleteTradingRuleAsync(ruleId);
    }
}
```

## Benefits

1. **Separation of Concerns** - Tab configuration and trading rule logic are now isolated in their own services
2. **Testability** - Services can be easily mocked for unit testing
3. **Maintainability** - Clear boundaries make the code easier to understand and modify
4. **Scalability** - Services can be extended without affecting other parts of the system
5. **Entity Framework Integration** - Properly uses EF Core for database operations
6. **Better Logging** - Uses `LoggingService` for consistent error tracking

## Next Steps

1. Gradually migrate consuming code to use dependency injection
2. Eventually remove the static facade methods from `DatabaseMonolith`
3. Consider creating interfaces (`ITabConfigurationService`) for better abstraction
4. Add unit tests for the new service
5. Migrate remaining DatabaseMonolith methods to dedicated services

## Testing

The changes compile successfully. The static facade methods ensure backward compatibility, so existing code will continue to work without modification.

## Files Modified

- ? Created: `Quantra.DAL/Services/TabConfigurationService.cs`
- ? Modified: `Quantra.DAL/DatabaseMonolith.cs` (added static facade methods)
- ? Modified: `Quantra/Utilities/TabManager.cs` (uses DatabaseMonolith methods)

## Notes

- All static methods in `DatabaseMonolith` are marked as obsolete to encourage migration
- The service uses proper error handling and logging via `LoggingService`
- Control configuration format: `"ControlType,Row,Column,RowSpan,ColSpan"` separated by semicolons
- `DeleteRule` now properly logs using `LoggingService.LogErrorWithContext` for better error tracking
- `TabManager` continues to use static methods from `DatabaseMonolith` for backward compatibility

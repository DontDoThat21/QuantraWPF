# DatabaseMonolith Migration TODO

This document tracks the remaining work to fully migrate away from `DatabaseMonolith` static methods.

## ? Completed

- [x] Implement `UserSettingsService` with EF Core persistence
- [x] Implement user preferences storage (`UserPreference` entity)
- [x] Implement account credentials storage (`UserCredential` entity)
- [x] Add backward-compatible static facade methods
- [x] Mark `DatabaseMonolith` as obsolete
- [x] Document refactoring and migration path
- [x] Add proper error handling and logging

## ?? In Progress

### High Priority: Update Direct Callers

Search for and update these static method calls throughout the codebase:

#### 1. UserSettings Methods
```csharp
// Search patterns:
- DatabaseMonolith.SaveUserSettings
- DatabaseMonolith.GetUserSettings
```

**Files likely affected:**
- `Quantra/Views/Configuration/ConfigurationControl.xaml.cs`
- `Quantra/Views/MainWindow/MainWindow.xaml.cs`
- `Quantra/ViewModels/*ViewModel.cs`
- Settings-related UI components

**Migration example:**
```csharp
// BEFORE
var settings = DatabaseMonolith.GetUserSettings();
settings.EnableDarkMode = true;
DatabaseMonolith.SaveUserSettings(settings);

// AFTER (inject UserSettingsService in constructor)
var settings = _userSettingsService.GetUserSettings();
settings.EnableDarkMode = true;
_userSettingsService.SaveUserSettings(settings);
```

#### 2. User Preference Methods
```csharp
// Search patterns:
- DatabaseMonolith.SaveUserPreference
- DatabaseMonolith.GetUserPreference
```

**Files likely affected:**
- UI components that save individual preferences
- Tab configuration code
- Layout management code

#### 3. Account Credential Methods
```csharp
// Search patterns:
- DatabaseMonolith.GetRememberedAccounts
- DatabaseMonolith.RememberAccount
```

**Files affected:**
- `Quantra/Views/LoginWindow/LoginWindow.xaml.cs` ? (already uses static method, needs DI update)

### Medium Priority: Service Registration

Update dependency injection configuration:

```csharp
// In App.xaml.cs or Startup.cs
services.AddDbContext<QuantraDbContext>(options =>
    options.UseSqlServer(ConnectionHelper.ConnectionString));

// Register services
services.AddScoped<UserSettingsService>();
services.AddScoped<OrderHistoryService>();  // If not already registered
services.AddScoped<TradeRecordService>();   // If not already registered
```

### Low Priority: Remove DatabaseMonolith Instance Methods

The following instance methods in `DatabaseMonolith` should be migrated to dedicated services:

#### 1. Logging Method
```csharp
// CURRENT: DatabaseMonolith.Log(level, message, details)
// MIGRATE TO: _loggingService.Log (already exists)
```

**Action:** Search and replace all `databaseMonolith.Log(...)` calls with `_loggingService.Log(...)`

#### 2. Order History Method
```csharp
// CURRENT: DatabaseMonolith.AddOrderToHistory(order)
// MIGRATE TO: OrderHistoryService (should exist or be created)
```

**Files to check:**
- Trading-related services
- Order execution code
- Paper trading simulator

#### 3. Trade Record Method
```csharp
// CURRENT: DatabaseMonolith.SaveTradeRecord(trade)
// MIGRATE TO: TradeRecordService (already exists)
```

**Files to check:**
- `Quantra.DAL/Services/TradeRecordService.cs` ? (already exists)
- Update callers to use service instead of DatabaseMonolith

## ?? Migration Checklist by File

### 1. LoginWindow.xaml.cs
**Current Usage:**
```csharp
- DatabaseMonolith.GetRememberedAccounts()
- DatabaseMonolith.RememberAccount(username, password, pin)
```

**Action Required:**
- [ ] Inject `UserSettingsService` into `LoginWindow`
- [ ] Update `LoadRememberedAccounts()` to use service
- [ ] Update `LoginButton_Click()` to use service
- [ ] Test login with remembered accounts

### 2. MainWindow / Configuration Controls
**Likely Usage:**
```csharp
- DatabaseMonolith.GetUserSettings()
- DatabaseMonolith.SaveUserSettings()
- DatabaseMonolith.GetUserPreference()
- DatabaseMonolith.SaveUserPreference()
```

**Action Required:**
- [ ] Search for static calls in all ViewModels
- [ ] Inject `UserSettingsService` via constructor
- [ ] Update all calls to use instance methods
- [ ] Test settings persistence

### 3. Services Using DatabaseMonolith
**Files to check:**
```bash
# Search for DatabaseMonolith usage
- Quantra.DAL/Services/*.cs
- Quantra/Views/**/*.cs
- Quantra/ViewModels/**/*.cs
```

**Action Required:**
- [ ] Identify all services using DatabaseMonolith
- [ ] Update to use specific services (UserSettingsService, etc.)
- [ ] Remove DatabaseMonolith dependencies

## ?? Testing Checklist

### Unit Tests
- [ ] Test `UserSettingsService.SaveUserSettings` with EF Core
- [ ] Test `UserSettingsService.GetUserSettings` returns correct data
- [ ] Test `UserSettingsService.GetUserPreference` with null values
- [ ] Test `UserSettingsService.SaveUserPreference` with special characters
- [ ] Test `UserSettingsService.GetRememberedAccounts` with multiple accounts
- [ ] Test `UserSettingsService.RememberAccount` updates existing accounts
- [ ] Test error handling when database is unavailable

### Integration Tests
- [ ] Test settings persistence across application restarts
- [ ] Test login with remembered accounts
- [ ] Test concurrent access to UserPreferences table
- [ ] Test settings migration from old format (if applicable)
- [ ] Performance test with large settings objects

### Manual Testing
- [ ] Save and load user settings in Configuration UI
- [ ] Test dark mode toggle persistence
- [ ] Test remembered account login
- [ ] Test preference-based features (alerts, notifications, etc.)
- [ ] Test error messages display correctly

## ?? Security Improvements (Optional)

### Credential Encryption
```csharp
// Current: Passwords stored as plain text in UserCredentials
// Recommended: Add encryption for sensitive fields

public class UserCredential
{
    [Encrypted] // Custom attribute
    public string Password { get; set; }
    
    [Encrypted]
    public string Pin { get; set; }
}
```

**Action Required:**
- [ ] Implement encryption for passwords in `UserCredential`
- [ ] Use Data Protection API or similar
- [ ] Migration strategy for existing credentials

### API Key Storage
```csharp
// Current: API keys stored in UserSettings JSON
// Recommended: Separate secure storage

// Option 1: Dedicated ApiKeys table with encryption
// Option 2: Use Windows Credential Manager
// Option 3: Use Azure Key Vault for production
```

**Action Required:**
- [ ] Evaluate security requirements
- [ ] Implement secure API key storage
- [ ] Migrate existing API keys

## ?? Performance Optimizations (Future)

### Caching Layer
```csharp
public class CachedUserSettingsService : IUserSettingsService
{
    private readonly UserSettingsService _innerService;
    private readonly IMemoryCache _cache;
    
    public UserSettings GetUserSettings()
    {
        return _cache.GetOrCreate("UserSettings", entry =>
     {
entry.AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(10);
        return _innerService.GetUserSettings();
        });
    }
}
```

**Benefits:**
- Reduces database calls for frequently accessed settings
- Improves UI responsiveness
- Automatic cache invalidation

**Action Required:**
- [ ] Implement caching decorator
- [ ] Register in DI container
- [ ] Configure cache expiration policies
- [ ] Add cache invalidation on updates

### Async Methods
```csharp
public async Task<UserSettings> GetUserSettingsAsync()
{
    var preference = await _dbContext.UserPreferences
        .FirstOrDefaultAsync(p => p.Key == USER_SETTINGS_KEY);
    // ...
}
```

**Benefits:**
- Better scalability for web-based scenarios
- Non-blocking UI operations
- Improved resource utilization

**Action Required:**
- [ ] Add async versions of all service methods
- [ ] Update callers to use async/await
- [ ] Test with concurrent operations

## ?? Success Criteria

The migration is complete when:
1. ? No compiler warnings about obsolete methods in production code
2. ? All unit tests pass
3. ? Integration tests verify EF Core persistence
4. ? Manual testing confirms all features work
5. ? Performance is equal or better than before
6. ? Code coverage >= 80% for new service methods
7. ? Documentation is complete and accurate
8. ? DatabaseMonolith can be safely removed (or kept as deprecated legacy code)

## ?? Notes

### Backward Compatibility
The static methods will remain for at least one major version to allow gradual migration. After all code is migrated, we can:
1. Remove the static facade methods
2. Make `DatabaseMonolith` internal
3. Eventually remove the class entirely

### Migration Timeline
- **Phase 1** (Current): Implement services with backward compatibility ?
- **Phase 2** (Next): Update high-traffic code paths to use DI
- **Phase 3** (Future): Update remaining code and remove static methods
- **Phase 4** (Final): Remove DatabaseMonolith entirely

### Known Issues
None identified yet. Report any issues during migration.

## ?? Related Documentation
- [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) - Detailed refactoring overview
- [EF Core Documentation](https://docs.microsoft.com/ef/core/) - Microsoft EF Core docs
- [Dependency Injection in .NET](https://docs.microsoft.com/dotnet/core/extensions/dependency-injection) - DI patterns

---

**Last Updated:** 2024
**Status:** In Progress
**Owner:** Development Team

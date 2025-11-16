# API Configuration Service Migration Guide

## Overview

The `SetConfiguration` method has been moved from `DatabaseMonolith` to a dedicated `ApiConfigurationService`. This provides better separation of concerns and follows proper dependency injection patterns.

## What Changed

### Before (DatabaseMonolith)
```csharp
var dbMonolith = new DatabaseMonolith(settingsService);
dbMonolith.SetConfiguration(configuration);
var apiKey = dbMonolith.AlphaVantageApiKey;
```

### After (ApiConfigurationService)
```csharp
// Via dependency injection
public class MyService
{
    private readonly IApiConfigurationService _apiConfigService;
    
    public MyService(IApiConfigurationService apiConfigService)
    {
        _apiConfigService = apiConfigService;
    }
    
    public void UseApiKey()
    {
        var apiKey = _apiConfigService.AlphaVantageApiKey;
    }
}
```

## Registering the Service

### In ServiceCollectionExtensions.cs

Add the following to your service registration:

```csharp
services.AddSingleton<IApiConfigurationService, ApiConfigurationService>();
```

## API Key Priority

The service loads API keys from multiple sources in this priority order:

1. **Environment Variables** (highest priority)
   - `ALPHA_VANTAGE_API_KEY`
   - `NEWS_API_KEY`
   - `OPENAI_API_KEY` or `CHATGPT_API_KEY`

2. **IConfiguration** (appsettings.json, user secrets, etc.)
   - `Api:AlphaVantageApiKey`
   - `Api:NewsApiKey`
   - `Api:OpenAiApiKey`

3. **Legacy JSON File** (alphaVantageSettings.json) - for backward compatibility

## Configuration Examples

### appsettings.json
```json
{
  "Api": {
    "AlphaVantageApiKey": "YOUR_ALPHA_VANTAGE_KEY",
    "NewsApiKey": "YOUR_NEWS_API_KEY",
    "OpenAiApiKey": "YOUR_OPENAI_KEY"
  }
}
```

### Environment Variables (Recommended for Production)
```bash
# Windows
set ALPHA_VANTAGE_API_KEY=your_key_here
set NEWS_API_KEY=your_key_here
set OPENAI_API_KEY=your_key_here

# Linux/Mac
export ALPHA_VANTAGE_API_KEY=your_key_here
export NEWS_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

### User Secrets (Recommended for Development)
```bash
dotnet user-secrets set "Api:AlphaVantageApiKey" "your_key_here"
dotnet user-secrets set "Api:NewsApiKey" "your_key_here"
dotnet user-secrets set "Api:OpenAiApiKey" "your_key_here"
```

## Using the Service

### Basic Usage
```csharp
public class MarketDataService
{
    private readonly IApiConfigurationService _apiConfig;
    
    public MarketDataService(IApiConfigurationService apiConfig)
    {
        _apiConfig = apiConfig;
        
        // Validate keys on startup
        if (!_apiConfig.ValidateApiKeys())
        {
            // Handle missing keys
        }
    }
    
    public async Task GetQuoteAsync(string symbol)
    {
        var apiKey = _apiConfig.AlphaVantageApiKey;
        // Use the API key...
    }
}
```

### Refreshing API Keys
```csharp
// Refresh keys from all sources (useful after configuration changes)
_apiConfig.RefreshApiKeys();
```

### Validation
```csharp
// Check if required API keys are configured
if (_apiConfig.ValidateApiKeys())
{
    Console.WriteLine("All required API keys are configured");
}
else
{
    Console.WriteLine("Some required API keys are missing");
}
```

## Migration Steps for Existing Code

1. **Replace DatabaseMonolith API Key Access**
   ```csharp
   // Old
   var apiKey = databaseMonolith.AlphaVantageApiKey;
   
   // New
   var apiKey = _apiConfigService.AlphaVantageApiKey;
   ```

2. **Remove SetConfiguration Calls**
   ```csharp
   // Old - remove this
   databaseMonolith.SetConfiguration(configuration);
   
   // New - inject IApiConfigurationService instead
   public MyClass(IApiConfigurationService apiConfigService)
   {
       // Service handles configuration automatically
   }
   ```

3. **Update Service Constructors**
   ```csharp
   // Old
   public AlphaVantageService(IUserSettingsService userSettings)
   {
       _apiKey = GetApiKey(); // Static method
   }
   
   // New
   public AlphaVantageService(
       IUserSettingsService userSettings,
       IApiConfigurationService apiConfig)
   {
       _apiKey = apiConfig.AlphaVantageApiKey;
   }
   ```

## Benefits of the New Approach

1. **Better Separation of Concerns** - Configuration logic is separate from database logic
2. **Dependency Injection** - Proper DI pattern for testability and maintainability
3. **Centralized API Key Management** - Single source of truth for all API keys
4. **Environment-Aware** - Supports different configurations per environment
5. **Secure** - Supports user secrets and environment variables for sensitive data
6. **Legacy Support** - Still supports old alphaVantageSettings.json for backward compatibility

## Security Best Practices

1. **Never commit API keys to source control**
2. **Use environment variables in production**
3. **Use user secrets for local development**
4. **Rotate API keys regularly**
5. **Use least-privilege API keys when possible**

## Troubleshooting

### API Keys Not Loading
1. Check the order of priority (environment > configuration > legacy)
2. Review logs for warnings about missing keys
3. Verify configuration file syntax
4. Ensure environment variables are set correctly

### Legacy Settings Not Migrating
1. Check that `alphaVantageSettings.json` exists and is valid JSON
2. Review logs for migration errors
3. Ensure the service has read access to the file

### Validation Failing
```csharp
// Enable detailed logging
if (!_apiConfig.ValidateApiKeys())
{
    Console.WriteLine($"Alpha Vantage: {_apiConfig.AlphaVantageApiKey}");
    Console.WriteLine($"News API: {_apiConfig.NewsApiKey}");
    Console.WriteLine($"OpenAI: {_apiConfig.OpenAiApiKey}");
}
```

## Testing

### Unit Test Example
```csharp
[Test]
public void ApiConfigurationService_LoadsFromEnvironment()
{
    // Arrange
    Environment.SetEnvironmentVariable("ALPHA_VANTAGE_API_KEY", "test_key");
    var config = new ConfigurationBuilder().Build();
    
    // Act
    var service = new ApiConfigurationService(config);
    
    // Assert
    Assert.AreEqual("test_key", service.AlphaVantageApiKey);
    
    // Cleanup
    Environment.SetEnvironmentVariable("ALPHA_VANTAGE_API_KEY", null);
}
```

## Related Files

- `ApiConfigurationService.cs` - Implementation
- `IApiConfigurationService.cs` - Interface
- `DatabaseMonolith.cs` - Legacy facade (SetConfiguration removed)
- `AlphaVantageService.cs` - Should be updated to use IApiConfigurationService
- `ServiceCollectionExtensions.cs` - DI registration

## Support

For issues or questions about API configuration, please:
1. Check this migration guide
2. Review the inline documentation in the service
3. Check the GitHub issues for similar problems
4. Create a new issue with details about your configuration

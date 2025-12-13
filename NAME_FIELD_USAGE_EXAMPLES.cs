// Example: How to initialize/update the StockSymbols cache with names from LISTING_STATUS

// Option 1: During application startup (e.g., in App.xaml.cs or MainWindow initialization)
public async Task InitializeStockSymbolCache()
{
    try
    {
        var alphaVantageService = ServiceLocator.Current.GetService<AlphaVantageService>();
        var stockSymbolCacheService = ServiceLocator.Current.GetService<StockSymbolCacheService>();
        
        // Check if cache is stale (older than 7 days)
        if (!stockSymbolCacheService.IsSymbolCacheValid(maxAgeDays: 7))
        {
            Console.WriteLine("StockSymbols cache is stale or empty. Refreshing from LISTING_STATUS API...");
            await alphaVantageService.CacheSymbolsWithNamesAsync();
        }
        else
        {
            Console.WriteLine("StockSymbols cache is up to date.");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Failed to initialize stock symbol cache: {ex.Message}");
    }
}

// Option 2: Manual refresh via UI button or maintenance task
public async Task RefreshStockSymbolCache()
{
    var alphaVantageService = ServiceLocator.Current.GetService<AlphaVantageService>();
    await alphaVantageService.CacheSymbolsWithNamesAsync();
    MessageBox.Show("Stock symbol cache refreshed successfully!");
}

// Option 3: Scheduled background task (e.g., weekly refresh)
public async Task ScheduledCacheRefresh()
{
    var timer = new System.Timers.Timer(TimeSpan.FromDays(7).TotalMilliseconds);
    timer.Elapsed += async (sender, e) =>
    {
        try
        {
            var alphaVantageService = ServiceLocator.Current.GetService<AlphaVantageService>();
            await alphaVantageService.CacheSymbolsWithNamesAsync();
            Console.WriteLine("Scheduled cache refresh completed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Scheduled cache refresh failed: {ex.Message}");
        }
    };
    timer.Start();
}

// Example: Verify that names are being stored correctly
public async Task VerifyStockExplorerDataNames()
{
    var stockExplorerDataService = ServiceLocator.Current.GetService<StockExplorerDataService>();
    var allData = await stockExplorerDataService.GetAllStockDataAsync();
    
    var withNames = allData.Where(d => !string.IsNullOrEmpty(d.Name)).Count();
    var withoutNames = allData.Where(d => string.IsNullOrEmpty(d.Name)).Count();
    
    Console.WriteLine($"StockExplorerData: {withNames} records with names, {withoutNames} without names");
    
    if (withoutNames > 0)
    {
        Console.WriteLine("Some records are missing names. Consider refreshing the cache.");
    }
}

// Example: Get symbol info with guaranteed name (uses cache or fetches)
public async Task<string> GetSymbolName(string symbol)
{
    // Try cache first
    var stockSymbolCacheService = ServiceLocator.Current.GetService<StockSymbolCacheService>();
    var cachedSymbol = stockSymbolCacheService.GetStockSymbol(symbol);
    
    if (cachedSymbol != null && !string.IsNullOrEmpty(cachedSymbol.Name))
    {
        return cachedSymbol.Name;
    }
    
    // Fallback to API call
    var alphaVantageService = ServiceLocator.Current.GetService<AlphaVantageService>();
    var quoteData = await alphaVantageService.GetQuoteDataAsync(symbol);
    
    return quoteData?.Name ?? "Unknown";
}

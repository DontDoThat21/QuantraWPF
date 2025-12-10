# Options Explorer - Integration Guide

## ?? Quick Start Integration

### Step 1: Register Services in App.xaml.cs

Update your `App.xaml.cs` to register the new ViewModel and ensure all required services are available:

```csharp
using Microsoft.Extensions.DependencyInjection;
using Quantra.ViewModels;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;

namespace Quantra
{
    public partial class App : Application
    {
        public IServiceProvider ServiceProvider { get; private set; }

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            
            var services = new ServiceCollection();
            ConfigureServices(services);
            ServiceProvider = services.BuildServiceProvider();
            
            // Your existing startup code...
        }

        private void ConfigureServices(IServiceCollection services)
        {
            // Core Services (if not already registered)
            services.AddSingleton<LoggingService>();
            services.AddSingleton<UserSettingsService>();
            
            // Data Services
            services.AddSingleton<IAlphaVantageService, AlphaVantageService>();
            services.AddSingleton<OptionsDataService>();
            services.AddSingleton<IStockDataCacheService, StockDataCacheService>();
            
            // Options-specific Services
            services.AddSingleton<GreekCalculationEngine>();
            services.AddSingleton<IVSurfaceService>();
            services.AddSingleton<OptionsPricingService>();
            
            // ViewModels
            services.AddTransient<OptionsViewModel>();
            
            // Keep existing OptionsExplorerViewModel for backward compatibility
            services.AddTransient<OptionsExplorerViewModel>();
        }
    }
}
```

### Step 2: Update OptionsExplorer.xaml.cs Code-Behind

**Option A: Use New ViewModel (Recommended)**

```csharp
using System.Windows.Controls;
using Microsoft.Extensions.DependencyInjection;
using Quantra.ViewModels;

namespace Quantra.Views.OptionsExplorer
{
    public partial class OptionsExplorer : UserControl
    {
        public OptionsExplorer()
        {
            InitializeComponent();
            
            // Get ViewModel from DI container
            var app = (App)Application.Current;
            DataContext = app.ServiceProvider.GetRequiredService<OptionsViewModel>();
        }
    }
}
```

**Option B: Keep Both ViewModels (For Gradual Migration)**

```csharp
using System.Windows.Controls;
using Microsoft.Extensions.DependencyInjection;
using Quantra.ViewModels;

namespace Quantra.Views.OptionsExplorer
{
    public partial class OptionsExplorer : UserControl
    {
        private readonly OptionsViewModel _newViewModel;
        private readonly OptionsExplorerViewModel _legacyViewModel;
        
        public OptionsExplorer()
        {
            InitializeComponent();
            
            var app = (App)Application.Current;
            
            // Use new ViewModel by default
            _newViewModel = app.ServiceProvider.GetRequiredService<OptionsViewModel>();
            _legacyViewModel = app.ServiceProvider.GetRequiredService<OptionsExplorerViewModel>();
            
            // Switch between ViewModels based on user preference or feature flag
            bool useNewViewModel = true; // TODO: Make this configurable
            
            DataContext = useNewViewModel ? _newViewModel : (object)_legacyViewModel;
        }
    }
}
```

### Step 3: Update XAML Bindings (If Needed)

Your existing `OptionsExplorer.xaml` should work with the new ViewModel since property names match. However, verify these key bindings:

```xaml
<!-- Symbol Input -->
<TextBox Text="{Binding SelectedSymbol, UpdateSourceTrigger=PropertyChanged}"/>

<!-- Expiration Selector -->
<ComboBox ItemsSource="{Binding ExpirationDates}"
          SelectedItem="{Binding SelectedExpiration}"
          DisplayMemberPath="{}{0:MMM dd, yyyy}"/>

<!-- Commands -->
<Button Command="{Binding LoadOptionsChainCommand}" Content="Load"/>
<Button Command="{Binding RefreshDataCommand}" Content="Refresh"/>
<Button Command="{Binding BuildIVSurfaceCommand}" Content="Build IV Surface"/>

<!-- Options Chains -->
<DataGrid ItemsSource="{Binding CallOptions}"
          SelectedItem="{Binding SelectedOption}"
          RowStyle="{StaticResource OptionsDataGridRowStyle}"/>

<DataGrid ItemsSource="{Binding PutOptions}"
          SelectedItem="{Binding SelectedOption}"
          RowStyle="{StaticResource OptionsDataGridRowStyle}"/>

<!-- Status -->
<TextBlock Text="{Binding StatusMessage}"/>
<ProgressBar IsIndeterminate="{Binding IsLoading}"
             Visibility="{Binding IsLoading, Converter={StaticResource BoolToVisibilityConverter}}"/>
```

### Step 4: Property Name Mapping

If your existing XAML uses different property names, here's the mapping:

| Old Property (OptionsExplorerViewModel) | New Property (OptionsViewModel) |
|----------------------------------------|----------------------------------|
| `Symbol` | `SelectedSymbol` |
| `CurrentPrice` | `UnderlyingPrice` |
| `IVSkew` | `IVSurface` (different structure) |
| `PortfolioOptions` | `SelectedLegs` |

Update any mismatched bindings in your XAML.

### Step 5: Test the Integration

Create a simple test to verify everything works:

```csharp
// In OptionsExplorer.xaml.cs or a test class
public void TestViewModel()
{
    var viewModel = DataContext as OptionsViewModel;
    
    // Test 1: Set symbol
    viewModel.SelectedSymbol = "AAPL";
    
    // Test 2: Wait for data to load
    await Task.Delay(2000);
    
    // Test 3: Verify data loaded
    Assert.IsTrue(viewModel.ExpirationDates.Count > 0, "Expirations should load");
    Assert.IsTrue(viewModel.UnderlyingPrice > 0, "Price should be set");
    
    // Test 4: Select expiration
    viewModel.SelectedExpiration = viewModel.ExpirationDates.FirstOrDefault();
    
    // Test 5: Wait for options chain
    await Task.Delay(2000);
    
    // Test 6: Verify chains loaded
    Assert.IsTrue(viewModel.CallOptions.Count > 0, "Calls should load");
    Assert.IsTrue(viewModel.PutOptions.Count > 0, "Puts should load");
}
```

## ?? Troubleshooting

### Issue: ViewModel not injecting

**Solution**: Ensure `ServiceProvider` is public in `App.xaml.cs` and services are registered in `OnStartup`.

### Issue: Commands not firing

**Solution**: 
1. Check `RelayCommand` implementation exists
2. Verify `CanExecute` logic isn't blocking
3. Add debug breakpoints in command handlers

### Issue: Data not loading

**Solution**:
1. Check Alpha Vantage API key is set
2. Verify `OptionsDataService` is properly configured
3. Check `LoggingService` output for errors
4. Ensure database connection string is correct

### Issue: UI not updating

**Solution**:
1. Verify `INotifyPropertyChanged` is implemented
2. Check `OnPropertyChanged()` is called for all property setters
3. Use `ObservableCollection` for collections
4. Ensure UI updates on UI thread (use `Dispatcher` if needed)

## ?? Migration Checklist

- [ ] Services registered in DI container
- [ ] ViewModel injected in code-behind
- [ ] XAML bindings updated (if needed)
- [ ] Command bindings verified
- [ ] Data loading tested
- [ ] Filter functionality tested
- [ ] Greeks calculation tested
- [ ] Multi-leg position builder tested
- [ ] Error handling verified
- [ ] UI responsiveness checked
- [ ] Performance tested with large chains

## ?? Feature Activation

Once basic integration is complete, enable advanced features:

### 1. IV Surface Visualization

```csharp
// In XAML
<TabItem Header="IV Surface">
    <Grid>
        <!-- Add 3D chart control here -->
        <Button Command="{Binding BuildIVSurfaceCommand}" 
                Content="Build Surface" 
                IsEnabled="{Binding DataAvailable}"/>
    </Grid>
</TabItem>
```

### 2. Theoretical Pricing

```csharp
// Add button to toolbar
<Button Command="{Binding CalculateTheoreticalPriceCommand}"
        Content="Calculate Fair Value"
        ToolTip="Calculate Black-Scholes theoretical prices"/>

// Display in DataGrid
<DataGridTextColumn Header="Fair Value" 
                    Binding="{Binding TheoreticalPrice, StringFormat=N2}"/>
```

### 3. Advanced Filters

```csharp
// Add filter controls
<StackPanel Orientation="Horizontal">
    <CheckBox IsChecked="{Binding ShowOnlyITM}" Content="ITM Only"/>
    <CheckBox IsChecked="{Binding ShowOnlyLiquid}" Content="Liquid Only"/>
    <TextBox Text="{Binding StrikeRange}" PlaceholderText="±Strike Range"/>
    <Button Command="{Binding ApplyFilterCommand}" Content="Apply"/>
    <Button Command="{Binding ResetFiltersCommand}" Content="Reset"/>
</StackPanel>
```

### 4. Multi-Leg Strategies

```csharp
// Portfolio panel
<Border>
    <Grid>
        <DataGrid ItemsSource="{Binding SelectedLegs}"/>
        <StackPanel>
            <TextBlock Text="{Binding PortfolioGreeks.Delta, StringFormat='Delta: {0:N2}'}"/>
            <TextBlock Text="{Binding PortfolioGreeks.Theta, StringFormat='Theta: {0:N2}'}"/>
            <Button Command="{Binding CalculateGreeksCommand}" Content="Calculate"/>
        </StackPanel>
    </Grid>
</Border>
```

## ?? Performance Tips

1. **Lazy Loading**: Only load options for selected expiration
2. **Virtualization**: Enable in DataGrid for large chains
3. **Filtering**: Filter on backend before UI updates
4. **Caching**: Use StockDataCacheService for repeated queries
5. **Throttling**: Debounce filter changes to reduce re-queries

## ?? Integration with Other Views

### SpreadsExplorer Integration

```csharp
// In OptionsViewModel, add navigation
public ICommand OpenInSpreadsExplorerCommand { get; }

private void OpenInSpreadsExplorer()
{
    // Create spread configuration from selected legs
    var spread = new SpreadConfiguration
    {
        UnderlyingSymbol = SelectedSymbol,
        Legs = SelectedLegs.Select(opt => new OptionLeg
        {
            Option = opt,
            Quantity = 1,
            Action = "BUY" // or let user specify
        }).ToList()
    };
    
    // Navigate to SpreadsExplorer with configuration
    var spreadsView = new SpreadsExplorer();
    var spreadsViewModel = spreadsView.DataContext as SpreadsExplorerViewModel;
    spreadsViewModel?.LoadSpread(spread);
    
    // Show in main window or navigate
    NavigationService.Navigate(spreadsView);
}
```

### Paper Trading Integration

```csharp
// In OptionsViewModel
public ICommand ExecutePaperTradeCommand { get; }

private async Task ExecutePaperTrade()
{
    if (SelectedLegs.Count == 0)
    {
        StatusMessage = "No legs selected";
        return;
    }
    
    // Create paper trade order
    foreach (var leg in SelectedLegs)
    {
        var order = new PaperTradeOrder
        {
            Symbol = leg.UnderlyingSymbol,
            OptionType = leg.OptionType,
            Strike = leg.StrikePrice,
            Expiration = leg.ExpirationDate,
            Quantity = 1, // TODO: Let user specify
            Price = leg.MidPrice,
            OrderType = "MARKET" // or LIMIT
        };
        
        await _paperTradingService.PlaceOrderAsync(order);
    }
    
    StatusMessage = $"Placed {SelectedLegs.Count} paper trades";
}
```

## ?? UI Enhancements Already Implemented

The XAML already includes:
- ? Moneyness-based row coloring (ITM/ATM/OTM)
- ? Greek symbol headers (?, ?, ?, ?)
- ? Intrinsic/Extrinsic value display
- ? Compact column widths
- ? Color-coded Greeks (Yellow IV, Cyan Greeks)
- ? Professional dark theme

## ?? Additional Resources

- See `OPTIONS_VIEWMODEL_IMPLEMENTATION.md` for detailed ViewModel documentation
- See `OPTIONS_UI_ENHANCEMENTS.md` for XAML styling details
- Check `OptionsChainFilter.cs` for filter implementation examples
- Review `IVSurfaceData.cs` for IV surface analysis capabilities

---

**Status**: Ready for Integration  
**Estimated Time**: 30-60 minutes  
**Complexity**: Low (if services already registered)  
**Risk**: Low (backward compatible)

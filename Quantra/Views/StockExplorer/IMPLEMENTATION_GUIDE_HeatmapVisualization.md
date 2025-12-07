# Implementation Guide: Feature Importance Heatmap Visualization

## Overview
This guide provides step-by-step instructions to add an interactive feature importance heatmap visualization to the StockExplorerV2 control for TFT (Temporal Fusion Transformer) predictions.

## Part 1: XAML Changes

### Step 1: Replace the Grid.Row="4" Section

**Location:** `Quantra\Views\StockExplorer\StockExplorerV2Control.xaml`

**Find this code (around line 254):**
```xaml
<StackPanel Grid.Row="4" Orientation="Horizontal" Margin="0,5,0,5">
    <TextBlock Text="TFT Feature Attention"
               FontSize="12"
               FontWeight="Bold"
               Foreground="#888888"/>
    <TextBlock Text=" (darker = higher influence)"
               FontSize="10"
               FontStyle="Italic"
               Foreground="#666666"
               VerticalAlignment="Center"
               Margin="10,0,0,0"/>
</StackPanel>

<!-- Feature Attention Heatmap Grid -->
<DataGrid Grid.Row="5"
```

**Replace it with this code:**
```xaml
<Grid Grid.Row="4">
    <Grid.RowDefinitions>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
        <RowDefinition Height="Auto"/>
    </Grid.RowDefinitions>

    <!-- Heatmap Section Header -->
    <StackPanel Grid.Row="0" Orientation="Horizontal" Margin="0,10,0,5">
        <TextBlock Text="?? TFT Feature Importance Heatmap"
                   FontSize="14"
                   FontWeight="Bold"
                   Foreground="#1E90FF"/>
        <TextBlock Text=" (color intensity = importance level)"
                   FontSize="10"
                   FontStyle="Italic"
                   Foreground="#888888"
                   VerticalAlignment="Center"
                   Margin="10,0,0,0"/>
    </StackPanel>

    <!-- Visual Heatmap Canvas -->
    <Border Grid.Row="1"
            Background="#1A1A2E"
            BorderBrush="#3E3E56"
            BorderThickness="1"
            CornerRadius="3"
            Padding="10"
            Margin="0,0,0,10">
        <Grid>
            <Grid.RowDefinitions>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="Auto"/>
            </Grid.RowDefinitions>

            <!-- Heatmap Canvas -->
            <Canvas x:Name="FeatureHeatmapCanvas"
                    Grid.Row="0"
                    Height="100"
                    Background="#23233A"
                    Margin="0,0,0,10"
                    ClipToBounds="True"/>

            <!-- Legend -->
            <StackPanel Grid.Row="1" Orientation="Horizontal" HorizontalAlignment="Center">
                <TextBlock Text="Low" 
                           FontSize="9" 
                           Foreground="#666666" 
                           VerticalAlignment="Center"
                           Margin="0,0,8,0"/>
                <Rectangle Width="80" 
                           Height="15"
                           RadiusX="2"
                           RadiusY="2">
                    <Rectangle.Fill>
                        <LinearGradientBrush StartPoint="0,0" EndPoint="1,0">
                            <GradientStop Color="#FFFFCC" Offset="0"/>
                            <GradientStop Color="#FFD700" Offset="0.25"/>
                            <GradientStop Color="#FFA500" Offset="0.5"/>
                            <GradientStop Color="#FF6347" Offset="0.75"/>
                            <GradientStop Color="#FF4500" Offset="1"/>
                        </LinearGradientBrush>
                    </Rectangle.Fill>
                </Rectangle>
                <TextBlock Text="High" 
                           FontSize="9" 
                           Foreground="#666666" 
                           VerticalAlignment="Center"
                           Margin="8,0,0,0"/>
            </StackPanel>
        </Grid>
    </Border>

    <!-- Detail Grid Header -->
    <StackPanel Grid.Row="2" Orientation="Horizontal" Margin="0,5,0,5">
        <TextBlock Text="Feature Attention Details"
                   FontSize="12"
                   FontWeight="Bold"
                   Foreground="#888888"/>
        <TextBlock Text=" (hover over bars for detailed information)"
                   FontSize="10"
                   FontStyle="Italic"
                   Foreground="#666666"
                   VerticalAlignment="Center"
                   Margin="10,0,0,0"/>
    </StackPanel>
</Grid>

<!-- Feature Attention Data Grid -->
<DataGrid Grid.Row="5"
```

## Part 2: Code-Behind Changes

### Step 2: Add Heatmap Drawing Method

**Location:** `Quantra\Views\StockExplorer\StockExplorerV2Control.xaml.cs`

**Add these methods to the class (after the `UpdateFeatureAttention` method):**

```csharp
private void DrawFeatureImportanceHeatmap(Dictionary<string, double> attentionWeights)
{
    if (attentionWeights == null || attentionWeights.Count == 0 || FeatureHeatmapCanvas == null)
        return;

    try
    {
        // Clear previous heatmap
        FeatureHeatmapCanvas.Children.Clear();

        // Get canvas dimensions
        double canvasWidth = FeatureHeatmapCanvas.ActualWidth;
        double canvasHeight = FeatureHeatmapCanvas.ActualHeight;

        if (canvasWidth <= 0 || canvasHeight <= 0)
        {
            // Canvas not yet measured, schedule redraw after layout
            FeatureHeatmapCanvas.Loaded += (s, e) => DrawFeatureImportanceHeatmap(attentionWeights);
            return;
        }

        // Sort features by importance (descending)
        var sortedFeatures = attentionWeights
            .OrderByDescending(kv => kv.Value)
            .Take(10) // Show top 10 features
            .ToList();

        if (sortedFeatures.Count == 0)
            return;

        // Calculate cell dimensions
        double cellWidth = canvasWidth / sortedFeatures.Count;
        double cellHeight = canvasHeight - 20; // Leave room for labels
        
        // Find max weight for normalization
        double maxWeight = sortedFeatures.Max(kv => kv.Value);
        if (maxWeight == 0) maxWeight = 1.0;

        // Draw heatmap cells
        for (int i = 0; i < sortedFeatures.Count; i++)
        {
            var feature = sortedFeatures[i];
            double normalizedWeight = feature.Value / maxWeight;

            // Create colored rectangle
            var rect = new System.Windows.Shapes.Rectangle
            {
                Width = cellWidth - 4,
                Height = cellHeight,
                Fill = GetHeatmapBrush(normalizedWeight),
                Stroke = new SolidColorBrush(Color.FromArgb(100, 255, 255, 255)),
                StrokeThickness = 1,
                RadiusX = 3,
                RadiusY = 3
            };

            // Position rectangle
            Canvas.SetLeft(rect, i * cellWidth + 2);
            Canvas.SetTop(rect, 0);

            // Add tooltip
            rect.ToolTip = CreateFeatureTooltip(feature.Key, feature.Value, normalizedWeight);

            // Add hover effects
            rect.MouseEnter += (s, e) =>
            {
                rect.Opacity = 0.8;
                rect.StrokeThickness = 2;
            };
            rect.MouseLeave += (s, e) =>
            {
                rect.Opacity = 1.0;
                rect.StrokeThickness = 1;
            };

            FeatureHeatmapCanvas.Children.Add(rect);

            // Add feature label
            var label = new TextBlock
            {
                Text = FormatFeatureName(feature.Key),
                FontSize = 9,
                Foreground = Brushes.White,
                Width = cellWidth - 4,
                TextAlignment = TextAlignment.Center,
                TextTrimming = TextTrimming.CharacterEllipsis
            };

            Canvas.SetLeft(label, i * cellWidth + 2);
            Canvas.SetTop(label, cellHeight + 2);

            FeatureHeatmapCanvas.Children.Add(label);

            // Add weight value overlay
            var valueText = new TextBlock
            {
                Text = feature.Value.ToString("F3"),
                FontSize = 10,
                FontWeight = FontWeights.Bold,
                Foreground = normalizedWeight > 0.5 ? Brushes.White : Brushes.Black,
                Width = cellWidth - 4,
                TextAlignment = TextAlignment.Center
            };

            Canvas.SetLeft(valueText, i * cellWidth + 2);
            Canvas.SetTop(valueText, cellHeight / 2 - 10);

            FeatureHeatmapCanvas.Children.Add(valueText);
        }
    }
    catch (Exception ex)
    {
        _loggingService?.LogErrorWithContext(ex, "Error drawing feature heatmap");
    }
}

private Brush GetHeatmapBrush(double normalizedWeight)
{
    // Create gradient from light yellow (low) to dark orange/red (high)
    if (normalizedWeight < 0.2)
    {
        // Very low: Light yellow
        return new SolidColorBrush(Color.FromRgb(255, 255, 204));
    }
    else if (normalizedWeight < 0.4)
    {
        // Low: Yellow-Gold
        return new SolidColorBrush(Color.FromRgb(255, 215, 0));
    }
    else if (normalizedWeight < 0.6)
    {
        // Medium: Orange
        return new SolidColorBrush(Color.FromRgb(255, 165, 0));
    }
    else if (normalizedWeight < 0.8)
    {
        // High: Tomato
        return new SolidColorBrush(Color.FromRgb(255, 99, 71));
    }
    else
    {
        // Very high: Dark Orange/Red
        return new SolidColorBrush(Color.FromRgb(255, 69, 0));
    }
}

private string FormatFeatureName(string featureName)
{
    if (string.IsNullOrEmpty(featureName))
        return "Unknown";

    // Capitalize first letter
    string formatted = char.ToUpperInvariant(featureName[0]) + 
        (featureName.Length > 1 ? featureName.Substring(1) : "");

    // Shorten if too long
    if (formatted.Length > 10)
        return formatted.Substring(0, 8) + "..";

    return formatted;
}

private ToolTip CreateFeatureTooltip(string featureName, double weight, double normalizedWeight)
{
    var tooltip = new ToolTip();
    var panel = new StackPanel { MaxWidth = 250 };

    // Feature name header
    panel.Children.Add(new TextBlock
    {
        Text = GetFullFeatureName(featureName),
        FontWeight = FontWeights.Bold,
        FontSize = 12,
        TextWrapping = TextWrapping.Wrap,
        Margin = new Thickness(0, 0, 0, 5)
    });

    // Weight information
    panel.Children.Add(new TextBlock
    {
        Text = $"Attention Weight: {weight:F4}",
        FontSize = 11,
        TextWrapping = TextWrapping.Wrap
    });

    // Normalized weight
    panel.Children.Add(new TextBlock
    {
        Text = $"Relative Importance: {normalizedWeight:P0}",
        FontSize = 11,
        TextWrapping = TextWrapping.Wrap
    });

    // Feature description
    panel.Children.Add(new TextBlock
    {
        Text = GetFeatureDescription(featureName),
        FontSize = 10,
        Foreground = new SolidColorBrush(Color.FromRgb(200, 200, 200)),
        TextWrapping = TextWrapping.Wrap,
        Margin = new Thickness(0, 5, 0, 0)
    });

    tooltip.Content = panel;
    return tooltip;
}

private string GetFullFeatureName(string featureName)
{
    var mapping = new Dictionary<string, string>
    {
        {"rsi", "Relative Strength Index"},
        {"macd", "Moving Average Convergence Divergence"},
        {"adx", "Average Directional Index"},
        {"cci", "Commodity Channel Index"},
        {"vwap", "Volume Weighted Average Price"},
        {"ema", "Exponential Moving Average"},
        {"sma", "Simple Moving Average"},
        {"bbands", "Bollinger Bands"},
        {"obv", "On-Balance Volume"},
        {"vol", "Volume"},
        {"volat", "Volatility"},
        {"atr", "Average True Range"},
        {"mom", "Momentum"},
        {"roc", "Rate of Change"},
        {"stoch", "Stochastic Oscillator"}
    };

    return mapping.ContainsKey(featureName.ToLower()) 
        ? mapping[featureName.ToLower()] 
        : featureName;
}

private string GetFeatureDescription(string featureName)
{
    var descriptions = new Dictionary<string, string>
    {
        {"rsi", "Measures momentum by comparing recent gains to losses. Values above 70 indicate overbought, below 30 oversold."},
        {"macd", "Trend-following momentum indicator showing the relationship between two moving averages."},
        {"adx", "Measures trend strength. Higher values indicate stronger trends regardless of direction."},
        {"vwap", "Average price weighted by volume, important support/resistance level."},
        {"ema", "Moving average giving more weight to recent prices."},
        {"bbands", "Shows volatility and relative price levels over a period."},
        {"obv", "Cumulative indicator relating volume to price change."},
        {"atr", "Measures market volatility by analyzing the range of price movement."},
        {"stoch", "Compares closing price to price range over a given period."}
    };

    return descriptions.ContainsKey(featureName.ToLower())
        ? descriptions[featureName.ToLower()]
        : "Technical indicator used in market analysis.";
}
```

### Step 3: Update the UpdateFeatureAttention Method

**Find the existing `UpdateFeatureAttention` method and add the heatmap call at the end:**

```csharp
private void UpdateFeatureAttention(Dictionary<string, double> attentionWeights)
{
    FeatureAttentionData.Clear();

    if (attentionWeights == null || attentionWeights.Count == 0)
        return;

    // Calculate total weight for percentage
    double totalWeight = attentionWeights.Values.Sum();

    // Sort by weight descending and take top features
    var sortedFeatures = attentionWeights
        .OrderByDescending(kv => kv.Value)
        .Take(15) // Top 15 features
        .ToList();

    foreach (var feature in sortedFeatures)
    {
        FeatureAttentionData.Add(new FeatureAttentionItem
        {
            FeatureName = feature.Key,
            AttentionWeight = feature.Value,
            ImportancePercent = totalWeight > 0 ? (feature.Value / totalWeight) * 100 : 0
        });
    }

    // **NEW: Draw the visual heatmap**
    DrawFeatureImportanceHeatmap(attentionWeights);
}
```

## Part 3: Add Required Using Statements

**Add these using statements at the top of the C# file if not already present:**

```csharp
using System.Windows.Shapes;
using System.Windows.Media;
using System.Windows.Controls;
```

## Testing

After implementing these changes:

1. **Build the project** to check for compilation errors
2. **Run the application**
3. **Load a stock configuration** in StockExplorerV2
4. **Select a stock** from the grid
5. **Observe the heatmap** visualization appear above the feature attention grid
6. **Hover over bars** to see detailed tooltips

## Expected Result

You should see:
- A visual heatmap with **color-coded bars** representing feature importance
- Colors ranging from **light yellow (low)** to **dark orange/red (high)**
- **Feature names** displayed below each bar
- **Weight values** overlaid on the bars
- **Interactive tooltips** with detailed information on hover
- A **color gradient legend** showing the intensity scale

## Benefits

This enhancement provides:
- **Visual feedback** on feature importance at a glance
- **Interactive experience** with hover effects and detailed tooltips
- **Professional appearance** matching the application's dark theme
- **Complementary view** alongside the detailed data grid

## Troubleshooting

If the heatmap doesn't appear:
1. Check that `FeatureHeatmapCanvas` is properly named in XAML
2. Verify the `DrawFeatureImportanceHeatmap` method is being called
3. Ensure attention weights are being passed from TFT predictions
4. Check the browser console for any JavaScript/rendering errors
5. Verify Canvas dimensions are calculated correctly (not zero)

## Support

For additional questions or issues, refer to:
- The original documentation in `FeatureImportanceHeatmapEnhancement.txt`
- LiveCharts documentation for WPF controls
- WPF Canvas and drawing documentation

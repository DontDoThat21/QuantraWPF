# XAML Snippet to Insert into PredictionAnalysis.xaml

## Location
Insert this XAML after the "Top Predictions Grid" section (Grid.Row="3") and before the "Temporal Attention and Feature Importance" section.

## How to Insert
1. Open `Quantra\Views\PredictionAnalysis\PredictionAnalysis.xaml`
2. Find the section with Grid.Row="3" (Top Predictions Grid)
3. After that section closes, insert the following XAML before the next major section

## XAML Code to Insert

```xml
        <!-- Multi-Horizon Candlestick Chart with TFT Predictions -->
        <Border Grid.Row="4" Background="#2D2D42" BorderBrush="#3E3E56" BorderThickness="1" 
                CornerRadius="5" Margin="10,10,10,5" Padding="15"
                Visibility="{Binding IsChartVisible, Converter={StaticResource BooleanToVisibilityConverter}}">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="*"/>
                </Grid.RowDefinitions>

                <!-- Chart Title -->
                <StackPanel Grid.Row="0" Orientation="Horizontal" Margin="0,0,0,10">
                    <TextBlock Text="Multi-Horizon Price Forecast" FontSize="16" FontWeight="Bold" Foreground="#1E90FF"/>
                    <TextBlock x:Name="ChartSymbolText" Text="" FontSize="16" FontWeight="Bold" 
                               Foreground="Cyan" Margin="10,0,0,0"/>
                    <TextBlock Text=" - OHLCV with Predictions" FontSize="14" Foreground="#888888" 
                               Margin="5,0,0,0" VerticalAlignment="Center"/>
                </StackPanel>

                <!-- Candlestick Chart with TFT Predictions -->
                <lvc:CartesianChart Grid.Row="1" x:Name="MultiHorizonCandlestickChart"
                                    Series="{Binding CandlestickSeriesCollection}"
                                    LegendLocation="Top"
                                    Hoverable="True"
                                    DataTooltip="{x:Null}">
                    <lvc:CartesianChart.AxisX>
                        <lvc:Axis Title="Time (Days)" 
                                  Labels="{Binding ChartDateLabels}"
                                  Foreground="White"
                                  FontSize="11">
                            <lvc:Axis.Separator>
                                <lvc:Separator StrokeThickness="1" Stroke="#2A2A3A"/>
                            </lvc:Axis.Separator>
                        </lvc:Axis>
                    </lvc:CartesianChart.AxisX>
                    <lvc:CartesianChart.AxisY>
                        <lvc:Axis Title="Price ($)" 
                                  LabelFormatter="{Binding PriceFormatter}"
                                  Foreground="White"
                                  FontSize="11">
                            <lvc:Axis.Separator>
                                <lvc:Separator StrokeThickness="1" Stroke="#3E3E56"/>
                            </lvc:Axis.Separator>
                        </lvc:Axis>
                    </lvc:CartesianChart.AxisY>
                </lvc:CartesianChart>
            </Grid>
        </Border>

        <!-- Volume Chart -->
        <Border Grid.Row="5" Background="#2D2D42" BorderBrush="#3E3E56" BorderThickness="1" 
                CornerRadius="5" Margin="10,5,10,5" Padding="15"
                Visibility="{Binding IsChartVisible, Converter={StaticResource BooleanToVisibilityConverter}}">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="*"/>
                </Grid.RowDefinitions>

                <TextBlock Grid.Row="0" Text="Volume" FontSize="12" FontWeight="Bold" 
                           Foreground="#888888" Margin="0,0,0,5"/>

                <lvc:CartesianChart Grid.Row="1" x:Name="VolumeChart"
                                    Series="{Binding VolumeSeriesCollection}"
                                    LegendLocation="None"
                                    Hoverable="True"
                                    DataTooltip="{x:Null}">
                    <lvc:CartesianChart.AxisX>
                        <lvc:Axis Labels="{Binding ChartDateLabels}"
                                  Foreground="White"
                                  FontSize="10"
                                  ShowLabels="False">
                            <lvc:Axis.Separator>
                                <lvc:Separator StrokeThickness="0"/>
                            </lvc:Axis.Separator>
                        </lvc:Axis>
                    </lvc:CartesianChart.AxisX>
                    <lvc:CartesianChart.AxisY>
                        <lvc:Axis LabelFormatter="{Binding VolumeFormatter}"
                                  Foreground="White"
                                  FontSize="10">
                            <lvc:Axis.Separator>
                                <lvc:Separator StrokeThickness="1" Stroke="#3E3E56"/>
                            </lvc:Axis.Separator>
                        </lvc:Axis>
                    </lvc:CartesianChart.AxisY>
                </lvc:CartesianChart>
            </Grid>
        </Border>
```

## Notes

1. **Grid.Row Numbers**: This assumes Grid.Row="4" and Grid.Row="5" for the candlestick and volume charts respectively, based on the updated row definitions.

2. **Bindings**: The following bindings must be available:
   - `IsChartVisible` - Boolean property controlling visibility
   - `CandlestickSeriesCollection` - SeriesCollection for main chart
   - `VolumeSeriesCollection` - SeriesCollection for volume chart
   - `ChartDateLabels` - List<string> for X-axis labels
   - `PriceFormatter` - Func<double, string> for Y-axis formatting
   - `VolumeFormatter` - Func<double, string> for volume Y-axis

3. **Dependencies**: Ensure the following are in place:
   - `xmlns:lvc="clr-namespace:LiveCharts.Wpf;assembly=LiveCharts.Wpf"` in the UserControl header
   - `BooleanToVisibilityConverter` in Resources
   - LiveCharts.Wpf NuGet package installed

4. **Existing Elements**: The XAML uses named controls:
   - `ChartSymbolText` - TextBlock for displaying symbol name
   - `MultiHorizonCandlestickChart` - Main candlestick chart control
   - `VolumeChart` - Volume chart control

5. **Styling**: Uses existing theme colors:
   - Background: `#2D2D42`
   - Border: `#3E3E56`
   - Title: `#1E90FF` (Dodger Blue)
   - Symbol: `Cyan`
   - Secondary text: `#888888`

## Verification

After inserting the XAML, verify:
- ? No XAML syntax errors
- ? Grid.Row numbers are sequential
- ? Bindings match property names in code-behind
- ? xmlns:lvc namespace is declared
- ? Resources (converters) are available
- ? The view builds without errors

## Alternative Insertion Method

If you prefer to insert programmatically or have conflicts with existing Grid.Row numbers, you can:

1. Check the actual row count in your XAML
2. Adjust Grid.Row="4" and Grid.Row="5" to match your layout
3. Update the RowDefinitions to include the new rows (Height="500" and Height="150")

## Example Context

The charts should appear in this sequence:
```
Grid.Row="0" - Title and Controls
Grid.Row="1" - Analysis Parameters
Grid.Row="2" - (Your existing content)
Grid.Row="3" - Top Predictions Grid
Grid.Row="4" - **Candlestick Chart (INSERT HERE)**
Grid.Row="5" - **Volume Chart (INSERT HERE)**
Grid.Row="6" - Temporal Attention and Feature Importance
Grid.Row="7" - Status Bar
```

## Complete Example with Context

```xml
<!-- Existing content above... -->

        <!-- Top Predictions Grid (Grid.Row="3") -->
        <Border Grid.Row="3" Background="#2D2D42">
            <!-- Top predictions content here -->
        </Border>

        <!-- INSERT THE CHART SECTIONS HERE -->
        
        <!-- Multi-Horizon Candlestick Chart with TFT Predictions -->
        <Border Grid.Row="4" Background="#2D2D42" BorderBrush="#3E3E56" BorderThickness="1" 
                CornerRadius="5" Margin="10,10,10,5" Padding="15"
                Visibility="{Binding IsChartVisible, Converter={StaticResource BooleanToVisibilityConverter}}">
            <!-- (Full candlestick chart XAML as shown above) -->
        </Border>

        <!-- Volume Chart -->
        <Border Grid.Row="5" Background="#2D2D42" BorderBrush="#3E3E56" BorderThickness="1" 
                CornerRadius="5" Margin="10,5,10,5" Padding="15"
                Visibility="{Binding IsChartVisible, Converter={StaticResource BooleanToVisibilityConverter}}">
            <!-- (Full volume chart XAML as shown above) -->
        </Border>
        
        <!-- END OF INSERTED SECTION -->

        <!-- Temporal Attention and Feature Importance (Grid.Row="6") -->
        <Border Grid.Row="6" Background="#2D2D42">
            <!-- Temporal attention content here -->
        </Border>

<!-- Existing content below... -->
```

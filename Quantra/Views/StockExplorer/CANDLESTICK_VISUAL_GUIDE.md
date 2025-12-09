# Candlestick Chart Visual Guide

## Visual Indicators Quick Reference

### ?? Time Labels
- **Date + Time** (`MM/dd HH:mm`): Shows when day changes
- **Time Only** (`HH:mm`): Shows for same-day candles
- **"AH" suffix**: Indicates after-hours trading period

### ??? Candlesticks

#### Regular Hours (9:30 AM - 4:00 PM ET)
- **Bright Green**: Price went up (close > open)
- **Bright Red**: Price went down (close < open)
- **Full Opacity**: Regular market hours

#### After-Hours (Before 9:30 AM or After 4:00 PM ET)
- **Dimmed Green**: Price went up during after-hours
- **Dimmed Red**: Price went down during after-hours
- **60% Opacity**: Easy to distinguish from regular hours

### ?? Volume Bars

#### Color Coding
- **Green Bars**: Buying pressure (price closed higher)
- **Red Bars**: Selling pressure (price closed lower)

#### Intensity Heatmap
- **Very Dark**: High volume (>150% of average)
- **Medium Dark**: Above average volume (100-150%)
- **Medium Light**: Below average volume (50-100%)
- **Very Light**: Low volume (<50% of average)

#### Average Volume Line
- **Orange Dashed Line**: Shows average volume across all candles
- **Purpose**: Quick reference to gauge if current volume is normal

### ?? Gap Markers
- **Yellow Diamonds**: Appear above candles where market gaps occur
- **Gap Definition**: Time between candles is more than 2x the normal interval
- **Common Causes**: 
  - Overnight gaps
  - Weekend gaps
  - Market halts/news events

## Tooltip Information

### Candlestick Tooltip
```
MM/dd HH:mm [GAP] [AFTER-HOURS]
Open:  $XXX.XX
High:  $XXX.XX
Low:   $XXX.XX
Close: $XXX.XX
Volume: X.XM
Change: ? $X.XX (+X.XX%)
```

### Volume Tooltip
```
HH:mm
Volume: X.XM
vs Avg: +XX% (High)
(Buying Pressure)
```

**Volume Intensity Labels:**
- **High**: >150% of average
- **Above Avg**: 100-150% of average
- **Below Avg**: 50-100% of average
- **Low**: <50% of average

## Trading Insights

### Reading Volume
1. **High volume with green bar**: Strong buying interest
2. **High volume with red bar**: Strong selling pressure
3. **Low volume candle**: Weak conviction, potential reversal
4. **Volume spike**: Significant event or news

### After-Hours Analysis
- **Dimmed candles**: Indicates less liquid market
- **Large after-hours moves**: Check for news events
- **Gap on next open**: Note difference between after-hours close and regular open

### Gap Interpretation
- **Gap Up (Yellow diamond, green candle)**: Bullish signal
- **Gap Down (Yellow diamond, red candle)**: Bearish signal
- **Gap Fill**: Price returns to pre-gap level
- **Gap Extension**: Price continues in gap direction

## Chart Controls

### Zoom Controls
- **Zoom In (??+)**: Show fewer candles with more detail
- **Zoom Out (??-)**: Show more candles with less detail
- **Reset Zoom (??)**: Return to default view

### Time Intervals
- **1 min**: Intraday scalping, shows ~390 candles per day
- **5 min**: Short-term day trading, shows ~78 candles per day
- **15 min**: Swing trading, shows ~26 candles per day
- **30 min**: Position analysis, shows ~13 candles per day
- **60 min**: Long-term trends, shows ~6.5 candles per day

### Display Options
- **Candles Selector**: Choose number of candles to display (50/100/200/All)
- **Auto-Refresh**: Toggle automatic data updates
- **Pause/Resume**: Temporarily stop chart updates

## Legend

### Main Chart
- **[Symbol] (Regular)**: Regular trading hours candles
- **[Symbol] (After-Hours)**: After-hours candles (if present)
- **Market Gaps**: Gap indicators (if present)

### Volume Chart
- **Buy Volume**: Green bars showing buying pressure
- **Sell Volume**: Red bars showing selling pressure
- **Avg Volume**: Orange dashed line showing average

## Tips for Analysis

### Volume Analysis
1. **Compare to average**: Look for bars significantly above/below orange line
2. **Volume at support/resistance**: High volume = stronger level
3. **Divergence**: Price makes new high but volume decreases = weak move
4. **Climax volume**: Extreme spike often signals reversal

### After-Hours Trading
1. **Lower liquidity**: Dimmed candles remind you of wider spreads
2. **News impact**: Often where big moves happen on earnings
3. **Gap potential**: Large after-hours moves = likely gap on open
4. **Risk**: Higher volatility and slippage possible

### Gap Trading
1. **Gap and go**: Price continues in gap direction = strong trend
2. **Gap fill**: Price returns to fill gap = mean reversion
3. **Partial fill**: Gap 50% filled then reverses = support/resistance
4. **Exhaustion gap**: Large gap near trend end = reversal signal

## Keyboard Shortcuts

- **F5**: Refresh chart data
- **Ctrl+R**: Toggle auto-refresh
- **Ctrl+P**: Pause/Resume updates
- **Ctrl++**: Zoom in
- **Ctrl+-**: Zoom out
- **Ctrl+0**: Reset zoom
- **Ctrl+I**: Change interval
- **ESC**: Close chart window

## Best Practices

1. **Check time labels**: Ensure you're analyzing the correct time period
2. **Note after-hours**: Don't confuse after-hours moves with regular trading
3. **Use gap markers**: Identify important breakout/breakdown points
4. **Compare volume**: Use average line and intensity to gauge strength
5. **Multi-timeframe**: Check multiple intervals for complete picture
6. **Consider context**: News, earnings, market conditions affect patterns

## Common Patterns

### Bullish Signals
- High volume green candle breaking resistance
- Gap up with high volume continuation
- Volume increasing on uptrend
- After-hours gap up on good news

### Bearish Signals
- High volume red candle breaking support
- Gap down with high volume continuation
- Volume increasing on downtrend
- After-hours gap down on bad news

### Reversal Signals
- Climax volume at extreme high/low
- Gap that immediately starts to fill
- Volume divergence (price up, volume down)
- After-hours move that reverses on open

## Troubleshooting

### "No after-hours candles shown"
- Data may only include regular hours
- Check time interval (1min/5min more likely to have after-hours data)

### "No gap markers visible"
- May not be gaps in this time period
- Try viewing multi-day data with overnight gaps

### "Volume colors all same"
- Check if viewing extremely low volatility period
- All candles same direction = all same volume color

### "Average volume line not visible"
- May be at bottom if viewing high volume spike
- Check volume chart scale

## Additional Resources

See also:
- `CANDLESTICK_FEATURES_GUIDE.md` - Complete feature documentation
- `CANDLESTICK_UX_ENHANCEMENTS.md` - User experience improvements
- `CANDLESTICK_TECHNICAL_INDICATORS.md` - Technical indicator overlays
- `CANDLESTICK_MODAL_IMPROVEMENTS.md` - Modal window enhancements

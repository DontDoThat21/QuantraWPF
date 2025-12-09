# Real-Time Features Visual Guide

## Quick Reference

### ?? Header Layout

```
???????????????????????????????????????????????????????????????????????????????
? AAPL [OPEN]                    ? Next refresh in 12s      [AUTO-REFRESH ON]?
? Last: $175.43 ? +1.25 (+0.72%)        Regular hours:       [? PAUSE]        ?
? Bid: $175.42 × 500  [???????????] 0.01%  Ask: $175.44 × 500  [? CONFIGURE]?
?                              9:30 AM - 4:00 PM ET         [?? REFRESH NOW] ?
?                                                            [? CLOSE]        ?
???????????????????????????????????????????????????????????????????????????????
```

## Market Status Indicators

### ?? OPEN (Green Badge)
```
AAPL [OPEN]
```
**When:** 9:30 AM - 4:00 PM ET, Monday-Friday  
**Meaning:** Regular market hours, full trading  
**Trading:** All orders accepted, best liquidity

### ?? PRE-MARKET (Yellow Badge)
```
AAPL [PRE-MARKET]
```
**When:** 4:00 AM - 9:30 AM ET, Monday-Friday  
**Meaning:** Pre-market trading session  
**Trading:** Limited liquidity, wider spreads

### ?? AFTER-HOURS (Orange Badge)
```
AAPL [AFTER-HOURS]
```
**When:** 4:00 PM - 8:00 PM ET, Monday-Friday  
**Meaning:** After-hours trading session  
**Trading:** Reduced volume, price discovery

### ?? CLOSED (Red Badge)
```
AAPL [CLOSED]
```
**When:** Weekends, holidays, outside extended hours  
**Meaning:** Market is closed  
**Trading:** No trading, price frozen

## Real-Time Price Ticker

### Price Up Movement
```
Last: $175.43 ? +1.25 (+0.72%)
      ?        ?    ?      ?
   Current   Tick  $ Chg  % Chg
   Price   Indicator Change Change
```
**Color:** Green (#20C040)  
**Symbol:** ? (up triangle)  
**Updates:** Every 5 seconds

### Price Down Movement
```
Last: $174.18 ? -1.25 (-0.71%)
      ?        ?    ?      ?
   Current   Tick  $ Chg  % Chg
   Price   Indicator Change Change
```
**Color:** Red (#C02020)  
**Symbol:** ? (down triangle)  
**Updates:** Every 5 seconds

### Tick Animation
The tick indicator (?/?) flashes briefly (0.5s) when price changes:
```
Normal: Last: $175.43   +1.25 (+0.72%)
Flash:  Last: $175.43 ? +1.25 (+0.72%)  ? Appears briefly
```

## Countdown Timer

### Active Countdown
```
? Next refresh in 12s
    ?              ?
   Icon        Seconds
```
**Updates:** Every second  
**Range:** From interval seconds down to 0  
**Color:** Orange (#FFA500)

### Countdown Behavior
```
15 ? 14 ? 13 ? ... ? 3 ? 2 ? 1 ? 0 ? [REFRESH] ? 15 ? ...
                                         ?
                                    Chart updates
```

### When Paused
```
? Paused
```
Countdown stops, no automatic refreshes

## Bid/Ask Spread Visualization

### Complete Display
```
Bid: $175.42 × 500  [???????????] 0.01%  Ask: $175.44 × 500
     ?       ?       ?            ?       ?       ?
  Bid Price Size   Visual Bar   Spread  Ask Price Size
```

### Visual Spread Bar
```
Full Width: 80px
????????????????????????????????????????
???????????????????????????????????????
?? Bid (Red) ??    Ask (Green)      ??
????????????????????????????????????????

Width Calculation:
Bid Width = (BidSize / TotalSize) * 80px
Ask Width = (AskSize / TotalSize) * 80px
```

### Spread Percentage
```
Spread% = ((Ask - Bid) / MidPrice) * 100

Example:
Ask = $175.44
Bid = $175.42
Mid = $175.43
Spread = (0.02 / 175.43) * 100 = 0.0114%
```

### Size Display
```
× 500
  ?
Quantity available at that price
```

## Spread Bar Color Coding

### Tight Spread (Good Liquidity)
```
[????????????????????????????????] 0.01%
 Balanced red/green = Good market
```

### Wide Spread (Poor Liquidity)
```
[????????????????????????????????] 0.25%
 Imbalanced = Poor liquidity
```

### Bid-Heavy Market
```
[????????????????????????] 0.05%
 More red = More sellers
```

### Ask-Heavy Market
```
[??????????????????????????] 0.05%
 More green = More buyers
```

## Market Hours Information

### Display Format
```
Regular hours: 9:30 AM - 4:00 PM ET
     ?               ?
   Market Open   Market Close
```

### Tooltip (Hover)
```
???????????????????????????????
? Market Status: OPEN         ?
?                             ?
? Current Time: 2:45 PM ET    ?
? Regular Hours:              ?
?   9:30 AM - 4:00 PM ET      ?
?                             ?
? Extended Hours:             ?
?   Pre:  4:00 AM - 9:30 AM   ?
?   After: 4:00 PM - 8:00 PM  ?
???????????????????????????????
```

## Control Buttons

### Auto-Refresh Toggle
```
[AUTO-REFRESH ON]  ? Active
[AUTO-REFRESH OFF] ? Inactive
```
Toggles automatic chart data refresh

### Pause/Resume Button
```
[? PAUSE]  ? Active
[? RESUME] ? Paused
```
Temporarily stops all updates

### Configure Button
```
[? CONFIGURE]
```
Opens refresh interval settings dialog

### Refresh Now Button
```
[?? REFRESH NOW]
```
Manual refresh, resets countdown

### Close Button
```
[? CLOSE]
```
Closes the chart window

## Status Bar (Bottom)

### Display Format
```
Showing 100 candles  | ? Price: $175.43  | API Calls Today: 12 | Refresh: 15s
     ?                      ?                      ?               ?
   Candle Count        Crosshair Price       API Usage        Interval
```

## Tooltip Examples

### Bid Price Tooltip
```
??????????????????????????????
? Bid Price: $175.42         ?
? Bid Size: 500 shares       ?
? Side: Sellers              ?
?                            ?
? Spread: $0.02 (0.01%)      ?
??????????????????????????????
```

### Ask Price Tooltip
```
??????????????????????????????
? Ask Price: $175.44         ?
? Ask Size: 500 shares       ?
? Side: Buyers               ?
?                            ?
? Spread: $0.02 (0.01%)      ?
??????????????????????????????
```

### Spread Bar Tooltip
```
??????????????????????????????
? Bid-Ask Spread             ?
?                            ?
? Bid: $175.42 (500)         ?
? Ask: $175.44 (500)         ?
? Spread: $0.02 (0.01%)      ?
?                            ?
? Liquidity: Good            ?
? Imbalance: Neutral         ?
??????????????????????????????
```

### Market Status Tooltip
```
??????????????????????????????
? Market Status: OPEN        ?
?                            ?
? Session: Regular Hours     ?
? Time: 2:45 PM ET           ?
?                            ?
? Closes in: 1h 15m          ?
??????????????????????????????
```

## Update Frequency

### Real-Time Features
```
Feature                Update Frequency
?????????????????????????????????????
Price Ticker           Every 5 seconds
Countdown Timer        Every 1 second
Market Status          Every 1 minute
Chart Data             Every 15 seconds (default)
Bid/Ask Spread         Every 5 seconds (with ticker)
```

### Synchronized Updates
```
Time (s)  Price  Countdown  Chart  Status
??????????????????????????????????????????
0         ?      ?          ?      ?       Start
1                ?
2                ?
3                ?
4                ?
5         ?      ?                          Ticker
6                ?
...
14               ?
15        ?      ?          ?      ?       Chart refresh
```

## Color Legend

### Market Status
- ?? **Green**: Trading active (OPEN)
- ?? **Yellow**: Pre-market trading
- ?? **Orange**: After-hours trading
- ?? **Red**: Market closed

### Price Movement
- ?? **Green**: Price increased
- ?? **Red**: Price decreased
- ? **White**: No change

### Bid/Ask
- ?? **Red**: Bid side (sellers)
- ?? **Green**: Ask side (buyers)
- ?? **Orange**: Spread indicator

### Countdown
- ?? **Orange**: Active countdown
- ? **Gray**: Paused/inactive

## Common Scenarios

### 1. Market Just Opened
```
AAPL [OPEN]
Last: $174.50 ? +0.25 (+0.14%)
Bid: $174.49 × 1200  [??????????] 0.01%  Ask: $174.51 × 800
```
**Meaning:** Fresh open, establishing price, good liquidity

### 2. High Volume Trading
```
AAPL [OPEN]
Last: $175.43 ? +1.25 (+0.72%)
Bid: $175.42 × 5000  [??????????] 0.01%  Ask: $175.44 × 3500
```
**Meaning:** High activity, tight spread, good liquidity

### 3. Low Volume After-Hours
```
AAPL [AFTER-HOURS]
Last: $174.80 ? -0.63 (-0.36%)
Bid: $174.70 × 50  [????????????] 0.14%  Ask: $174.95 × 100
```
**Meaning:** Wide spread, low liquidity, be cautious

### 4. Market Closed Weekend
```
AAPL [CLOSED]
Last: $175.43   +1.25 (+0.72%)
Bid: N/A          [            ]        Ask: N/A
```
**Meaning:** No trading, last price from Friday close

## Pro Tips

### Reading the Spread
- **< 0.05%**: Very tight, excellent liquidity
- **0.05% - 0.10%**: Normal for liquid stocks
- **0.10% - 0.25%**: Moderate liquidity
- **> 0.25%**: Wide spread, poor liquidity

### Using the Countdown
- Plan trades before countdown hits 0
- Prepare orders during final 5 seconds
- Use manual refresh for immediate updates

### Interpreting Market Status
- **OPEN**: Best time for active trading
- **PRE-MARKET**: News-driven moves, be careful
- **AFTER-HOURS**: Limited orders, wider spreads
- **CLOSED**: No trading, plan for next session

### Price Ticker Strategy
- Watch for tick direction changes
- Consecutive up/down ticks = momentum
- Tick animation = recent price movement
- Compare ticker to chart for confirmation

### Bid/Ask Analysis
- Balanced spread = healthy market
- Bid-heavy = selling pressure
- Ask-heavy = buying pressure
- Widening spread = increasing volatility

## Keyboard Shortcuts Quick Reference

```
Key             Action
?????????????????????????????????????
F5              Refresh chart now
Ctrl+R          Toggle auto-refresh
Ctrl+P          Pause/Resume
Ctrl+M          Toggle market status
Ctrl+B          Toggle bid/ask
Ctrl+T          Toggle ticker
Ctrl+C          Toggle countdown
ESC             Close window
```

## Troubleshooting

### Issue: Price ticker not updating
**Solution:** Check internet connection, verify API limits not exceeded

### Issue: Market status wrong
**Solution:** Check computer's timezone settings, ensure ET time is correct

### Issue: Countdown stuck
**Solution:** Click Pause/Resume to reset timers

### Issue: No bid/ask data
**Solution:** Normal for free API - uses estimated values from current price

### Issue: Wide spread displayed
**Solution:** Low volume stock or after-hours trading, use caution

## Best Practices

1. **Monitor countdown** - Plan actions before refresh
2. **Check market status** - Trade during regular hours for best execution
3. **Watch spread** - Tighter spread = better entry/exit prices
4. **Use ticker** - Quick price updates without full refresh
5. **Enable auto-refresh** - Stay current with market moves
6. **Pause when needed** - Freeze display to analyze patterns
7. **Check bid/ask balance** - Understand current market sentiment
8. **Note tick direction** - Spot momentum early
9. **Use tooltips** - Hover for detailed information
10. **Keyboard shortcuts** - Faster navigation and control

## Additional Resources

See also:
- `REAL_TIME_FEATURES_COMPLETE.md` - Technical implementation
- `CANDLESTICK_VISUAL_GUIDE.md` - Chart features guide
- `CANDLESTICK_UX_ENHANCEMENTS_COMPLETE.md` - UX improvements

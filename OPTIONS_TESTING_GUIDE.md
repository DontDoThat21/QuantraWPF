# Options Explorer - Quick Start Testing Guide

## ? Pre-Flight Check

Before testing, verify:
- [x] Solution builds without errors
- [x] All services registered in DI
- [x] ViewModel properly injected
- [x] Alpha Vantage API key configured

---

## ?? Quick Test Scenarios

### Scenario 1: Basic Symbol Loading (2 minutes)

**Steps**:
1. Launch application
2. Navigate to Options Explorer
3. Enter symbol: `AAPL`
4. Click "Load" button

**Expected Results**:
- ? Underlying price displays (~$185)
- ? Expiration dates populate in dropdown
- ? Status message shows "AAPL loaded - $XXX.XX"
- ? No errors in log

**Pass Criteria**: Symbol loads within 2 seconds

---

### Scenario 2: Options Chain Display (3 minutes)

**Steps**:
1. From Scenario 1
2. Select nearest expiration from dropdown
3. Wait for chain to load

**Expected Results**:
- ? Calls DataGrid populates (left side)
- ? Puts DataGrid populates (right side)
- ? Strikes centered around current price
- ? ITM options highlighted in green
- ? ATM options highlighted in yellow
- ? Status shows "Loaded X calls and Y puts"

**Pass Criteria**: Chain loads within 3 seconds, >10 strikes displayed

---

### Scenario 3: Filtering (2 minutes)

**Steps**:
1. From Scenario 2
2. Check "ITM Only" checkbox
3. Observe chain updates

**Expected Results**:
- ? Only ITM options remain
- ? OTM options hidden
- ? Chain updates instantly
- ? Count in status bar decreases

**Steps (continued)**:
4. Uncheck "ITM Only"
5. Check "Liquid Only"

**Expected Results**:
- ? Only high-volume options shown
- ? Low-volume options filtered out
- ? Status updates with new count

**Pass Criteria**: Filters apply instantly (<100ms)

---

### Scenario 4: Option Selection & Greeks (2 minutes)

**Steps**:
1. From Scenario 2
2. Click on any call option row
3. Observe details panel (right side)

**Expected Results**:
- ? Details panel populates
- ? Intrinsic value displayed
- ? Extrinsic value displayed
- ? IV displayed (yellow)
- ? Delta displayed (cyan)
- ? Gamma displayed (cyan)
- ? Theta displayed (cyan)
- ? Vega displayed (cyan)
- ? Volume and OI shown

**Pass Criteria**: Details populate within 200ms of selection

---

### Scenario 5: Multi-Leg Position (3 minutes)

**Steps**:
1. From Scenario 2
2. Click "Add" button on a call option (e.g., Strike 180)
3. Click "Add" button on a put option (e.g., Strike 170)
4. Observe portfolio panel (bottom right)

**Expected Results**:
- ? Portfolio DataGrid shows 2 legs
- ? Portfolio Greeks display:
  - Delta (combined)
  - Gamma (combined)
  - Theta (combined)
  - Vega (combined)
  - Rho (combined)
- ? Status shows portfolio summary

**Steps (continued)**:
5. Click "X" button to remove one leg

**Expected Results**:
- ? Leg removed from grid
- ? Portfolio Greeks recalculate
- ? Status updates

**Pass Criteria**: Greeks calculate within 300ms

---

### Scenario 6: Advanced Features (5 minutes)

#### 6a: Refresh Data
**Steps**:
1. From any scenario
2. Click "Refresh" button

**Expected**:
- ? All data reloads
- ? Prices update
- ? Chain refreshes
- ? No errors

#### 6b: Build IV Surface
**Steps**:
1. From Scenario 2
2. Click "Build IV Surface" button

**Expected**:
- ? Loading indicator appears
- ? Status shows "Building IV surface..."
- ? Success message after ~5 seconds
- ? No crashes

#### 6c: Reset Filters
**Steps**:
1. Apply ITM and Liquid filters
2. Click "Reset" button

**Expected**:
- ? All filters clear
- ? Full chain displays
- ? Status updates

---

## ?? Visual Verification Checklist

### DataGrid Appearance
- [ ] Calls labeled "CALLS" in green header
- [ ] Puts labeled "PUTS" in red header
- [ ] Strike column is bold and centered
- [ ] IV column is yellow and bold
- [ ] Greek columns use symbols: ?, ?, ?, ?
- [ ] Greek values are cyan
- [ ] Numbers right-aligned
- [ ] ITM rows have green background
- [ ] ATM rows have yellow background
- [ ] OTM rows have dark background

### Details Panel
- [ ] Last Price displayed
- [ ] Mid Price displayed
- [ ] Intrinsic value in light green
- [ ] Extrinsic value in orange
- [ ] IV in yellow
- [ ] Greeks in cyan
- [ ] Volume/OI in white
- [ ] Days to expiration shown

### Status Bar
- [ ] Shows current operation
- [ ] Updates during loading
- [ ] Shows success/error messages
- [ ] Progress bar appears during load

---

## ?? Edge Cases to Test

### No Options Available
**Symbol**: Any symbol without listed options
**Expected**: Graceful error message, no crash

### Weekend/After Hours
**Time**: Outside market hours
**Expected**: Last available data shown, note in status

### Invalid Symbol
**Symbol**: `INVALID123`
**Expected**: Error message, no crash

### Network Error
**Action**: Disconnect internet, try to load
**Expected**: Timeout message, graceful degradation

### Very Large Chain
**Symbol**: `SPY` (500+ strikes)
**Expected**: Handles smoothly, virtualization works

---

## ?? Bug Report Template

If you find a bug, document:

```markdown
**Bug**: [Short description]

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**:
[What should happen]

**Actual Behavior**:
[What actually happened]

**Environment**:
- OS: Windows 11
- .NET: 9.0
- Symbol: AAPL
- Expiration: [Date]

**Logs**:
[Paste relevant log entries]

**Screenshot**:
[Attach if applicable]
```

---

## ? Success Metrics

### Performance
- Symbol load: < 2 seconds ?
- Chain load: < 3 seconds ?
- Filter apply: < 100ms ?
- Greeks calc: < 200ms ?
- Portfolio Greeks: < 300ms ?

### Reliability
- No crashes during normal use ?
- Graceful error handling ?
- Informative error messages ?
- Recovers from API failures ?

### Usability
- Intuitive UI layout ?
- Clear visual feedback ?
- Responsive interactions ?
- Helpful status messages ?

---

## ?? Test Results Template

```markdown
## Test Session: [Date/Time]
**Tester**: [Name]
**Environment**: [OS, .NET version]

### Scenario 1: Basic Symbol Loading
- Status: ? PASS / ? FAIL
- Time: [Seconds]
- Notes: [Any observations]

### Scenario 2: Options Chain Display
- Status: ? PASS / ? FAIL
- Time: [Seconds]
- Notes: [Any observations]

### Scenario 3: Filtering
- Status: ? PASS / ? FAIL
- Notes: [Any observations]

### Scenario 4: Option Selection & Greeks
- Status: ? PASS / ? FAIL
- Notes: [Any observations]

### Scenario 5: Multi-Leg Position
- Status: ? PASS / ? FAIL
- Notes: [Any observations]

### Scenario 6: Advanced Features
- 6a Refresh: ? PASS / ? FAIL
- 6b IV Surface: ? PASS / ? FAIL
- 6c Reset: ? PASS / ? FAIL

### Overall Assessment
- All Critical: ? PASS / ? FAIL
- Performance: ? ACCEPTABLE / ? NEEDS WORK
- Usability: ? GOOD / ?? FAIR / ? POOR

### Blockers
[List any issues that prevent release]

### Recommendations
[Suggested improvements or fixes]
```

---

## ?? Ready to Ship Checklist

Before marking as production-ready:

- [ ] All scenarios pass
- [ ] No critical bugs
- [ ] Performance meets targets
- [ ] Error handling works
- [ ] Logging is adequate
- [ ] UI is polished
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Unit tests added
- [ ] Integration tested

---

## ?? Need Help?

**Check Documentation**:
- `OPTIONS_VIEWMODEL_IMPLEMENTATION.md` - ViewModel details
- `INTEGRATION_GUIDE.md` - Integration steps
- `OPTIONS_INTEGRATION_COMPLETE.md` - Status and features

**Check Logs**:
- Location: Application logs folder
- Look for: Errors, warnings, API failures

**Debug Tips**:
- Add breakpoints in ViewModel methods
- Check ServiceProvider for service registration
- Verify Alpha Vantage API key
- Test with highly liquid symbols first

---

**Happy Testing!** ??

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Framework**: .NET 9 / WPF

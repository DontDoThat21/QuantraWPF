# Scroll-Based Pagination Implementation

## Overview
This document describes the implementation of scroll-based pagination for the StockDataGrid in StockExplorer. The feature allows users to load previous or next pages by scrolling up or down, even when the grid doesn't physically scroll because all items fit on the screen.

## Implementation Details

### Mouse Wheel Event Handler
- **Event**: `PreviewMouseWheel` on `StockDataGrid`
- **Location**: `StockExplorer.UIEventHandlers.cs`
- **Method**: `StockDataGrid_PreviewMouseWheel`

### Behavior
1. **Scroll Up** (positive mouse wheel delta):
   - Loads the **previous page** if `CurrentPage > 1`
   - Triggers: `_viewModel.LoadPreviousPageAsync()`

2. **Scroll Down** (negative mouse wheel delta):
   - Loads the **next page** if `HasMorePages` is true
   - Triggers: `_viewModel.LoadMoreCachedStocksAsync()`

### Debouncing
- Uses a **300ms debounce timer** to prevent rapid pagination requests
- Ensures smooth user experience by grouping multiple scroll events
- Prevents concurrent load operations with `_scrollLoadPending` flag

### Filtering Behavior
- **Disabled during filtering**: Pagination is automatically disabled when filters are active
- Checks `IsFiltering` property before processing scroll events
- Prevents confusion when viewing filtered results

### Fallback to Traditional Scroll
- The existing `StockDataGrid_ScrollChanged` handler remains as a fallback
- Handles cases where users use scrollbar directly instead of mouse wheel
- Loads next page when scrolling past 90% of the scroll area

## Key Features

### Works Without Physical Scrolling
- The mouse wheel event captures scroll attempts even when the grid doesn't scroll
- Perfect for scenarios where all 20 items per page fit on screen
- User simply scrolls up/down to navigate between pages

### Thread-Safe
- Uses `_scrollLoadPending` flag to prevent concurrent operations
- Checks `_viewModel.IsLoading` before triggering new loads
- Debounce timer ensures clean operation sequencing

### User-Friendly
- Natural scroll interaction matches user expectations
- No need to click pagination buttons manually
- Smooth transitions with debouncing

## Code Changes

### Modified Files
1. **StockExplorer.UIEventHandlers.cs**
   - Added `StockDataGrid_PreviewMouseWheel` handler
   - Enhanced `InitializeScrollDebounceTimer` to support both directions
   - Updated `StockExplorer_Loaded` to subscribe to mouse wheel events
   - Modified `StockDataGrid_ScrollChanged` to use new debounce mechanism

### New Methods
- `StockDataGrid_PreviewMouseWheel`: Main handler for mouse wheel events
- `InitializeScrollDebounceTimer`: Configurable debounce timer initialization
- `ScrollDebounceTimer_Tick`: Generic tick handler placeholder

## Testing Recommendations

### Test Scenarios
1. **Standard Pagination**
   - Scroll down multiple times to advance pages
   - Scroll up to go back to previous pages
   - Verify page numbers update correctly

2. **Edge Cases**
   - Try scrolling up on page 1 (should do nothing)
   - Try scrolling down on last page (should do nothing)
   - Rapid scrolling in both directions

3. **Filtering**
   - Apply a filter and verify scroll pagination is disabled
   - Remove filter and verify pagination re-enables

4. **Performance**
   - Test with large datasets
   - Verify debouncing works (no duplicate requests)
   - Check loading indicators display correctly

## Future Enhancements
- Add visual feedback during page transitions
- Consider touch/gesture support for tablets
- Add keyboard shortcuts (Page Up/Down) for accessibility
- Implement smooth scrolling animation between pages

## Related Files
- `StockExplorer.xaml`: DataGrid definition with ScrollChanged event
- `StockExplorer.xaml.cs`: Main control logic and pagination properties
- `StockExplorerViewModel.cs`: Pagination state and data loading methods
- `SavedFilterService.cs`: Filter service (pagination disabled during filtering)

# Quick Reference: UI Styling Migration Guide

## Overview
This document provides quick reference information for developers migrating XAML components to use EnhancedStyles.xaml.

## Available Enhanced Styles

### Text Elements
| Element | Enhanced Style | Use For |
|---------|----------------|---------|
| `Label` | `EnhancedLabelStyle` | Standard form labels |
| `Label` | `EnhancedSmallLabelStyle` | Secondary/caption labels |
| `TextBlock` | `EnhancedTextBlockStyle` | Standard text content |
| `TextBlock` | `EnhancedHeaderTextBlockStyle` | Section headers |
| `TextBlock` | `EnhancedSmallTextBlockStyle` | Captions, hints, secondary text |

### Input Controls
| Element | Enhanced Style | Use For |
|---------|----------------|---------|
| `TextBox` | `EnhancedTextBoxStyle` | Text input fields |
| `ComboBox` | `EnhancedComboBoxStyle` | Dropdown selections |
| `CheckBox` | `EnhancedCheckBoxStyle` | Boolean input |
| `DatePicker` | `EnhancedDatePickerStyle` | Date selection |
| `ToggleButton` | `MaterialDesignSwitchToggleButton` | On/off switches |

### Data Display
| Element | Enhanced Style | Use For |
|---------|----------------|---------|
| `DataGridColumnHeader` | `EnhancedDataGridColumnHeaderStyle` | DataGrid headers |
| `TextBlock` (in DataGrid) | `EnhancedDataGridColumnStyle` | DataGrid cell content |

### Layout & Containers
| Element | Enhanced Style | Use For |
|---------|----------------|---------|
| `materialDesign:Card` | `EnhancedContentCardStyle` | Content cards |
| `Window` | `StandardizedWindowStyle` | Window chrome |
| `Rectangle` | `GridVisualizationCellStyle` | Grid visualization |

## Migration Patterns

### Before (Hardcoded):
```xml
<TextBlock Text="Sample Text" 
           FontSize="18" 
           Foreground="#CCCCCC" 
           FontFamily="Franklin Gothic Medium"
           Margin="0,2,0,4"/>
```

### After (Enhanced):
```xml
<TextBlock Text="Sample Text" 
           Style="{StaticResource EnhancedTextBlockStyle}"/>
```

### Before (Legacy):
```xml
<Button Content="Click Me" 
        Style="{StaticResource ButtonStyle1}"/>
```

### After (Enhanced):
```xml
<Button Content="Click Me" 
        Style="{StaticResource EnhancedButtonStyle}"/>
```

## Adding EnhancedStyles Reference

### Method 1: Resource Dictionary Merge (Recommended)
```xml
<UserControl.Resources>
    <ResourceDictionary>
        <ResourceDictionary.MergedDictionaries>
            <ResourceDictionary Source="/Quantra;component/Styles/EnhancedStyles.xaml" />
        </ResourceDictionary.MergedDictionaries>
    </ResourceDictionary>
</UserControl.Resources>
```

### Method 2: Global Reference (Already in App.xaml)
Enhanced styles are globally available through App.xaml, so explicit reference may not be needed.

## Common Migration Tasks

### 1. Remove Hardcoded Properties
**Find and replace:**
- `FontSize="XX"` → Remove (handled by style)
- `Foreground="#XXXXXX"` → Remove (handled by style)
- `FontFamily="XXX"` → Remove (handled by style)
- `Margin="XX"` → Remove or keep if component-specific

### 2. Replace Legacy Styles
**Find and replace:**
- `ButtonStyle1` → `EnhancedButtonStyle` (when available)
- `TextBoxStyle1` → `EnhancedTextBoxStyle`
- `ComboBoxStyle1` → `EnhancedComboBoxStyle`

### 3. Convert Local Resources
**Before:**
```xml
<UserControl.Resources>
    <Style x:Key="LocalTextStyle" TargetType="TextBlock">
        <Setter Property="Foreground" Value="White"/>
        <Setter Property="FontSize" Value="14"/>
    </Style>
</UserControl.Resources>
```

**After:**
```xml
<!-- Remove local style, use Enhanced style -->
<TextBlock Style="{StaticResource EnhancedTextBlockStyle}"/>
```

## Testing Checklist

After migrating a component:

- [ ] Visual appearance matches original design
- [ ] All interactive elements function correctly
- [ ] No console errors or binding issues
- [ ] Responsive behavior maintained
- [ ] Accessibility not degraded
- [ ] Performance not impacted

## Common Issues & Solutions

### Issue: Style not found
**Error:** `Cannot find resource named 'EnhancedXxxStyle'`
**Solution:** Add EnhancedStyles.xaml reference to component resources

### Issue: Visual differences after migration
**Cause:** Enhanced style properties differ from hardcoded values
**Solution:** 
1. Verify Enhanced style is appropriate for use case
2. Consider if differences are acceptable improvements
3. If needed, create component-specific style inheriting from Enhanced style

### Issue: Local resources conflict
**Cause:** Local styles override Enhanced styles
**Solution:** Remove conflicting local styles or rename to avoid conflicts

## Style Inheritance Pattern

For component-specific customizations:
```xml
<UserControl.Resources>
    <Style x:Key="CustomTextStyle" 
           TargetType="TextBlock" 
           BasedOn="{StaticResource EnhancedTextBlockStyle}">
        <Setter Property="FontWeight" Value="Bold"/>
    </Style>
</UserControl.Resources>
```

## Priority Components for Migration

### Phase 1 (Immediate):
- MainWindow.xaml
- AlertsControl.xaml
- ResizeControlWindow.xaml

### Phase 2 (High Priority):
- OrdersPage.xaml
- SettingsWindow.xaml
- TransactionsControl.xaml
- AddControlWindow.xaml

### Phase 3 (Medium Priority):
- StockExplorer.xaml
- ConfigurationControl.xaml
- TradingRulesControl.xaml

## Review Requirements

Before merging UI styling changes:
1. Screenshot comparison (before/after)
2. Functional testing of interactive elements
3. Code review for proper Enhanced style usage
4. Verification that hardcoded styles are removed
5. Confirmation that visual consistency is maintained
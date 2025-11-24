using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.Models;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Customize Layout Dialog
    /// </summary>
    public class CustomizeLayoutDialogViewModel : ViewModelBase
    {
        private LayoutConfig _currentLayout;
        private ChartPanelLayout _selectedPanel;
        private int _totalRows;
        private int _totalColumns;
        private bool _showGridLines;
        private string _gridLineColor;
        private string _layoutName;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="initialLayout">Initial layout configuration</param>
        public CustomizeLayoutDialogViewModel(LayoutConfig initialLayout = null)
        {
            _currentLayout = initialLayout ?? LayoutConfig.CreateDefault();
            InitializeFromLayout();
            InitializeCommands();
            RefreshPanels();
        }

        #region Properties

        /// <summary>
        /// Collection of chart panels
        /// </summary>
        public ObservableCollection<ChartPanelLayout> Panels { get; private set; } = new ObservableCollection<ChartPanelLayout>();

        /// <summary>
        /// Available panel types for adding new panels
        /// </summary>
        public ObservableCollection<string> AvailablePanelTypes { get; } = new ObservableCollection<string>
        {
            "RSI",
            "MACD",
            "StochRSI",
            "Williams %R",
            "CCI",
            "ADX",
            "ROC",
            "UO",
            "ATR",
            "Bull Power",
            "Bear Power",
            "Breadth Thrust"
        };

        /// <summary>
        /// Currently selected panel
        /// </summary>
        public ChartPanelLayout SelectedPanel
        {
            get => _selectedPanel;
            set => SetProperty(ref _selectedPanel, value);
        }

        /// <summary>
        /// Total number of grid rows
        /// </summary>
        public int TotalRows
        {
            get => _totalRows;
            set
            {
                if (SetProperty(ref _totalRows, value))
                {
                    _currentLayout.TotalRows = value;
                }
            }
        }

        /// <summary>
        /// Total number of grid columns
        /// </summary>
        public int TotalColumns
        {
            get => _totalColumns;
            set
            {
                if (SetProperty(ref _totalColumns, value))
                {
                    _currentLayout.TotalColumns = value;
                }
            }
        }

        /// <summary>
        /// Whether to show grid lines
        /// </summary>
        public bool ShowGridLines
        {
            get => _showGridLines;
            set
            {
                if (SetProperty(ref _showGridLines, value))
                {
                    _currentLayout.ShowGridLines = value;
                }
            }
        }

        /// <summary>
        /// Grid line color
        /// </summary>
        public string GridLineColor
        {
            get => _gridLineColor;
            set
            {
                if (SetProperty(ref _gridLineColor, value))
                {
                    _currentLayout.GridLineColor = value;
                }
            }
        }

        /// <summary>
        /// Layout name
        /// </summary>
        public string LayoutName
        {
            get => _layoutName;
            set
            {
                if (SetProperty(ref _layoutName, value))
                {
                    _currentLayout.LayoutName = value;
                }
            }
        }

        /// <summary>
        /// Gets the current layout configuration
        /// </summary>
        public LayoutConfig CurrentLayout => _currentLayout;

        #endregion

        #region Commands

        public ICommand AddPanelCommand { get; private set; }
        public ICommand RemovePanelCommand { get; private set; }
        public ICommand MovePanelUpCommand { get; private set; }
        public ICommand MovePanelDownCommand { get; private set; }
        public ICommand ResetToDefaultCommand { get; private set; }
        public ICommand ApplyCommand { get; private set; }
        public ICommand CancelCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when the dialog should be closed
        /// </summary>
        public event EventHandler<bool> CloseRequested;

        /// <summary>
        /// Event fired when layout changes should be applied
        /// </summary>
        public event EventHandler<LayoutConfig> LayoutApplied;

        #endregion

        #region Private Methods

        private void InitializeFromLayout()
        {
            _totalRows = _currentLayout.TotalRows;
            _totalColumns = _currentLayout.TotalColumns;
            _showGridLines = _currentLayout.ShowGridLines;
            _gridLineColor = _currentLayout.GridLineColor;
            _layoutName = _currentLayout.LayoutName;
        }

        private void InitializeCommands()
        {
            AddPanelCommand = new RelayCommand(ExecuteAddPanel, CanExecuteAddPanel);
            RemovePanelCommand = new RelayCommand(ExecuteRemovePanel, CanExecuteRemovePanel);
            MovePanelUpCommand = new RelayCommand(ExecuteMovePanelUp, CanExecuteMovePanelUp);
            MovePanelDownCommand = new RelayCommand(ExecuteMovePanelDown, CanExecuteMovePanelDown);
            ResetToDefaultCommand = new RelayCommand(ExecuteResetToDefault);
            ApplyCommand = new RelayCommand(ExecuteApply);
            CancelCommand = new RelayCommand(ExecuteCancel);
        }

        private void RefreshPanels()
        {
            Panels.Clear();
            foreach (var panel in _currentLayout.Panels.OrderBy(p => p.DisplayOrder))
            {
                Panels.Add(panel);
            }
        }

        #endregion

        #region Command Implementations

        private void ExecuteAddPanel(object parameter)
        {
            if (parameter is string panelType)
            {
                // Find next available position
                int nextRow = _currentLayout.Panels.Any() ? _currentLayout.Panels.Max(p => p.Row + p.RowSpan) : 0;

                var newPanel = new ChartPanelLayout
                {
                    PanelId = panelType,
                    DisplayName = panelType,
                    Row = Math.Min(nextRow, _totalRows - 1),
                    Column = 0,
                    RowSpan = 1,
                    ColumnSpan = 1,
                    HeightRatio = 1.0,
                    IsVisible = true,
                    DisplayOrder = _currentLayout.Panels.Count + 1
                };

                _currentLayout.Panels.Add(newPanel);
                RefreshPanels();
                SelectedPanel = newPanel;
            }
        }

        private bool CanExecuteAddPanel(object parameter)
        {
            return parameter is string panelType &&
                   !_currentLayout.Panels.Any(p => p.PanelId == panelType);
        }

        private void ExecuteRemovePanel(object parameter)
        {
            if (SelectedPanel != null && SelectedPanel.PanelId != "Price" && SelectedPanel.PanelId != "Volume")
            {
                _currentLayout.Panels.Remove(SelectedPanel);
                RefreshPanels();
                SelectedPanel = null;
            }
        }

        private bool CanExecuteRemovePanel(object parameter)
        {
            return SelectedPanel != null &&
                   SelectedPanel.PanelId != "Price" &&
                   SelectedPanel.PanelId != "Volume";
        }

        private void ExecuteMovePanelUp(object parameter)
        {
            if (SelectedPanel != null && SelectedPanel.DisplayOrder > 1)
            {
                var upperPanel = _currentLayout.Panels.FirstOrDefault(p => p.DisplayOrder == SelectedPanel.DisplayOrder - 1);
                if (upperPanel != null)
                {
                    int temp = upperPanel.DisplayOrder;
                    upperPanel.DisplayOrder = SelectedPanel.DisplayOrder;
                    SelectedPanel.DisplayOrder = temp;
                    RefreshPanels();
                }
            }
        }

        private bool CanExecuteMovePanelUp(object parameter)
        {
            return SelectedPanel != null && SelectedPanel.DisplayOrder > 1;
        }

        private void ExecuteMovePanelDown(object parameter)
        {
            if (SelectedPanel != null)
            {
                var maxOrder = _currentLayout.Panels.Max(p => p.DisplayOrder);
                if (SelectedPanel.DisplayOrder < maxOrder)
                {
                    var lowerPanel = _currentLayout.Panels.FirstOrDefault(p => p.DisplayOrder == SelectedPanel.DisplayOrder + 1);
                    if (lowerPanel != null)
                    {
                        int temp = lowerPanel.DisplayOrder;
                        lowerPanel.DisplayOrder = SelectedPanel.DisplayOrder;
                        SelectedPanel.DisplayOrder = temp;
                        RefreshPanels();
                    }
                }
            }
        }

        private bool CanExecuteMovePanelDown(object parameter)
        {
            if (SelectedPanel == null) return false;
            var maxOrder = _currentLayout.Panels.Any() ? _currentLayout.Panels.Max(p => p.DisplayOrder) : 0;
            return SelectedPanel.DisplayOrder < maxOrder;
        }

        private void ExecuteResetToDefault(object parameter)
        {
            _currentLayout = LayoutConfig.CreateDefault();
            InitializeFromLayout();
            RefreshPanels();
            OnPropertyChanged(nameof(TotalRows));
            OnPropertyChanged(nameof(TotalColumns));
            OnPropertyChanged(nameof(ShowGridLines));
            OnPropertyChanged(nameof(GridLineColor));
            OnPropertyChanged(nameof(LayoutName));
        }

        private void ExecuteApply(object parameter)
        {
            LayoutApplied?.Invoke(this, _currentLayout);
            CloseRequested?.Invoke(this, true);
        }

        private void ExecuteCancel(object parameter)
        {
            CloseRequested?.Invoke(this, false);
        }

        #endregion
    }
}
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Move Control Window
    /// </summary>
    public class MoveControlWindowViewModel : ViewModelBase
    {
        private string _sourceTabName;
        private string _selectedTabName;
        private int _resultRow;
        private int _resultColumn;
        private readonly int _rowSpan;
        private readonly int _columnSpan;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public MoveControlWindowViewModel(
            string sourceTabName,
            List<string> availableTabs,
            int currentRow,
            int currentColumn,
            int rowSpan,
            int columnSpan)
        {
            _sourceTabName = sourceTabName ?? throw new ArgumentNullException(nameof(sourceTabName));
            _selectedTabName = sourceTabName;
            _resultRow = currentRow;
            _resultColumn = currentColumn;
            _rowSpan = rowSpan;
            _columnSpan = columnSpan;

            AvailableTabs = new ObservableCollection<string>(availableTabs ?? new List<string>());
            
            InitializeCommands();
        }

        #region Properties

        /// <summary>
        /// Source tab name
        /// </summary>
        public string SourceTabName
        {
            get => _sourceTabName;
            set => SetProperty(ref _sourceTabName, value);
        }

        /// <summary>
        /// Selected target tab name
        /// </summary>
        public string SelectedTabName
        {
            get => _selectedTabName;
            set => SetProperty(ref _selectedTabName, value);
        }

        /// <summary>
        /// Result row position
        /// </summary>
        public int ResultRow
        {
            get => _resultRow;
            set => SetProperty(ref _resultRow, value);
        }

        /// <summary>
        /// Result column position
        /// </summary>
        public int ResultColumn
        {
            get => _resultColumn;
            set => SetProperty(ref _resultColumn, value);
        }

        /// <summary>
        /// Row span of the control
        /// </summary>
        public int RowSpan => _rowSpan;

        /// <summary>
        /// Column span of the control
        /// </summary>
        public int ColumnSpan => _columnSpan;

        /// <summary>
        /// Available tabs for moving to
        /// </summary>
        public ObservableCollection<string> AvailableTabs { get; }

        #endregion

        #region Commands

        public ICommand ApplyCommand { get; private set; }
        public ICommand CancelCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when move should be applied
        /// </summary>
        public event EventHandler<MoveControlEventArgs> MoveApplied;

        /// <summary>
        /// Event fired when dialog should be closed
        /// </summary>
        public event EventHandler<bool> CloseRequested;

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            ApplyCommand = new RelayCommand(ExecuteApply, CanExecuteApply);
            CancelCommand = new RelayCommand(ExecuteCancel);
        }

        #endregion

        #region Command Implementations

        private bool CanExecuteApply(object parameter)
        {
            return !string.IsNullOrWhiteSpace(SelectedTabName) && ResultRow >= 0 && ResultColumn >= 0;
        }

        private void ExecuteApply(object parameter)
        {
            MoveApplied?.Invoke(this, new MoveControlEventArgs
            {
                TargetTabName = SelectedTabName,
                Row = ResultRow,
                Column = ResultColumn
            });
            CloseRequested?.Invoke(this, true);
        }

        private void ExecuteCancel(object parameter)
        {
            CloseRequested?.Invoke(this, false);
        }

        #endregion
    }

    /// <summary>
    /// Event arguments for move control operation
    /// </summary>
    public class MoveControlEventArgs : EventArgs
    {
        public string TargetTabName { get; set; }
        public int Row { get; set; }
        public int Column { get; set; }
    }
}

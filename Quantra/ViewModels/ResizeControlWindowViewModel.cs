using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Resize Control Window
    /// </summary>
    public class ResizeControlWindowViewModel : ViewModelBase
    {
        private string _tabName;
        private int _resultRow;
        private int _resultColumn;
        private int _resultRowSpan;
        private int _resultColumnSpan;
        private readonly int _gridMaxRows;
        private readonly int _gridMaxCols;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public ResizeControlWindowViewModel(
            string tabName,
            int row,
            int column,
            int rowSpan,
            int columnSpan,
            int gridMaxRows,
            int gridMaxCols)
        {
            _tabName = tabName ?? throw new ArgumentNullException(nameof(tabName));
            _resultRow = row;
            _resultColumn = column;
            _resultRowSpan = rowSpan;
            _resultColumnSpan = columnSpan;
            _gridMaxRows = gridMaxRows;
            _gridMaxCols = gridMaxCols;

            InitializeCommands();
        }

        #region Properties

        /// <summary>
        /// Tab name containing the control
        /// </summary>
        public string TabName
        {
            get => _tabName;
            set => SetProperty(ref _tabName, value);
        }

        /// <summary>
        /// Result row position
        /// </summary>
        public int ResultRow
        {
            get => _resultRow;
            set => SetProperty(ref _resultRow, Math.Max(0, Math.Min(value, _gridMaxRows - 1)));
        }

        /// <summary>
        /// Result column position
        /// </summary>
        public int ResultColumn
        {
            get => _resultColumn;
            set => SetProperty(ref _resultColumn, Math.Max(0, Math.Min(value, _gridMaxCols - 1)));
        }

        /// <summary>
        /// Result row span
        /// </summary>
        public int ResultRowSpan
        {
            get => _resultRowSpan;
            set => SetProperty(ref _resultRowSpan, Math.Max(1, Math.Min(value, _gridMaxRows - _resultRow)));
        }

        /// <summary>
        /// Result column span
        /// </summary>
        public int ResultColumnSpan
        {
            get => _resultColumnSpan;
            set => SetProperty(ref _resultColumnSpan, Math.Max(1, Math.Min(value, _gridMaxCols - _resultColumn)));
        }

        /// <summary>
        /// Maximum rows in grid
        /// </summary>
        public int GridMaxRows => _gridMaxRows;

        /// <summary>
        /// Maximum columns in grid
        /// </summary>
        public int GridMaxCols => _gridMaxCols;

        #endregion

        #region Commands

        public ICommand ApplyCommand { get; private set; }
        public ICommand CancelCommand { get; private set; }
        public ICommand ResetCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when resize should be applied
        /// </summary>
        public event EventHandler<ResizeControlEventArgs> ResizeApplied;

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
            ResetCommand = new RelayCommand(ExecuteReset);
        }

        #endregion

        #region Command Implementations

        private bool CanExecuteApply(object parameter)
        {
            return ResultRowSpan > 0 && ResultColumnSpan > 0 &&
                   ResultRow + ResultRowSpan <= _gridMaxRows &&
                   ResultColumn + ResultColumnSpan <= _gridMaxCols;
        }

        private void ExecuteApply(object parameter)
        {
            ResizeApplied?.Invoke(this, new ResizeControlEventArgs
            {
                Row = ResultRow,
                Column = ResultColumn,
                RowSpan = ResultRowSpan,
                ColumnSpan = ResultColumnSpan
            });
            CloseRequested?.Invoke(this, true);
        }

        private void ExecuteCancel(object parameter)
        {
            CloseRequested?.Invoke(this, false);
        }

        private void ExecuteReset(object parameter)
        {
            // Reset to original values would require storing them
            ResultRowSpan = 1;
            ResultColumnSpan = 1;
        }

        #endregion
    }

    /// <summary>
    /// Event arguments for resize control operation
    /// </summary>
    public class ResizeControlEventArgs : EventArgs
    {
        public int Row { get; set; }
        public int Column { get; set; }
        public int RowSpan { get; set; }
        public int ColumnSpan { get; set; }
    }
}

using System;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Create Tab Window
    /// </summary>
    public class CreateTabWindowViewModel : ViewModelBase
    {
        private readonly UserSettingsService _userSettingsService;
        private string _tabName;
        private int _gridRows;
        private int _gridColumns;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public CreateTabWindowViewModel(UserSettingsService userSettingsService)
        {
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            
            // Load default grid settings
            try
            {
                var settings = _userSettingsService.GetUserSettings();
                GridRows = Math.Max(1, settings.DefaultGridRows);
                GridColumns = Math.Max(1, settings.DefaultGridColumns);
            }
            catch (Exception)
            {
                // Fall back to 4x4 grid
                GridRows = 4;
                GridColumns = 4;
            }

            InitializeCommands();
        }

        #region Properties

        /// <summary>
        /// Name of the new tab
        /// </summary>
        public string TabName
        {
            get => _tabName;
            set => SetProperty(ref _tabName, value);
        }

        /// <summary>
        /// Number of grid rows
        /// </summary>
        public int GridRows
        {
            get => _gridRows;
            set
            {
                if (SetProperty(ref _gridRows, Math.Max(1, Math.Min(value, 20))))
                {
                    OnPropertyChanged(nameof(GridPreviewInfo));
                }
            }
        }

        /// <summary>
        /// Number of grid columns
        /// </summary>
        public int GridColumns
        {
            get => _gridColumns;
            set
            {
                if (SetProperty(ref _gridColumns, Math.Max(1, Math.Min(value, 20))))
                {
                    OnPropertyChanged(nameof(GridPreviewInfo));
                }
            }
        }

        /// <summary>
        /// Grid preview info text
        /// </summary>
        public string GridPreviewInfo => $"{GridRows} x {GridColumns} Grid";

        #endregion

        #region Commands

        public ICommand CreateCommand { get; private set; }
        public ICommand CancelCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when tab should be created
        /// </summary>
        public event EventHandler<CreateTabEventArgs> TabCreated;

        /// <summary>
        /// Event fired when dialog should be closed
        /// </summary>
        public event EventHandler<bool> CloseRequested;

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            CreateCommand = new RelayCommand(ExecuteCreate, CanExecuteCreate);
            CancelCommand = new RelayCommand(ExecuteCancel);
        }

        #endregion

        #region Command Implementations

        private bool CanExecuteCreate(object parameter)
        {
            return !string.IsNullOrWhiteSpace(TabName) && GridRows > 0 && GridColumns > 0;
        }

        private void ExecuteCreate(object parameter)
        {
            TabCreated?.Invoke(this, new CreateTabEventArgs
            {
                TabName = TabName,
                GridRows = GridRows,
                GridColumns = GridColumns
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
    /// Event arguments for tab creation
    /// </summary>
    public class CreateTabEventArgs : EventArgs
    {
        public string TabName { get; set; }
        public int GridRows { get; set; }
        public int GridColumns { get; set; }
    }
}

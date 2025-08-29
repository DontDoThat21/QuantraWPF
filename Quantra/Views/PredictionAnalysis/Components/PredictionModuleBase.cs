using System;
using System.ComponentModel;
using System.Windows.Controls;
using Quantra.Data;
using Quantra.Services;
using Quantra.Services.Interfaces;

namespace Quantra.Views.PredictionAnalysis.Components
{
    /// <summary>
    /// Base class for all prediction analysis module controls
    /// </summary>
    public abstract class PredictionModuleBase : UserControl, INotifyPropertyChanged, IDisposable
    {
        protected readonly ITechnicalIndicatorService IndicatorService;
        protected readonly INotificationService NotificationService;
        private bool _disposed;

        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            protected set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
            }
        }

        private string _symbol;
        public string Symbol
        {
            get => _symbol;
            set
            {
                _symbol = value;
                OnPropertyChanged(nameof(Symbol));
                OnSymbolChanged();
            }
        }

        private string _timeframe;
        public string Timeframe
        {
            get => _timeframe;
            set
            {
                _timeframe = value;
                OnPropertyChanged(nameof(Timeframe));
                OnTimeframeChanged();
            }
        }

        public PredictionModuleBase(
            ITechnicalIndicatorService indicatorService = null,
            INotificationService notificationService = null)
        {
            IndicatorService = indicatorService ?? new TechnicalIndicatorService();
            NotificationService = notificationService ?? new NotificationService(DatabaseMonolith.GetUserSettings(), new AudioService(DatabaseMonolith.GetUserSettings()));
        }

        protected virtual void OnSymbolChanged()
        {
            // Override in derived classes to handle symbol changes
        }

        protected virtual void OnTimeframeChanged()
        {
            // Override in derived classes to handle timeframe changes
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~PredictionModuleBase()
        {
            Dispose(false);
        }
    }
}
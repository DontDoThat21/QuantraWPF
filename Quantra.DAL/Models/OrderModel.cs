using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Quantra.Models
{
    public class OrderModel : INotifyPropertyChanged
    {
        private int _id;
        private string _symbol = string.Empty;
        private string _orderType = string.Empty;
        private int _quantity;
        private double _price;
        private double _stopLoss;
        private double _takeProfit;
        private bool _isPaperTrade;
        private string _status = string.Empty;
        private string _predictionSource = string.Empty;
        private DateTime _timestamp;

        public event PropertyChangedEventHandler? PropertyChanged;

        public int Id
        {
            get => _id;
            set
            {
                if (_id != value)
                {
                    _id = value;
                    OnPropertyChanged();
                }
            }
        }

        public required string Symbol
        {
            get => _symbol;
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnPropertyChanged();
                }
            }
        }

        public required string OrderType
        {
            get => _orderType;
            set
            {
                if (_orderType != value)
                {
                    _orderType = value;
                    OnPropertyChanged();
                }
            }
        }

        public int Quantity
        {
            get => _quantity;
            set
            {
                if (_quantity != value)
                {
                    _quantity = value;
                    OnPropertyChanged();
                }
            }
        }

        public double Price
        {
            get => _price;
            set
            {
                if (_price != value)
                {
                    _price = value;
                    OnPropertyChanged();
                }
            }
        }

        public double StopLoss
        {
            get => _stopLoss;
            set
            {
                if (_stopLoss != value)
                {
                    _stopLoss = value;
                    OnPropertyChanged();
                }
            }
        }

        public double TakeProfit
        {
            get => _takeProfit;
            set
            {
                if (_takeProfit != value)
                {
                    _takeProfit = value;
                    OnPropertyChanged();
                }
            }
        }

        public bool IsPaperTrade
        {
            get => _isPaperTrade;
            set
            {
                if (_isPaperTrade != value)
                {
                    _isPaperTrade = value;
                    OnPropertyChanged();
                }
            }
        }

        public required string Status
        {
            get => _status;
            set
            {
                if (_status != value)
                {
                    _status = value;
                    OnPropertyChanged();
                }
            }
        }

        public required string PredictionSource
        {
            get => _predictionSource;
            set
            {
                if (_predictionSource != value)
                {
                    _predictionSource = value;
                    OnPropertyChanged();
                }
            }
        }

        public DateTime Timestamp
        {
            get => _timestamp;
            set
            {
                if (_timestamp != value)
                {
                    _timestamp = value;
                    OnPropertyChanged();
                }
            }
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

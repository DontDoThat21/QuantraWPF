using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;
using Quantra.DAL.Utilities;

namespace Quantra.Models
{
    public class TradingRule : INotifyPropertyChanged
    {
        // Basic properties
        private string name = string.Empty;
        private string symbol = string.Empty;
        private string predictedAction = string.Empty;
        private string orderType = string.Empty;
        private string timeframe = string.Empty;
        private string condition = string.Empty;
        private string indicatorsJson = string.Empty;
        private Dictionary<string, double> indicators = new();
        private List<string> conditions = new();
        private DateTime createdDate = DateTime.Now;
        private bool isActive;
        private double minConfidence;
        private double targetPrice;
        private double entryPrice;
        private double exitPrice;
        private double stopLoss;
        private int quantity;
        private string description = string.Empty;
        private double riskRewardRatio;

        public int Id { get; set; }
        public DateTime LastModified { get; set; }

        public string Name
        {
            get => name;
            set => SetField(ref name, value);
        }

        public string Symbol
        {
            get => symbol;
            set => SetField(ref symbol, value);
        }

        public string PredictedAction
        {
            get => predictedAction;
            set => SetField(ref predictedAction, value);
        }

        public string OrderType
        {
            get => orderType;
            set => SetField(ref orderType, value);
        }

        public string Timeframe
        {
            get => timeframe;
            set => SetField(ref timeframe, value);
        }

        public string Condition
        {
            get => condition;
            set => SetField(ref condition, value);
        }

        public Dictionary<string, double> Indicators
        {
            get => indicators;
            set
            {
                if (SetField(ref indicators, value))
                {
                    indicatorsJson = JsonConvert.SerializeObject(value);
                }
            }
        }

        public List<string> Conditions
        {
            get => conditions;
            set => SetField(ref conditions, value);
        }

        public DateTime CreatedDate
        {
            get => createdDate;
            set => SetField(ref createdDate, value);
        }

        public bool IsActive
        {
            get => isActive;
            set => SetField(ref isActive, value);
        }

        public double MinConfidence
        {
            get => minConfidence;
            set => SetField(ref minConfidence, value);
        }

        public double TargetPrice
        {
            get => targetPrice;
            set => SetField(ref targetPrice, value);
        }

        public double EntryPrice
        {
            get => entryPrice;
            set => SetField(ref entryPrice, value);
        }

        public double ExitPrice
        {
            get => exitPrice;
            set => SetField(ref exitPrice, value);
        }

        public double StopLoss
        {
            get => stopLoss;
            set => SetField(ref stopLoss, value);
        }

        public int Quantity
        {
            get => quantity;
            set => SetField(ref quantity, value);
        }

        public string Description
        {
            get => description;
            set => SetField(ref description, value);
        }

        public double RiskRewardRatio
        {
            get => riskRewardRatio;
            set => SetField(ref riskRewardRatio, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public bool Validate()
        {
            if (string.IsNullOrEmpty(Symbol) || string.IsNullOrEmpty(OrderType))
                return false;

            if (EntryPrice <= 0 || ExitPrice <= 0 || StopLoss <= 0)
                return false;

            if (Quantity <= 0)
                return false;

            if (MinConfidence < 0 || MinConfidence > 1)
                return false;

            if (OrderType == "BUY")
            {
                if (ExitPrice <= EntryPrice)
                    return false;
                if (StopLoss >= EntryPrice)
                    return false;
            }

            if (OrderType == "SELL")
            {
                if (ExitPrice >= EntryPrice)
                    return false;
                if (StopLoss <= EntryPrice)
                    return false;
            }

            double reward = Math.Abs(ExitPrice - EntryPrice);
            double risk = Math.Abs(StopLoss - EntryPrice);
            RiskRewardRatio = reward / risk;

            return RiskRewardRatio >= 1.0;
        }

        // Returns a hex color string representing the status color, keeping UI concerns out of the model
        public string GetStatusColorHex()
        {
            return TradingRuleColorHelper.GetStatusColorHex(IsActive, RiskRewardRatio);
        }

        public override string ToString()
        {
            var direction = OrderType == "BUY" ? "Buy" : "Sell";
            return $"{Symbol} {direction} {Quantity} @ {EntryPrice:C2} -> {ExitPrice:C2} (R/R: {RiskRewardRatio:F1})";
        }
    }
}

using System;

namespace Quantra.Models
{
    public class TradeRecord
    {
        public int Id { get; set; }
        public string Symbol { get; set; }
        public string Action { get; set; }
        public double Price { get; set; }
        public double TargetPrice { get; set; }
        public double Confidence { get; set; }
        public DateTime ExecutionTime { get; set; }
        public string Status { get; set; } = "Executed";
        public string Notes { get; set; }
        public int Quantity { get; set; }
        public DateTime TimeStamp { get; set; }
    }
}
using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents an insider transaction from the Alpha Vantage INSIDER_TRANSACTIONS API
    /// </summary>
    public class InsiderTransactionData
    {
        /// <summary>
        /// Stock ticker symbol
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Filing date
        /// </summary>
        public DateTime FilingDate { get; set; }

        /// <summary>
        /// Transaction date
        /// </summary>
        public DateTime TransactionDate { get; set; }

        /// <summary>
        /// Name of the reporting owner
        /// </summary>
        public string OwnerName { get; set; }

        /// <summary>
        /// CIK number of the owner
        /// </summary>
        public string OwnerCik { get; set; }

        /// <summary>
        /// Title or relationship of the owner
        /// </summary>
        public string OwnerTitle { get; set; }

        /// <summary>
        /// Type of security transacted
        /// </summary>
        public string SecurityType { get; set; }

        /// <summary>
        /// Type of transaction code
        /// </summary>
        public string TransactionCode { get; set; }

        /// <summary>
        /// Number of shares involved
        /// </summary>
        public int SharesTraded { get; set; }

        /// <summary>
        /// Price per share
        /// </summary>
        public double PricePerShare { get; set; }

        /// <summary>
        /// Total value of transaction
        /// </summary>
        public double TotalValue => SharesTraded * PricePerShare;

        /// <summary>
        /// Shares owned following the transaction
        /// </summary>
        public int SharesOwnedFollowing { get; set; }

        /// <summary>
        /// Acquisition or disposal (A or D)
        /// </summary>
        public string AcquisitionOrDisposal { get; set; }

        /// <summary>
        /// Friendly transaction type description
        /// </summary>
        public string TransactionTypeDescription
        {
            get
            {
                return TransactionCode switch
                {
                    "P" => "Purchase",
                    "S" => "Sale",
                    "A" => "Award",
                    "D" => "Sale to Issuer",
                    "F" => "Tax Payment",
                    "I" => "Discretionary Transaction",
                    "M" => "Exercise or Conversion",
                    "C" => "Conversion of Derivative",
                    "E" => "Expiration",
                    "G" => "Gift",
                    "H" => "Expiration (Short)",
                    "J" => "Other",
                    "K" => "Voting Trust",
                    "L" => "Small Acquisition",
                    "O" => "Exercise Out-of-Money",
                    "U" => "Disposition to Trust",
                    "W" => "Will or Inheritance",
                    "X" => "Exercise In-the-Money",
                    "Z" => "Deposit into Trust",
                    _ => TransactionCode ?? "Unknown"
                };
            }
        }

        /// <summary>
        /// Whether this is a buy transaction
        /// </summary>
        public bool IsBuy => AcquisitionOrDisposal == "A" || TransactionCode == "P";

        /// <summary>
        /// Whether this is a sell transaction
        /// </summary>
        public bool IsSell => AcquisitionOrDisposal == "D" || TransactionCode == "S";
    }

    /// <summary>
    /// Response container for insider transactions API
    /// </summary>
    public class InsiderTransactionsResponse
    {
        /// <summary>
        /// Stock symbol for the query
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// List of insider transactions
        /// </summary>
        public List<InsiderTransactionData> Transactions { get; set; } = new List<InsiderTransactionData>();
    }
}

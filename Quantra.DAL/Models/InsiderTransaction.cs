using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Model for insider trading transactions
    /// </summary>
    public class InsiderTransaction
    {
        /// <summary>
        /// Stock symbol
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Name of the insider making the transaction
        /// </summary>
        public string InsiderName { get; set; }

        /// <summary>
        /// Title or position of the insider in the company
        /// </summary>
        public string InsiderTitle { get; set; }

        /// <summary>
        /// Type of transaction (Purchase, Sale, Option Exercise, etc.)
        /// </summary>
        public InsiderTransactionType TransactionType { get; set; }

        /// <summary>
        /// Date the transaction occurred
        /// </summary>
        public DateTime TransactionDate { get; set; }

        /// <summary>
        /// Date the transaction was filed with SEC
        /// </summary>
        public DateTime FilingDate { get; set; }

        /// <summary>
        /// Number of shares or contracts involved
        /// </summary>
        public int Quantity { get; set; }

        /// <summary>
        /// Price per share in the transaction
        /// </summary>
        public double Price { get; set; }

        /// <summary>
        /// Total value of the transaction
        /// </summary>
        public double Value => Math.Round(Quantity * Price, 2);

        /// <summary>
        /// For options: strike price
        /// </summary>
        public double? StrikePrice { get; set; }

        /// <summary>
        /// For options: expiration date
        /// </summary>
        public DateTime? ExpirationDate { get; set; }

        /// <summary>
        /// Whether this transaction was made by a notable figure (being tracked)
        /// </summary>
        public bool IsNotableFigure { get; set; }

        /// <summary>
        /// Additional details about the transaction or insider
        /// </summary>
        public string Notes { get; set; }

        /// <summary>
        /// Category of the notable individual (if applicable)
        /// </summary>
        public NotableFigureCategory? NotableCategory { get; set; }

        /// <summary>
        /// Returns a sentiment score for this particular transaction:
        /// 1) Purchase is positive, Sale is negative
        /// 2) Scale is affected by transaction value and insider position
        /// 3) Range is -1.0 to 1.0
        /// </summary>
        public double GetSentimentScore()
        {
            // Base value: positive for purchase, negative for sale
            double baseValue;
            switch (TransactionType)
            {
                case InsiderTransactionType.Purchase:
                    baseValue = 1.0;
                    break;
                case InsiderTransactionType.Sale:
                    baseValue = -1.0;
                    break;
                case InsiderTransactionType.OptionExercise:
                    baseValue = 0.5;  // Less strong signal than direct purchase
                    break;
                case InsiderTransactionType.GrantAward:
                    baseValue = 0.1;  // Minimal signal (routine compensation)
                    break;
                default:
                    baseValue = 0;
                    break;
            }

            // Scale by position importance
            double positionMultiplier = 1.0;
            if (InsiderTitle?.Contains("CEO") == true || InsiderTitle?.Contains("Chief Executive") == true)
                positionMultiplier = 1.5;
            else if (InsiderTitle?.Contains("CFO") == true || InsiderTitle?.Contains("Chief Financial") == true)
                positionMultiplier = 1.3;
            else if (InsiderTitle?.Contains("Director") == true || InsiderTitle?.Contains("Board") == true)
                positionMultiplier = 1.0;
            else
                positionMultiplier = 0.8;

            // Notable figure bonus
            double notableFigureBonus = IsNotableFigure ? 1.2 : 1.0;

            // Calculate raw score
            double rawScore = baseValue * positionMultiplier * notableFigureBonus;

            // Cap at -1.0 to 1.0
            return Math.Max(-1.0, Math.Min(1.0, rawScore));
        }
    }

    /// <summary>
    /// Types of insider transactions
    /// </summary>
    public enum InsiderTransactionType
    {
        Purchase,
        Sale,
        OptionExercise,
        GrantAward,
        Other
    }

    /// <summary>
    /// Categories of notable figures being tracked
    /// </summary>
    public enum NotableFigureCategory
    {
        PoliticalFigure,
        FundManager,
        CorporateExecutive,
        CelebrityInvestor,
        Other
    }
}
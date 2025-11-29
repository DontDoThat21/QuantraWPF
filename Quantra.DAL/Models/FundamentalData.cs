using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Company Overview data from Alpha Vantage OVERVIEW endpoint
    /// </summary>
    public class CompanyOverview
    {
        // Company Information
        public string Symbol { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public string Exchange { get; set; }
        public string Currency { get; set; }
        public string Country { get; set; }
        public string Sector { get; set; }
        public string Industry { get; set; }
        public string Address { get; set; }
        public string FiscalYearEnd { get; set; }

        // Valuation Metrics
        public decimal? MarketCapitalization { get; set; }
        public decimal? EBITDA { get; set; }
        public decimal? PERatio { get; set; }
        public decimal? PEGRatio { get; set; }
        public decimal? BookValue { get; set; }
        public decimal? DividendPerShare { get; set; }
        public decimal? DividendYield { get; set; }
        public decimal? EPS { get; set; }
        public decimal? RevenuePerShareTTM { get; set; }
        public decimal? ProfitMargin { get; set; }
        public decimal? OperatingMarginTTM { get; set; }
        public decimal? ReturnOnAssetsTTM { get; set; }
        public decimal? ReturnOnEquityTTM { get; set; }
        public decimal? RevenueTTM { get; set; }
        public decimal? GrossProfitTTM { get; set; }
        public decimal? DilutedEPSTTM { get; set; }
        public decimal? QuarterlyEarningsGrowthYOY { get; set; }
        public decimal? QuarterlyRevenueGrowthYOY { get; set; }
        public decimal? AnalystTargetPrice { get; set; }
        public decimal? TrailingPE { get; set; }
        public decimal? ForwardPE { get; set; }
        public decimal? PriceToSalesRatioTTM { get; set; }
        public decimal? PriceToBookRatio { get; set; }
        public decimal? EVToRevenue { get; set; }
        public decimal? EVToEBITDA { get; set; }
        public decimal? Beta { get; set; }

        // Price Data
        public decimal? Week52High { get; set; }
        public decimal? Week52Low { get; set; }
        public decimal? Day50MovingAverage { get; set; }
        public decimal? Day200MovingAverage { get; set; }

        // Shares Data
        public long? SharesOutstanding { get; set; }
        public string DividendDate { get; set; }
        public string ExDividendDate { get; set; }

        // Metadata
        public DateTime LastUpdated { get; set; }

        /// <summary>
        /// Formats market cap for display (in Trillions/Billions/Millions)
        /// </summary>
        public string FormattedMarketCap
        {
            get
            {
                if (!MarketCapitalization.HasValue) return "N/A";
                var value = MarketCapitalization.Value;
                if (value >= 1_000_000_000_000) return $"${value / 1_000_000_000_000:F2}T";
                if (value >= 1_000_000_000) return $"${value / 1_000_000_000:F2}B";
                if (value >= 1_000_000) return $"${value / 1_000_000:F2}M";
                return $"${value:N0}";
            }
        }
    }

    /// <summary>
    /// Annual/Quarterly Income Statement Report
    /// </summary>
    public class IncomeStatementReport
    {
        public string FiscalDateEnding { get; set; }
        public string ReportedCurrency { get; set; }
        public decimal? GrossProfit { get; set; }
        public decimal? TotalRevenue { get; set; }
        public decimal? CostOfRevenue { get; set; }
        public decimal? CostOfGoodsAndServicesSold { get; set; }
        public decimal? OperatingIncome { get; set; }
        public decimal? SellingGeneralAndAdministrative { get; set; }
        public decimal? ResearchAndDevelopment { get; set; }
        public decimal? OperatingExpenses { get; set; }
        public decimal? InvestmentIncomeNet { get; set; }
        public decimal? NetInterestIncome { get; set; }
        public decimal? InterestIncome { get; set; }
        public decimal? InterestExpense { get; set; }
        public decimal? NonInterestIncome { get; set; }
        public decimal? OtherNonOperatingIncome { get; set; }
        public decimal? Depreciation { get; set; }
        public decimal? DepreciationAndAmortization { get; set; }
        public decimal? IncomeBeforeTax { get; set; }
        public decimal? IncomeTaxExpense { get; set; }
        public decimal? InterestAndDebtExpense { get; set; }
        public decimal? NetIncomeFromContinuingOperations { get; set; }
        public decimal? ComprehensiveIncomeNetOfTax { get; set; }
        public decimal? EBIT { get; set; }
        public decimal? EBITDA { get; set; }
        public decimal? NetIncome { get; set; }

        /// <summary>
        /// Formats currency values in Billions
        /// </summary>
        public string FormatAsBillions(decimal? value)
        {
            if (!value.HasValue) return "N/A";
            return $"${value.Value / 1_000_000_000:F2}B";
        }
    }

    /// <summary>
    /// Income Statement data from Alpha Vantage INCOME_STATEMENT endpoint
    /// </summary>
    public class IncomeStatement
    {
        public string Symbol { get; set; }
        public List<IncomeStatementReport> AnnualReports { get; set; } = new List<IncomeStatementReport>();
        public List<IncomeStatementReport> QuarterlyReports { get; set; } = new List<IncomeStatementReport>();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// Annual/Quarterly Balance Sheet Report
    /// </summary>
    public class BalanceSheetReport
    {
        public string FiscalDateEnding { get; set; }
        public string ReportedCurrency { get; set; }
        public decimal? TotalAssets { get; set; }
        public decimal? TotalCurrentAssets { get; set; }
        public decimal? CashAndCashEquivalentsAtCarryingValue { get; set; }
        public decimal? CashAndShortTermInvestments { get; set; }
        public decimal? Inventory { get; set; }
        public decimal? CurrentNetReceivables { get; set; }
        public decimal? TotalNonCurrentAssets { get; set; }
        public decimal? PropertyPlantEquipment { get; set; }
        public decimal? AccumulatedDepreciationAmortizationPPE { get; set; }
        public decimal? IntangibleAssets { get; set; }
        public decimal? IntangibleAssetsExcludingGoodwill { get; set; }
        public decimal? Goodwill { get; set; }
        public decimal? Investments { get; set; }
        public decimal? LongTermInvestments { get; set; }
        public decimal? ShortTermInvestments { get; set; }
        public decimal? OtherCurrentAssets { get; set; }
        public decimal? OtherNonCurrentAssets { get; set; }
        public decimal? TotalLiabilities { get; set; }
        public decimal? TotalCurrentLiabilities { get; set; }
        public decimal? CurrentAccountsPayable { get; set; }
        public decimal? DeferredRevenue { get; set; }
        public decimal? CurrentDebt { get; set; }
        public decimal? ShortTermDebt { get; set; }
        public decimal? TotalNonCurrentLiabilities { get; set; }
        public decimal? CapitalLeaseObligations { get; set; }
        public decimal? LongTermDebt { get; set; }
        public decimal? CurrentLongTermDebt { get; set; }
        public decimal? LongTermDebtNoncurrent { get; set; }
        public decimal? ShortLongTermDebtTotal { get; set; }
        public decimal? OtherCurrentLiabilities { get; set; }
        public decimal? OtherNonCurrentLiabilities { get; set; }
        public decimal? TotalShareholderEquity { get; set; }
        public decimal? TreasuryStock { get; set; }
        public decimal? RetainedEarnings { get; set; }
        public decimal? CommonStock { get; set; }
        public decimal? CommonStockSharesOutstanding { get; set; }

        /// <summary>
        /// Current Ratio = Current Assets / Current Liabilities
        /// </summary>
        public decimal? CurrentRatio
        {
            get
            {
                if (!TotalCurrentAssets.HasValue || !TotalCurrentLiabilities.HasValue || TotalCurrentLiabilities.Value == 0)
                    return null;
                return TotalCurrentAssets.Value / TotalCurrentLiabilities.Value;
            }
        }

        /// <summary>
        /// Debt to Equity Ratio = Total Liabilities / Total Shareholder Equity
        /// </summary>
        public decimal? DebtToEquityRatio
        {
            get
            {
                if (!TotalLiabilities.HasValue || !TotalShareholderEquity.HasValue || TotalShareholderEquity.Value == 0)
                    return null;
                return TotalLiabilities.Value / TotalShareholderEquity.Value;
            }
        }
    }

    /// <summary>
    /// Balance Sheet data from Alpha Vantage BALANCE_SHEET endpoint
    /// </summary>
    public class BalanceSheet
    {
        public string Symbol { get; set; }
        public List<BalanceSheetReport> AnnualReports { get; set; } = new List<BalanceSheetReport>();
        public List<BalanceSheetReport> QuarterlyReports { get; set; } = new List<BalanceSheetReport>();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// Annual/Quarterly Cash Flow Report
    /// </summary>
    public class CashFlowReport
    {
        public string FiscalDateEnding { get; set; }
        public string ReportedCurrency { get; set; }
        public decimal? OperatingCashflow { get; set; }
        public decimal? PaymentsForOperatingActivities { get; set; }
        public decimal? ProceedsFromOperatingActivities { get; set; }
        public decimal? ChangeInOperatingLiabilities { get; set; }
        public decimal? ChangeInOperatingAssets { get; set; }
        public decimal? DepreciationDepletionAndAmortization { get; set; }
        public decimal? CapitalExpenditures { get; set; }
        public decimal? ChangeInReceivables { get; set; }
        public decimal? ChangeInInventory { get; set; }
        public decimal? ProfitLoss { get; set; }
        public decimal? CashflowFromInvestment { get; set; }
        public decimal? CashflowFromFinancing { get; set; }
        public decimal? ProceedsFromRepaymentsOfShortTermDebt { get; set; }
        public decimal? PaymentsForRepurchaseOfCommonStock { get; set; }
        public decimal? PaymentsForRepurchaseOfEquity { get; set; }
        public decimal? PaymentsForRepurchaseOfPreferredStock { get; set; }
        public decimal? DividendPayout { get; set; }
        public decimal? DividendPayoutCommonStock { get; set; }
        public decimal? DividendPayoutPreferredStock { get; set; }
        public decimal? ProceedsFromIssuanceOfCommonStock { get; set; }
        public decimal? ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet { get; set; }
        public decimal? ProceedsFromIssuanceOfPreferredStock { get; set; }
        public decimal? ProceedsFromRepurchaseOfEquity { get; set; }
        public decimal? ProceedsFromSaleOfTreasuryStock { get; set; }
        public decimal? ChangeInCashAndCashEquivalents { get; set; }
        public decimal? ChangeInExchangeRate { get; set; }
        public decimal? NetIncome { get; set; }

        /// <summary>
        /// Free Cash Flow = Operating Cash Flow - Capital Expenditures
        /// </summary>
        public decimal? FreeCashFlow
        {
            get
            {
                if (!OperatingCashflow.HasValue) return null;
                var capex = CapitalExpenditures ?? 0;
                return OperatingCashflow.Value - Math.Abs(capex);
            }
        }
    }

    /// <summary>
    /// Cash Flow data from Alpha Vantage CASH_FLOW endpoint
    /// </summary>
    public class CashFlowStatement
    {
        public string Symbol { get; set; }
        public List<CashFlowReport> AnnualReports { get; set; } = new List<CashFlowReport>();
        public List<CashFlowReport> QuarterlyReports { get; set; } = new List<CashFlowReport>();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// Annual Earnings Report
    /// </summary>
    public class AnnualEarningsReport
    {
        public string FiscalDateEnding { get; set; }
        public decimal? ReportedEPS { get; set; }
    }

    /// <summary>
    /// Quarterly Earnings Report
    /// </summary>
    public class QuarterlyEarningsReport
    {
        public string FiscalDateEnding { get; set; }
        public string ReportedDate { get; set; }
        public decimal? ReportedEPS { get; set; }
        public decimal? EstimatedEPS { get; set; }
        public decimal? Surprise { get; set; }
        public decimal? SurprisePercentage { get; set; }
    }

    /// <summary>
    /// Earnings data from Alpha Vantage EARNINGS endpoint
    /// </summary>
    public class EarningsData
    {
        public string Symbol { get; set; }
        public List<AnnualEarningsReport> AnnualEarnings { get; set; } = new List<AnnualEarningsReport>();
        public List<QuarterlyEarningsReport> QuarterlyEarnings { get; set; } = new List<QuarterlyEarningsReport>();
        public DateTime LastUpdated { get; set; }

        /// <summary>
        /// Gets the next upcoming earnings date if available
        /// </summary>
        public string UpcomingEarningsDate
        {
            get
            {
                if (QuarterlyEarnings == null || QuarterlyEarnings.Count == 0)
                    return "N/A";
                
                var latestReportDate = QuarterlyEarnings[0]?.ReportedDate;
                if (string.IsNullOrEmpty(latestReportDate))
                    return "N/A";

                // Estimate next earnings date (approximately 3 months from last report)
                if (DateTime.TryParse(latestReportDate, out var lastDate))
                {
                    var estimatedNext = lastDate.AddMonths(3);
                    if (estimatedNext > DateTime.Now)
                        return $"~{estimatedNext:MMM dd, yyyy}";
                }
                
                return "Check company IR";
            }
        }
    }
}

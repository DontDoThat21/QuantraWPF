using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Linq;
using Quantra.Models;

namespace Quantra.Repositories
{
    public class PredictionAnalysisRepository
    {
        // Remove in-memory _db, use database

        public void SaveAnalysisResults(IEnumerable<PredictionAnalysisResult> results)
        {
            // Optionally implement DB save if needed
        }

        public List<PredictionAnalysisResult> GetLatestAnalyses(int count = 50)
        {
            var result = new List<PredictionAnalysisResult>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    string sql = @"
                        SELECT p1.Id, p1.Symbol, p1.PredictedAction, p1.Confidence, p1.CurrentPrice, p1.TargetPrice, p1.PotentialReturn, p1.CreatedDate
                        FROM StockPredictions p1
                        INNER JOIN (
                            SELECT Symbol, MAX(CreatedDate) AS MaxDate
                            FROM StockPredictions
                            GROUP BY Symbol
                        ) p2 ON p1.Symbol = p2.Symbol AND p1.CreatedDate = p2.MaxDate
                        ORDER BY p1.Confidence DESC
                        LIMIT @Count
                    ";
                    using (var command = new SQLiteCommand(sql, connection))
                    {
                        command.Parameters.AddWithValue("@Count", count);
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                var model = new PredictionAnalysisResult
                                {
                                    Id = reader.GetInt32(reader.GetOrdinal("Id")),
                                    Symbol = reader["Symbol"].ToString(),
                                    PredictedAction = reader["PredictedAction"].ToString(),
                                    Confidence = reader.GetDouble(reader.GetOrdinal("Confidence")),
                                    CurrentPrice = reader.GetDouble(reader.GetOrdinal("CurrentPrice")),
                                    TargetPrice = reader.GetDouble(reader.GetOrdinal("TargetPrice")),
                                    PotentialReturn = reader.GetDouble(reader.GetOrdinal("PotentialReturn")),
                                    TradingRule = null, // TradingRule column removed from query
                                    AnalysisTime = reader.GetDateTime(reader.GetOrdinal("CreatedDate")),
                                    Indicators = new Dictionary<string, double>()
                                };
                                // Load indicators for this prediction
                                using (var indCmd = new SQLiteCommand("SELECT IndicatorName, IndicatorValue FROM PredictionIndicators WHERE PredictionId = @PredictionId", connection))
                                {
                                    indCmd.Parameters.AddWithValue("@PredictionId", model.Id);
                                    using (var indReader = indCmd.ExecuteReader())
                                    {
                                        while (indReader.Read())
                                        {
                                            string name = indReader["IndicatorName"].ToString();
                                            double value = Convert.ToDouble(indReader["IndicatorValue"]);
                                            model.Indicators[name] = value;
                                        }
                                    }
                                }
                                result.Add(model);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error retrieving latest analyses from database", ex.ToString());
            }
            return result;
        }

        public List<string> GetSymbols()
        {
            var symbols = new List<string>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var command = new SQLiteCommand("SELECT DISTINCT Symbol FROM StockPredictions", connection))
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            symbols.Add(reader["Symbol"].ToString());
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error retrieving symbols from database", ex.ToString());
            }
            return symbols;
        }

        public PredictionAnalysisResult AnalyzeSymbol(string symbol)
        {
            // Optionally implement DB-backed or ML-backed analysis
            return null;
        }

        // Overload to support strategy profile
        public PredictionAnalysisResult AnalyzeSymbol(string symbol, StrategyProfile strategy)
        {
            // Use the provided strategy for analysis
            // Example: run backtest or generate signals using the strategy
            var historical = GetHistoricalPrices(symbol); // Use existing method
            var signal = strategy.GenerateSignal(historical, historical.Count - 1);
            // ...rest of analysis logic...
            return new PredictionAnalysisResult
            {
                Symbol = symbol,
                PredictedAction = signal ?? "HOLD",
                Confidence = 0.75, // Example
                CurrentPrice = historical.LastOrDefault()?.Close ?? 0,
                TargetPrice = (historical.LastOrDefault()?.Close ?? 0) * 1.05,
                PotentialReturn = 0.05,
                TradingRule = strategy.Name,
                Indicators = new Dictionary<string, double>()
            };
        }

        // Returns historical price data for a symbol, ordered by date ascending
        public List<HistoricalPrice> GetHistoricalPrices(string symbol)
        {
            var prices = new List<HistoricalPrice>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    string sql = @"SELECT Date, Open, High, Low, Close, Volume, AdjClose FROM HistoricalPrices WHERE Symbol = @Symbol ORDER BY Date ASC";
                    using (var command = new SQLiteCommand(sql, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol);
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                prices.Add(new HistoricalPrice
                                {
                                    Date = reader.GetDateTime(reader.GetOrdinal("Date")),
                                    Open = reader.GetDouble(reader.GetOrdinal("Open")),
                                    High = reader.GetDouble(reader.GetOrdinal("High")),
                                    Low = reader.GetDouble(reader.GetOrdinal("Low")),
                                    Close = reader.GetDouble(reader.GetOrdinal("Close")),
                                    Volume = reader.GetInt64(reader.GetOrdinal("Volume")),
                                    AdjClose = reader.GetDouble(reader.GetOrdinal("AdjClose"))
                                });
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error retrieving historical prices for {symbol}", ex.ToString());
            }
            return prices;
        }
    }
}

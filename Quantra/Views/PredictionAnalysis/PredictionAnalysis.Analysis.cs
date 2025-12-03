using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Adapters;
using Quantra.Models; // Ensures Quantra.Models.PredictionModel is accessible
using System.Windows;
using System.Windows.Controls;
using Quantra.DAL.Services; // Ensures Quantra.Models.PredictionModel is accessible
using Microsoft.EntityFrameworkCore;

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl
    {
        // NOTE: These service fields are initialized via DI in the main constructor (xaml.cs)
        // They are declared here for clarity but assigned in the main partial class constructor
        private TwitterSentimentService _twitterSentimentService;
        private FinancialNewsSentimentService _financialNewsSentimentService;
        private IEarningsTranscriptService _earningsTranscriptService;
        private IAnalystRatingService _analystRatingService;
        private IInsiderTradingService _insiderTradingService;
        private UserSettings _userSettings;
        
        // Sentiment metrics storage
        private double sentimentScore = 0;
        private string earningsKeyTopics = string.Empty;
        private string earningsKeyEntities = string.Empty;
        private string earningsQuarter = string.Empty;

        // Changed return type to Quantra.Models.PredictionModel
        private async Task<Quantra.Models.PredictionModel> AnalyzeStockWithAllAlgorithms(string symbol)
        {
            try
            {
                // Ensure trading bot is initialized
                if (_tradingBot == null)
                {
                    InitializeTradingComponents();
                }

                // Get current stock price using AlphaVantageService
                double currentPrice = 0.0;
                try 
                {
                    var quote = await _alphaVantageService.GetQuoteDataAsync(symbol);
                    currentPrice = quote?.Price ?? 0.0;
                    
                    // If quote price is 0, log a warning
                    if (currentPrice <= 0)
                    {
                        _loggingService?.Log("Warning", $"Quote price is 0 for {symbol}, prediction may be inaccurate");
                    }
                } 
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Failed to get current price for {symbol}: {ex.Message}");
                    currentPrice = 0.0;
                }

                Dictionary<string, double> indicators = new Dictionary<string, double>();

                // API Calls for technical indicators removed.
                //DatabaseMonolith.Log("Warning", $"API calls for technical indicators for {symbol} have been removed. Indicators will be empty.");

                // Sentiment analysis integration
                try
                {
                    // Twitter sentiment analysis
                    double twitterSentiment = await _twitterSentimentService.GetSymbolSentimentAsync(symbol);
                    indicators["TwitterSentiment"] = twitterSentiment;
                    
                    // Financial news sentiment analysis
                    double financialNewsSentiment = await _financialNewsSentimentService.GetSymbolSentimentAsync(symbol);
                    indicators["FinancialNewsSentiment"] = financialNewsSentiment;
                    
                    // Earnings call transcript analysis
                    double earningsTranscriptSentiment = await _earningsTranscriptService.GetSymbolSentimentAsync(symbol);
                    indicators["EarningsTranscriptSentiment"] = earningsTranscriptSentiment;
                    
                    // Analyst rating sentiment analysis
                    double analystRatingSentiment = await _analystRatingService.GetAnalystSentimentAsync(symbol);
                    indicators["AnalystRatingSentiment"] = analystRatingSentiment;
                    
                    // Insider trading sentiment analysis
                    double insiderTradingSentiment = await _insiderTradingService.GetInsiderSentimentAsync(symbol);
                    indicators["InsiderTradingSentiment"] = insiderTradingSentiment;
                    
                    // Get user settings for sentiment weights
                    var userSettings = _userSettings ?? new UserSettings();
                    double analystWeight = userSettings.AnalystRatingSentimentWeight;
                    double insiderWeight = userSettings.InsiderTradingSentimentWeight;
                    
                    // Calculate combined sentiment (weighted average with configurable weights)
                    double combinedSentiment = (twitterSentiment + 
                                               (financialNewsSentiment * 2) + (earningsTranscriptSentiment * 3) +
                                               (analystRatingSentiment * analystWeight) + 
                                               (insiderTradingSentiment * insiderWeight)) / (6.0 + analystWeight + insiderWeight);
                    indicators["SocialSentiment"] = combinedSentiment;
                    
                    // Store earnings transcript analysis details for additional insights
                    try
                    {
                        var transcriptAnalysis = await _earningsTranscriptService.GetEarningsTranscriptAnalysisAsync(symbol);
                        if (transcriptAnalysis != null)
                        {
                            // Store sentiment distribution
                            if (transcriptAnalysis.SentimentDistribution != null)
                            {
                                foreach (var kvp in transcriptAnalysis.SentimentDistribution)
                                {
                                    indicators[$"Earnings_{kvp.Key}"] = kvp.Value;
                                }
                            }
                            
                            // Store key topics as a concatenated string in the prediction model notes
                            if (transcriptAnalysis.KeyTopics != null && transcriptAnalysis.KeyTopics.Count > 0)
                            {
                                earningsKeyTopics = string.Join(", ", transcriptAnalysis.KeyTopics);
                            }
                            
                            // Store important entities
                            if (transcriptAnalysis.NamedEntities != null && 
                                transcriptAnalysis.NamedEntities.TryGetValue("ORG", out var orgs) && 
                                orgs.Count > 0)
                            {
                                earningsKeyEntities = string.Join(", ", orgs.Take(5));
                            }
                            
                            // Store quarter information
                            earningsQuarter = transcriptAnalysis.Quarter;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", "Error fetching detailed earnings transcript analysis", ex.ToString());
                    }
                    
                    // Store individual news source sentiment for detailed analysis if needed
                    try
                    {
                        var newsSourceSentiment = await _financialNewsSentimentService.GetDetailedSourceSentimentAsync(symbol);
                        foreach (var kvp in newsSourceSentiment)
                        {
                            indicators[$"News_{kvp.Key.Replace(".", "_")}"] = kvp.Value;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", "Error fetching detailed news source sentiment", ex.ToString());
                    }
                    
                    // Store analyst rating data for detailed analysis
                    try
                    {
                        var analystRatings = await _analystRatingService.GetAggregatedRatingsAsync(symbol);
                        if (analystRatings != null)
                        {
                            indicators["AnalystConsensus"] = analystRatings.ConsensusScore;
                            indicators["AnalystBuyCount"] = analystRatings.BuyCount;
                            indicators["AnalystHoldCount"] = analystRatings.HoldCount;
                            indicators["AnalystSellCount"] = analystRatings.SellCount;
                            
                            if (analystRatings.AveragePriceTarget > 0)
                                indicators["AnalystPriceTarget"] = analystRatings.AveragePriceTarget;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", "Error fetching analyst rating data", ex.ToString());
                    }

                    // Store insider trading data for detailed analysis
                    try
                    {
                        var insiderMetrics = await _insiderTradingService.GetInsiderMetricsAsync(symbol);
                        foreach (var kvp in insiderMetrics)
                        {
                            indicators[$"Insider_{kvp.Key}"] = kvp.Value;
                        }
                        
                        // Also retrieve notable insider sentiment
                        var notableInsiderSentiment = await _insiderTradingService.GetNotableIndividualSentimentAsync(symbol);
                        foreach (var kvp in notableInsiderSentiment)
                        {
                            indicators[$"Notable_{kvp.Key.Replace(" ", "_")}"] = kvp.Value;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", "Error fetching insider trading metrics", ex.ToString());
                    }
                    
                    // Set the main sentiment score for later use
                    sentimentScore = combinedSentiment;
                }
                catch (Exception ex) 
                { 
                    //DatabaseMonolith.Log("Warning", "Error fetching social media sentiment", ex.ToString());
                }

                // --- PRIMARY: Call PythonStockPredictor for ML-based prediction using trained models ---
                string action = "HOLD";
                double confidence = 0.5;
                double targetPrice = currentPrice;
                Dictionary<string, double> weights = null;
                List<string> featureNames = null;

                // CRITICAL: Always pass current_price to Python for proper target price calculation
                // The ML model predicts percentage change, which MUST be converted to actual price
                // Python cannot calculate target price correctly without this value
                if (currentPrice > 0)
                {
                    indicators["current_price"] = currentPrice;
                }
                else
                {
                    // If we don't have current price, try to use any price indicator as a fallback
                    // This should rarely happen but prevents completely broken predictions
                    if (indicators.Count > 0)
                    {
                        // Try to find any price-like indicator
                        var priceKey = indicators.Keys.FirstOrDefault(k => 
                            k.ToLower().Contains("price") || 
                            k.ToLower().Contains("close") ||
                            k.ToLower().Contains("open"));
                        
                        if (priceKey != null && indicators[priceKey] > 0)
                        {
                            currentPrice = indicators[priceKey];
                            indicators["current_price"] = currentPrice;
                            _loggingService?.Log("Warning", $"Using {priceKey} ({currentPrice:F2}) as fallback current_price for {symbol}");
                        }
                        else
                        {
                            _loggingService?.Log("Error", $"No valid price data available for {symbol}, prediction will be unreliable");
                        }
                    }
                }

                // This should ALWAYS use the trained ML model from Python
                // The Python script will automatically load the trained model or train a new one if needed
                try
                {
                    var result = await Quantra.Models.PythonStockPredictor.PredictAsync(indicators);
                    if (result == null)
                        throw new Exception("Failed to get prediction result from trained ML model");

                    action = result.Action;
                    confidence = result.Confidence;
                    targetPrice = result.TargetPrice;
                    weights = result.FeatureWeights;
                    
                    //DatabaseMonolith.Log("Info", $"ML prediction for {symbol}: {action} with {confidence:P0} confidence (model-based)");
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Python ML prediction failed for {symbol}", ex.ToString());
                    // NOTE: Consider this a critical error - predictions should come from trained models
                    // If the model isn't trained or Python fails, we should notify the user to train the model
                    // rather than silently falling back to rule-based logic
                    throw new InvalidOperationException(
                        $"ML prediction failed for {symbol}. Please ensure the ML model is trained. " +
                        $"Use the 'Train Model' button to train the model with historical data.", ex);
                }

                // Incorporate sentiment into confidence/decision
                if (indicators.ContainsKey("SocialSentiment"))
                {
                    double socialSentiment = indicators["SocialSentiment"];
                    // If sentiment is strongly positive/negative, nudge confidence and action
                    if (socialSentiment > 0.2) { action = "BUY"; confidence += 0.15; }
                    else if (socialSentiment < -0.2) { action = "SELL"; confidence += 0.15; }
                    
                    // Adjust target price based on sentiment strength
                    if (action == "BUY" && socialSentiment > 0.4) {
                        targetPrice *= 1.02; // 2% higher target for strong positive sentiment
                    }
                    else if (action == "SELL" && socialSentiment < -0.4) {
                        targetPrice *= 0.98; // 2% lower target for strong negative sentiment
                    }
                }
                
                // Incorporate insider trading into prediction
                if (indicators.ContainsKey("InsiderTradingSentiment"))
                {
                    double insiderSentiment = indicators["InsiderTradingSentiment"];
                    
                    // Strong insider sentiment can influence action and confidence
                    if (insiderSentiment > 0.4) {
                        // Strong insider buying
                        if (action != "BUY") {
                            action = "BUY";
                            confidence = Math.Max(confidence, 0.75); // Even higher than analyst consensus
                        } else {
                            confidence += 0.25; // Strongly reinforce existing buy decision
                        }
                    }
                    else if (insiderSentiment < -0.4) {
                        // Strong insider selling
                        if (action != "SELL") {
                            action = "SELL";
                            confidence = Math.Max(confidence, 0.75);
                        } else {
                            confidence += 0.25; // Strongly reinforce existing sell decision
                        }
                    }
                    
                    // Adjust target price based on insider activity strength
                    if (action == "BUY" && insiderSentiment > 0.2) {
                        targetPrice *= 1.03; // 3% higher target for positive insider activity
                    }
                    else if (action == "SELL" && insiderSentiment < -0.2) {
                        targetPrice *= 0.97; // 3% lower target for negative insider activity
                    }
                    
                    // CEO transactions have special significance
                    if (indicators.ContainsKey("Insider_CEOTransactionValue"))
                    {
                        double ceoValue = indicators["Insider_CEOTransactionValue"];
                        if (Math.Abs(ceoValue) > 500000) // Large CEO transaction
                        {
                            // If CEO is buying, strengthen buy signal or weaken sell signal
                            if (ceoValue > 0) {
                                if (action == "BUY") confidence += 0.1;
                                else if (action == "SELL") confidence -= 0.15;
                            }
                            // If CEO is selling, strengthen sell signal or weaken buy signal
                            else if (ceoValue < 0) {
                                if (action == "SELL") confidence += 0.1;
                                else if (action == "BUY") confidence -= 0.15;
                            }
                        }
                    }
                }
                
                // Incorporate analyst consensus into prediction
                if (indicators.ContainsKey("AnalystConsensus"))
                {
                    double analystConsensus = indicators["AnalystConsensus"];
                    
                    // Strong analyst consensus can override or reinforce the decision
                    if (analystConsensus > 0.5) {
                        // Strong buy consensus
                        if (action != "BUY") {
                            action = "BUY";
                            // Higher confidence for consensus-driven actions
                            confidence = Math.Max(confidence, 0.7);
                        } else {
                            confidence += 0.2; // Reinforce existing buy decision
                        }
                    }
                    else if (analystConsensus < -0.5) {
                        // Strong sell consensus
                        if (action != "SELL") {
                            action = "SELL"; 
                            confidence = Math.Max(confidence, 0.7);
                        } else {
                            confidence += 0.2; // Reinforce existing sell decision
                        }
                    }
                    
                    // Use analyst price target when available
                    if (indicators.ContainsKey("AnalystPriceTarget") && indicators["AnalystPriceTarget"] > 0)
                    {
                        double analystPriceTarget = indicators["AnalystPriceTarget"];
                        
                        // Blend ML model prediction with analyst target
                        if (currentPrice > 0) {
                            targetPrice = (targetPrice + analystPriceTarget) / 2.0;
                        } else {
                            targetPrice = analystPriceTarget; // Use analyst target if no current price available
                        }
                    }
                }
                else if (indicators.ContainsKey("EarningsTranscriptSentiment"))
                {
                    // Prioritize earnings transcript sentiment if combined is not available
                    double earningsSentiment = indicators["EarningsTranscriptSentiment"];
                    // Earnings calls have more weight than other sentiment sources
                    if (earningsSentiment > 0.15) { action = "BUY"; confidence += 0.2; }
                    else if (earningsSentiment < -0.15) { action = "SELL"; confidence += 0.2; }
                }
                else if (indicators.ContainsKey("TwitterSentiment"))
                {
                    // Fallback to Twitter sentiment if others are not available
                    double twitterSentiment = indicators["TwitterSentiment"];
                    if (twitterSentiment > 0.2) { action = "BUY"; confidence += 0.1; }
                    else if (twitterSentiment < -0.2) { action = "SELL"; confidence += 0.1; }
                }
                else if (indicators.ContainsKey("FinancialNewsSentiment"))
                {
                    // Secondary fallback to financial news sentiment
                    double newsSentiment = indicators["FinancialNewsSentiment"];
                    if (newsSentiment > 0.15) { action = "BUY"; confidence += 0.15; }
                    else if (newsSentiment < -0.15) { action = "SELL"; confidence += 0.15; }
                }

                // Calculate potential return
                // Positive return = price expected to rise (BUY signal)
                // Negative return = price expected to fall (SELL signal)
                double potentialReturn = 0;
                if (currentPrice != 0) // Avoid division by zero
                {
                    potentialReturn = (targetPrice - currentPrice) / currentPrice;
                }
                
                //DatabaseMonolith.Log("Debug", $"AnalyzeStockWithAllAlgorithms (ML+API): symbol={symbol}, action={action}, currentPrice={currentPrice}, targetPrice={targetPrice}, potentialReturn={potentialReturn}, sentiment={sentimentScore}, earnings_quarter={earningsQuarter}, weights={weights}");

                // Run sentiment-price correlation analysis if not already done
                try
                {
                    if (_sentimentCorrelationAnalysis == null)
                    {
                        InitializeSentimentCorrelationAnalysis();
                    }
                    
                    // Analyze sentiment-price correlation with 30-day lookback
                    await AnalyzeSentimentPriceCorrelation(symbol, 30);
                    
                    // Correlation analysis results will be used to adjust prediction in IntegrateSentimentCorrelationIntoPrediction
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", "Error performing sentiment-price correlation analysis", ex.ToString());
                }

                // Create the prediction model with all available data
                var predictionModel = new Quantra.Models.PredictionModel
                {
                    Symbol = symbol,
                    PredictedAction = action,
                    Confidence = confidence,
                    CurrentPrice = currentPrice,
                    TargetPrice = targetPrice,
                    Indicators = indicators,
                    PotentialReturn = potentialReturn,
                    PredictionDate = DateTime.Now,
                    Notes = !string.IsNullOrEmpty(earningsKeyTopics) ? 
                           $"Earnings Topics: {earningsKeyTopics}" : string.Empty
                };
                
                // Add earnings entities if available
                if (!string.IsNullOrEmpty(earningsKeyEntities))
                {
                    predictionModel.Notes += string.IsNullOrEmpty(predictionModel.Notes) ? 
                        $"Key Entities: {earningsKeyEntities}" : 
                        $"\nKey Entities: {earningsKeyEntities}";
                }
                
                // Add earnings quarter if available
                if (!string.IsNullOrEmpty(earningsQuarter))
                {
                    predictionModel.Notes += string.IsNullOrEmpty(predictionModel.Notes) ? 
                        $"Latest Earnings: {earningsQuarter}" : 
                        $"\nLatest Earnings: {earningsQuarter}";
                }
                
                // Integrate sentiment-price correlation data if available
                if (_lastSentimentCorrelation != null)
                {
                    // Incorporate sentiment correlation data into prediction
                    IntegrateSentimentCorrelationIntoPrediction(ref predictionModel);
                }

                // Add analyst ratings information if available
                if (indicators.ContainsKey("AnalystConsensus") && 
                    indicators.ContainsKey("AnalystBuyCount") && 
                    indicators.ContainsKey("AnalystHoldCount") && 
                    indicators.ContainsKey("AnalystSellCount"))
                {
                    string analystConsensusText = string.Empty;
                    double consensus = indicators["AnalystConsensus"];
                    int buyCount = (int)indicators["AnalystBuyCount"];
                    int holdCount = (int)indicators["AnalystHoldCount"];
                    int sellCount = (int)indicators["AnalystSellCount"];
                    
                    if (consensus > 0.5) analystConsensusText = "Strong Buy";
                    else if (consensus > 0.2) analystConsensusText = "Buy";
                    else if (consensus > -0.2) analystConsensusText = "Hold";
                    else if (consensus > -0.5) analystConsensusText = "Sell";
                    else analystConsensusText = "Strong Sell";
                    
                    string analystBreakdown = $"Analyst Consensus: {analystConsensusText} ({buyCount} Buy, {holdCount} Hold, {sellCount} Sell)";
                    
                    // Add price target if available
                    if (indicators.ContainsKey("AnalystPriceTarget") && indicators["AnalystPriceTarget"] > 0)
                    {
                        analystBreakdown += $"\nPrice Target: ${indicators["AnalystPriceTarget"]:F2}";
                    }
                    
                    predictionModel.Notes += string.IsNullOrEmpty(predictionModel.Notes) ? 
                        analystBreakdown : $"\n{analystBreakdown}";
                }

                // Add insider trading information if available
                if (indicators.ContainsKey("InsiderTradingSentiment"))
                {
                    double insiderSentiment = indicators["InsiderTradingSentiment"];
                    
                    // Create insider trading summary text based on sentiment
                    string insiderSentimentText;
                    if (insiderSentiment > 0.5) insiderSentimentText = "Very Positive";
                    else if (insiderSentiment > 0.2) insiderSentimentText = "Positive";
                    else if (insiderSentiment > -0.2) insiderSentimentText = "Neutral";
                    else if (insiderSentiment > -0.5) insiderSentimentText = "Negative";
                    else insiderSentimentText = "Very Negative";
                    
                    string insiderBreakdown = $"Insider Sentiment: {insiderSentimentText}";
                    
                    // Add insider buy/sell ratio if available
                    if (indicators.ContainsKey("Insider_BuySellRatio"))
                    {
                        double buySellRatio = indicators["Insider_BuySellRatio"];
                        insiderBreakdown += $" (Buy/Sell Ratio: {buySellRatio:F2})";
                    }
                    
                    // Add notable figure transactions if available
                    if (indicators.ContainsKey("Insider_NotableFigurePercentage") && 
                        indicators["Insider_NotableFigurePercentage"] > 0)
                    {
                        double notablePct = indicators["Insider_NotableFigurePercentage"] * 100;
                        insiderBreakdown += $"\nNotable Figure Activity: {notablePct:F1}% of transactions";
                        
                        // Add specific notable individuals with strong signals
                        var notableNames = indicators.Keys
                            .Where(k => k.StartsWith("Notable_") && Math.Abs(indicators[k]) > 0.3)
                            .OrderByDescending(k => Math.Abs(indicators[k]))
                            .Take(3);
                        
                        if (notableNames.Any())
                        {
                            insiderBreakdown += "\nNotable Individuals:";
                            foreach (var key in notableNames)
                            {
                                string name = key.Replace("Notable_", "").Replace("_", " ");
                                double individualSentiment = indicators[key];
                                string sentiment = individualSentiment > 0 ? "Bullish" : "Bearish";
                                insiderBreakdown += $"\n- {name}: {sentiment}";
                            }
                        }
                    }
                    
                    predictionModel.Notes += string.IsNullOrEmpty(predictionModel.Notes) ? 
                        insiderBreakdown : $"\n\n{insiderBreakdown}";
                }
                
                // Optionally, store weights in AggregationMethod or another property for UI
                if (weights != null)
                {
                    predictionModel.AggregationMethod = $"ML Weights: {string.Join(", ", weights)}";
                    predictionModel.FeatureWeights = weights;
                }
                return predictionModel;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to analyze {symbol} with all algorithms (ML/API)", ex.ToString());
                return new Quantra.Models.PredictionModel // Return a default/error model
                {
                    Symbol = symbol,
                    PredictedAction = "ERROR",
                    Confidence = 0.0,
                    CurrentPrice = 0.0,
                    TargetPrice = 0.0,
                    Indicators = new Dictionary<string, double>(),
                    PotentialReturn = 0,
                    PredictionDate = DateTime.Now
                };
            }
        }

        private (string action, double confidence, double targetPrice) DeterminePredictionFromIndicators(
            double currentPrice, Dictionary<string, double> indicators)
        {
            int buySignals = 0;
            int sellSignals = 0;
            int totalSignals = 0;

            // Optional: Assign weights to indicators based on perceived reliability
            var indicatorWeights = new Dictionary<string, double>
            {
                { "RSI", 1.0 },
                { "MACD", 1.2 },
                { "VWAP", 1.1 },
                { "ADX", 0.8 },
                { "CCI", 0.9 },
                { "WilliamsR", 0.8 },
                { "BullPower", 0.7 },
                { "BearPower", 0.7 },
                { "STOCH_RSI", 1.0 },
                { "STOCH_K", 1.0 },
                { "STOCH_D", 1.0 }
            };

            double weightedBuy = 0;
            double weightedSell = 0;
            double weightedTotal = 0;

            // RSI analysis
            if (indicators.TryGetValue("RSI", out double rsi))
            {
                totalSignals++;
                double w = indicatorWeights["RSI"];
                weightedTotal += w;
                if (rsi < 30) { buySignals++; weightedBuy += w; }
                else if (rsi > 70) { sellSignals++; weightedSell += w; }
            }

            // MACD analysis
            if (indicators.TryGetValue("MACD", out double macd) &&
                indicators.TryGetValue("MACD_Signal", out double macdSignal))
            {
                totalSignals++;
                double w = indicatorWeights["MACD"];
                weightedTotal += w;
                if (macd > macdSignal) { buySignals++; weightedBuy += w; }
                else if (macd < macdSignal) { sellSignals++; weightedSell += w; }
            }

            // VWAP analysis
            if (indicators.TryGetValue("VWAP", out double vwap))
            {
                totalSignals++;
                double w = indicatorWeights["VWAP"];
                weightedTotal += w;
                if (currentPrice < vwap) { buySignals++; weightedBuy += w; }
                else if (currentPrice > vwap * 1.05) { sellSignals++; weightedSell += w; }
            }

            // ADX analysis (trend strength)
            if (indicators.TryGetValue("ADX", out double adx))
            {
                double w = indicatorWeights["ADX"];
                if (adx > 25)
                {
                    if (buySignals > sellSignals) { buySignals++; weightedBuy += w; }
                    else if (sellSignals > buySignals) { sellSignals++; weightedSell += w; }
                }
            }

            // CCI analysis
            if (indicators.TryGetValue("CCI", out double cci))
            {
                totalSignals++;
                double w = indicatorWeights["CCI"];
                weightedTotal += w;
                if (cci < -100) { buySignals++; weightedBuy += w; }
                else if (cci > 100) { sellSignals++; weightedSell += w; }
            }

            // Williams %R
            if (indicators.TryGetValue("WilliamsR", out double williamsR))
            {
                totalSignals++;
                double w = indicatorWeights["WilliamsR"];
                weightedTotal += w;
                if (williamsR < -80) { buySignals++; weightedBuy += w; }
                else if (williamsR > -20) { sellSignals++; weightedSell += w; }
            }

            // Bull/Bear Power
            if (indicators.TryGetValue("BullPower", out double bullPower) &&
                indicators.TryGetValue("BearPower", out double bearPower))
            {
                totalSignals++;
                double w = indicatorWeights["BullPower"];
                weightedTotal += w;
                if (bullPower > 0 && bearPower < 0 && Math.Abs(bullPower) > Math.Abs(bearPower))
                { buySignals++; weightedBuy += w; }
                else if (bullPower < 0 || (bearPower < 0 && Math.Abs(bearPower) > Math.Abs(bullPower)))
                { sellSignals++; weightedSell += w; }
            }

            // Stochastic RSI
            if (indicators.TryGetValue("STOCH_RSI", out double stochRsi))
            {
                totalSignals++;
                double w = indicatorWeights["STOCH_RSI"];
                weightedTotal += w;
                if (stochRsi < 20) { buySignals++; weightedBuy += w; }
                else if (stochRsi > 80) { sellSignals++; weightedSell += w; }
            }

            // Stochastic Oscillator
            if (indicators.TryGetValue("STOCH_K", out double k) &&
                indicators.TryGetValue("STOCH_D", out double d))
            {
                totalSignals++;
                double w = indicatorWeights["STOCH_K"];
                weightedTotal += w;
                if (k < 20 && d < 20) { buySignals++; weightedBuy += w; }
                else if (k > 80 && d > 80) { sellSignals++; weightedSell += w; }
            }

            // --- Improved confidence calculation ---
            // Confidence is now based on weighted signal agreement and normalized by total possible weight.
            // Optionally, adjust for volatility (ATR) to reduce confidence in high-volatility environments.
            double confidence = 0.5; // Neutral

            // Volatility adjustment (optional, can be tuned)
            double volatilityPenalty = 1.0;
            if (indicators.TryGetValue("ATR", out double atrForVol) && currentPrice > 0)
            {
                double atrRatio = atrForVol / currentPrice;
                // If ATR is more than 5% of price, reduce confidence up to 20%
                if (atrRatio > 0.05)
                    volatilityPenalty = Math.Max(0.8, 1.0 - (atrRatio - 0.05));
            }

            if (weightedTotal > 0)
            {
                if (weightedBuy > weightedSell)
                {
                    // Confidence is proportional to weighted agreement, scaled and volatility-adjusted
                    confidence = 0.5 + (0.5 * (weightedBuy - weightedSell) / weightedTotal) * volatilityPenalty;
                    confidence = Math.Min(1.0, Math.Max(0.5, confidence)); // Clamp

                    // Calculate target price (based on ATR if available, otherwise use percentage)
                    double targetPrice = currentPrice;
                    if (indicators.TryGetValue("ATR", out double atr))
                    {
                        targetPrice = currentPrice + (2 * atr);
                    }
                    else
                    {
                        targetPrice = currentPrice * 1.08;
                    }

                    return ("BUY", confidence, targetPrice);
                }
                else if (weightedSell > weightedBuy)
                {
                    confidence = 0.5 + (0.5 * (weightedSell - weightedBuy) / weightedTotal) * volatilityPenalty;
                    confidence = Math.Min(1.0, Math.Max(0.5, confidence)); // Clamp

                    double targetPrice = currentPrice;
                    if (indicators.TryGetValue("ATR", out double atr))
                    {
                        targetPrice = currentPrice - (2 * atr);
                    }
                    else
                    {
                        targetPrice = currentPrice * 0.95;
                    }

                    return ("SELL", confidence, targetPrice);
                }
            }

            // No clear signal, return neutral with current price as target
            return ("HOLD", confidence, currentPrice);
        }

        // Model Training Service
        private ModelTrainingService _modelTrainingService;
        private ModelTrainingHistoryService _modelTrainingHistoryService;
        private List<string> _selectedTrainingSymbols;
        private int? _maxTrainingSymbols;

        // Initialize training service - call this in the main constructor
        private void InitializeTrainingService()
        {
            _modelTrainingService = new ModelTrainingService(_loggingService);
            
            // Initialize training history service
            var optionsBuilder = new Microsoft.EntityFrameworkCore.DbContextOptionsBuilder<Quantra.DAL.Data.QuantraDbContext>();
            optionsBuilder.UseSqlServer(Quantra.DAL.Data.ConnectionHelper.ConnectionString);
            var dbContext = new Quantra.DAL.Data.QuantraDbContext(optionsBuilder.Options);
            _modelTrainingHistoryService = new Quantra.DAL.Services.ModelTrainingHistoryService(dbContext, _loggingService);
        }

        /// <summary>
        /// Train ML model using all cached historical data from database
        /// </summary>
        private async Task TrainModelFromDatabaseAsync()
        {
            if (StatusText != null)
                StatusText.Text = "Preparing to train model from database...";

            // Disable buttons during training
            if (AnalyzeButton != null)
                AnalyzeButton.IsEnabled = false;
            
            // Note: TrainModelButton will be available after XAML compilation
            var trainButton = this.FindName("TrainModelButton") as Button;
            if (trainButton != null)
                trainButton.IsEnabled = false;

            try
            {
                // Get selected model type and architecture from UI
                string modelType = GetSelectedModelType();
                string architectureType = GetSelectedArchitectureType();
                
                // Get max symbols from UI or selected symbols
                int? maxSymbols = _maxTrainingSymbols;
                
                // If specific symbols are selected, use those instead
                if (_selectedTrainingSymbols != null && _selectedTrainingSymbols.Count > 0)
                {
                    maxSymbols = _selectedTrainingSymbols.Count;
                    if (StatusText != null)
                        StatusText.Text = $"Preparing to train on {_selectedTrainingSymbols.Count} selected symbols...";
                }
                else if (maxSymbols.HasValue)
                {
                    if (StatusText != null)
                        StatusText.Text = $"Preparing to train on up to {maxSymbols.Value} symbols...";
                }
                else
                {
                    if (StatusText != null)
                        StatusText.Text = "Preparing to train on all available symbols...";
                }
                
                // Progress callback to update UI
                Action<string> progressCallback = (message) =>
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        if (StatusText != null)
                            StatusText.Text = message;
                        
                        _loggingService?.Log("Info", $"Training: {message}");
                    });
                };

                // Start training
                var result = await _modelTrainingService.TrainModelFromDatabaseAsync(
                    modelType: modelType,
                    architectureType: architectureType,
                    maxSymbols: maxSymbols,
                    progressCallback: progressCallback
                );

                // Display results
                if (result.Success)
                {
                    // Log training session to database
                    try
                    {
                        int trainingHistoryId = await _modelTrainingHistoryService.LogTrainingSessionAsync(
                            result,
                            notes: $"Trained via UI with {result.SymbolsCount} symbols"
                        );

                        _loggingService?.Log("Info", $"Training session saved with ID: {trainingHistoryId}");
                        
                        // Save per-symbol training results if available
                        if (result.SymbolResults != null && result.SymbolResults.Count > 0)
                        {
                            await _modelTrainingHistoryService.LogSymbolResultsWithDatesAsync(trainingHistoryId, result.SymbolResults);
                            
                            _loggingService?.Log("Info", $"Saved {result.SymbolResults.Count} symbol training results");
                        }
                    }
                    catch (Exception logEx)
                    {
                        _loggingService?.LogErrorWithContext(logEx, "Failed to log training session to database");
                    }

                    var message = $"Model Training Complete!\n\n" +
                                 $"Model Type: {result.ModelType}\n" +
                                 $"Architecture: {result.ArchitectureType}\n" +
                                 $"Symbols Used: {result.SymbolsCount}\n" +
                                 $"Training Samples: {result.TrainingSamples}\n" +
                                 $"Test Samples: {result.TestSamples}\n" +
                                 $"Training Time: {result.TrainingTimeSeconds:F1}s\n\n" +
                                 $"Performance Metrics:\n" +
                                 $"  MAE: {result.Performance?.Mae:F6}\n" +
                                 $"  RMSE: {result.Performance?.Rmse:F6}\n" +
                                 $"  R� Score: {result.Performance?.R2Score:F4}";

                    MessageBox.Show(message, "Training Successful", MessageBoxButton.OK, MessageBoxImage.Information);

                    if (StatusText != null)
                        StatusText.Text = $"Model trained successfully with {result.SymbolsCount} symbols";
                }
                else
                {
                    MessageBox.Show($"Training failed: {result.Error}", "Training Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    
                    if (StatusText != null)
                        StatusText.Text = "Model training failed";
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error during model training");
                MessageBox.Show($"Training error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                
                if (StatusText != null)
                    StatusText.Text = "Training error occurred";
            }
            finally
            {
                // Re-enable buttons
                if (AnalyzeButton != null)
                    AnalyzeButton.IsEnabled = true;
                    
                var trainBtn = this.FindName("TrainModelButton") as Button;
                if (trainBtn != null)
                    trainBtn.IsEnabled = true;
            }
        }

        // Helper methods to get UI selections
        private string GetSelectedModelType()
        {
            var comboBox = this.FindName("ModelTypeComboBox") as ComboBox;
            if (comboBox?.SelectedItem is ComboBoxItem modelItem)
            {
                return modelItem.Tag?.ToString() ?? "auto";
            }
            return "auto";
        }

        private string GetSelectedArchitectureType()
        {
            var comboBox = this.FindName("ArchitectureComboBox") as ComboBox;
            if (comboBox?.SelectedItem is ComboBoxItem archItem)
            {
                return archItem.Tag?.ToString() ?? "lstm";
            }
            return "lstm";
        }

        // Event handler for Train Model button
        private async void TrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            await TrainModelFromDatabaseAsync();
        }
        
        /// <summary>
        /// Handle the Select Training Symbols button click
        /// </summary>
        private void SelectTrainingSymbolsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Use the Stock Configuration Manager to select symbols
                var stockConfigService = new StockConfigurationService(_loggingService);
                var selectedSymbols = Quantra.Views.StockExplorer.StockConfigurationManagerWindow.ShowAndGetSymbols(
                    stockConfigService,
                    Window.GetWindow(this)
                );

                if (selectedSymbols != null && selectedSymbols.Count > 0)
                {
                    _selectedTrainingSymbols = selectedSymbols;

                    // Update the UI to show how many symbols are selected
                    var symbolsCountTextBlock = this.FindName("TrainingSymbolsCountTextBlock") as TextBlock;
                    if (symbolsCountTextBlock != null)
                    {
                        symbolsCountTextBlock.Text = $"{selectedSymbols.Count} symbols selected";
                        symbolsCountTextBlock.Visibility = Visibility.Visible;
                    }

                    if (StatusText != null)
                        StatusText.Text = $"Selected {selectedSymbols.Count} symbols for training";
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error selecting training symbols");
                MessageBox.Show($"Error selecting symbols:\n\n{ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Handle changes to the max symbols TextBox
        /// </summary>
        private void MaxSymbolsTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (sender is TextBox textBox)
            {
                if (string.IsNullOrWhiteSpace(textBox.Text))
                {
                    _maxTrainingSymbols = null;
                }
                else if (int.TryParse(textBox.Text, out int maxSymbols) && maxSymbols > 0)
                {
                    _maxTrainingSymbols = maxSymbols;
                }
                else
                {
                    // Invalid input - reset
                    _maxTrainingSymbols = null;
                }
            }
        }

        /// <summary>
        /// Clear selected training symbols
        /// </summary>
        private void ClearTrainingSymbolsButton_Click(object sender, RoutedEventArgs e)
        {
            _selectedTrainingSymbols = null;

            var symbolsCountTextBlock = this.FindName("TrainingSymbolsCountTextBlock") as TextBlock;
            if (symbolsCountTextBlock != null)
            {
                symbolsCountTextBlock.Text = "";
                symbolsCountTextBlock.Visibility = Visibility.Collapsed;
            }

            if (StatusText != null)
                StatusText.Text = "Cleared selected training symbols";
        }
    }
}
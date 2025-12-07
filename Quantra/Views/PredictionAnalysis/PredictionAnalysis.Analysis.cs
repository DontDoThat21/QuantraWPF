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
        // TFT Prediction Constants
        private const string TFT_ARCHITECTURE_TYPE = "tft";
        private const int TFT_DEFAULT_LOOKBACK_DAYS = 60;
        private const int TFT_DEFAULT_FUTURE_HORIZON = 30;
        private const int TFT_HISTORICAL_VISUALIZATION_DAYS = 30;
        
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
            // Cache UI values on the UI thread before any await calls
            // This prevents cross-thread UI access exceptions
            string cachedModelType = GetSelectedModelType();
            string cachedArchitectureType = GetSelectedArchitectureType();
            
            // Check if TFT multi-horizon prediction is requested
            bool useTFT = cachedArchitectureType.ToLower() == "tft";
            
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
                
                // Store historical data for TFT prediction (needs 60+ days)
                List<HistoricalPrice> historicalDataForTFT = null;

                // Fetch technical indicators from Alpha Vantage API
                // These indicators are critical for ML model prediction accuracy
                try
                {
                    _loggingService?.Log("Info", $"Fetching technical indicators for {symbol}...");
                    
                    // RSI (Relative Strength Index) - standard 14-period
                    try
                    {
                        double rsi = await _alphaVantageService.GetRSI(symbol, "daily");
                        if (!double.IsNaN(rsi) && rsi > 0)
                            indicators["RSI"] = rsi;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get RSI for {symbol}: {ex.Message}");
                    }

                    // Stochastic Oscillator (%K and %D)
                    try
                    {
                        var (stochK, stochD) = await _alphaVantageService.GetSTOCH(symbol, "daily");
                        if (!double.IsNaN(stochK))
                            indicators["STOCH_K"] = stochK;
                        if (!double.IsNaN(stochD))
                            indicators["STOCH_D"] = stochD;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get Stochastic for {symbol}: {ex.Message}");
                    }

                    // MACD (Moving Average Convergence Divergence) with Histogram
                    try
                    {
                        var (macd, macdSignal, macdHist) = await _alphaVantageService.GetMACD(symbol, "daily");
                        if (!double.IsNaN(macd))
                            indicators["MACD"] = macd;
                        if (!double.IsNaN(macdSignal))
                            indicators["MACD_Signal"] = macdSignal;
                        if (!double.IsNaN(macdHist))
                            indicators["MACD_Hist"] = macdHist;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get MACD for {symbol}: {ex.Message}");
                    }

                    // ATR (Average True Range) - volatility indicator
                    try
                    {
                        double atr = await _alphaVantageService.GetATR(symbol, "daily");
                        if (!double.IsNaN(atr) && atr > 0)
                            indicators["ATR"] = atr;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get ATR for {symbol}: {ex.Message}");
                    }

                    // VWAP (Volume-Weighted Average Price) - typically intraday, but calculated from daily data for consistency
                    try
                    {
                        double vwap = await _alphaVantageService.GetVWAP(symbol, "daily");
                        if (!double.IsNaN(vwap) && vwap > 0)
                            indicators["VWAP"] = vwap;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get VWAP for {symbol}: {ex.Message}");
                    }

                    // ADX (Average Directional Index) - trend strength
                    try
                    {
                        double adx = await _alphaVantageService.GetLatestADX(symbol, "daily");
                        if (!double.IsNaN(adx) && adx > 0)
                            indicators["ADX"] = adx;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get ADX for {symbol}: {ex.Message}");
                    }

                    // CCI (Commodity Channel Index)
                    try
                    {
                        double cci = await _alphaVantageService.GetCCI(symbol, "daily");
                        if (!double.IsNaN(cci))
                            indicators["CCI"] = cci;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get CCI for {symbol}: {ex.Message}");
                    }

                    // MFI (Money Flow Index)
                    try
                    {
                        double mfi = await _alphaVantageService.GetMFI(symbol, "daily");
                        if (!double.IsNaN(mfi) && mfi > 0)
                            indicators["MFI"] = mfi;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get MFI for {symbol}: {ex.Message}");
                    }

                    // OBV (On-Balance Volume)
                    try
                    {
                        double obv = await _alphaVantageService.GetOBV(symbol, "daily");
                        if (!double.IsNaN(obv))
                            indicators["OBV"] = obv;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get OBV for {symbol}: {ex.Message}");
                    }

                    // Ultimate Oscillator
                    try
                    {
                        double ultimateOsc = await _alphaVantageService.GetUltimateOscillator(symbol, "daily");
                        if (!double.IsNaN(ultimateOsc))
                            indicators["UltimateOscillator"] = ultimateOsc;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get Ultimate Oscillator for {symbol}: {ex.Message}");
                    }

                    // Momentum Score
                    try
                    {
                        double momentum = await _alphaVantageService.GetMomentumScore(symbol, "daily");
                        if (!double.IsNaN(momentum))
                            indicators["Momentum"] = momentum;
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get Momentum for {symbol}: {ex.Message}");
                    }

                    // Fetch historical data for calculating lagged features, SMA, ROC, and Bollinger Bands
                    try
                    {
                        var historicalData = await _alphaVantageService.GetDailyData(symbol, "compact");
                        if (historicalData != null && historicalData.Count > 0)
                        {
                            // Sort by date descending (most recent first)
                            historicalData = historicalData.OrderByDescending(h => h.Date).ToList();
                            
                            // Store for TFT (need at least 60 days)
                            historicalDataForTFT = historicalData;

                            // Lagged Features: Close(t-1), Close(t-2), Volume(t-1), Volume(t-2)
                            if (historicalData.Count >= 3)
                            {
                                indicators["Close_t0"] = historicalData[0].Close;  // Current close
                                indicators["Close_t1"] = historicalData[1].Close;  // Previous close (lag 1)
                                indicators["Close_t2"] = historicalData[2].Close;  // Two days ago (lag 2)
                                indicators["Volume_t0"] = historicalData[0].Volume;
                                indicators["Volume_t1"] = historicalData[1].Volume;
                                indicators["Volume_t2"] = historicalData[2].Volume;
                                
                                // Open, High, Low for current day
                                indicators["Open"] = historicalData[0].Open;
                                indicators["High"] = historicalData[0].High;
                                indicators["Low"] = historicalData[0].Low;
                                indicators["Close"] = historicalData[0].Close;
                                indicators["Volume"] = historicalData[0].Volume;
                            }

                            // Calculate SMA (Simple Moving Averages) for 7, 14, 30 days
                            var closePrices = historicalData.Select(h => h.Close).ToList();
                            
                            if (closePrices.Count >= 7)
                            {
                                double sma7 = closePrices.Take(7).Average();
                                indicators["SMA_7"] = sma7;
                            }
                            if (closePrices.Count >= 14)
                            {
                                double sma14 = closePrices.Take(14).Average();
                                indicators["SMA_14"] = sma14;
                            }
                            if (closePrices.Count >= 30)
                            {
                                double sma30 = closePrices.Take(30).Average();
                                indicators["SMA_30"] = sma30;
                            }
                            if (closePrices.Count >= 50)
                            {
                                double sma50 = closePrices.Take(50).Average();
                                indicators["SMA_50"] = sma50;
                            }

                            // Rate of Change (ROC) - 10-day and 20-day
                            if (closePrices.Count >= 11)
                            {
                                double roc10 = ((closePrices[0] - closePrices[10]) / closePrices[10]) * 100;
                                indicators["ROC_10"] = roc10;
                            }
                            if (closePrices.Count >= 21)
                            {
                                double roc20 = ((closePrices[0] - closePrices[20]) / closePrices[20]) * 100;
                                indicators["ROC_20"] = roc20;
                            }

                            // Bollinger Bands (20-day with 2 std dev)
                            if (closePrices.Count >= 20)
                            {
                                var last20Prices = closePrices.Take(20).ToList();
                                double bbMiddle = last20Prices.Average();
                                double bbStdDev = Math.Sqrt(last20Prices.Sum(p => Math.Pow(p - bbMiddle, 2)) / last20Prices.Count);
                                double bbUpper = bbMiddle + (2 * bbStdDev);
                                double bbLower = bbMiddle - (2 * bbStdDev);
                                double bbWidth = (bbUpper - bbLower) / bbMiddle;
                                double bbPct = (closePrices[0] - bbLower) / (bbUpper - bbLower);

                                indicators["BB_Middle"] = bbMiddle;
                                indicators["BB_Upper"] = bbUpper;
                                indicators["BB_Lower"] = bbLower;
                                indicators["BB_Width"] = bbWidth;
                                indicators["BB_Pct"] = bbPct;  // Where price is within the bands (0-1)
                            }

                            // Rolling Volatility (standard deviation of returns)
                            if (closePrices.Count >= 21)
                            {
                                var returns = new List<double>();
                                for (int i = 0; i < 20; i++)
                                {
                                    double ret = (closePrices[i] - closePrices[i + 1]) / closePrices[i + 1];
                                    returns.Add(ret);
                                }
                                double returnsAvg = returns.Average();
                                double volatility = Math.Sqrt(returns.Sum(r => Math.Pow(r - returnsAvg, 2)) / returns.Count) * Math.Sqrt(252);
                                indicators["Volatility_20"] = volatility;
                            }

                            // Log returns (most recent)
                            if (closePrices.Count >= 2)
                            {
                                double logReturn = Math.Log(closePrices[0] / closePrices[1]);
                                indicators["LogReturn_1"] = logReturn;
                            }
                            if (closePrices.Count >= 6)
                            {
                                double logReturn5 = Math.Log(closePrices[0] / closePrices[5]);
                                indicators["LogReturn_5"] = logReturn5;
                            }

                            // Rolling Min/Max (20-day)
                            if (closePrices.Count >= 20)
                            {
                                var last20 = closePrices.Take(20).ToList();
                                indicators["Min_20"] = last20.Min();
                                indicators["Max_20"] = last20.Max();
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Failed to get historical data for lagged features for {symbol}: {ex.Message}");
                    }

                    _loggingService?.Log("Info", $"Fetched {indicators.Count} technical indicators for {symbol}");
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Error fetching technical indicators for {symbol}: {ex.Message}");
                }

                // Static metadata features for TFT model (Sector, MarketCap, Exchange)
                // These are unchanging characteristics of stocks critical for TFT model
                try
                {
                    var overview = await _alphaVantageService.GetCompanyOverviewAsync(symbol);
                    if (overview != null)
                    {
                        // Convert categorical data to numerical codes for ML model
                        indicators["Sector"] = _alphaVantageService.GetSectorCode(overview.Sector);
                        indicators["MarketCapCategory"] = _alphaVantageService.GetMarketCapCategory(overview.MarketCapitalization);
                        indicators["Exchange"] = _alphaVantageService.GetExchangeCode(overview.Exchange);
                        
                        // Also store the raw market cap value if available (useful for normalization)
                        if (overview.MarketCapitalization.HasValue && overview.MarketCapitalization.Value > 0)
                        {
                            // Store as billions for reasonable scale
                            indicators["MarketCapBillions"] = (double)(overview.MarketCapitalization.Value / 1_000_000_000m);
                        }
                        
                        // Store Beta as additional static metadata (risk measure)
                        if (overview.Beta.HasValue)
                        {
                            indicators["Beta"] = (double)overview.Beta.Value;
                        }

                        _loggingService?.Log("Info", $"Static metadata for {symbol}: Sector={overview.Sector} ({indicators["Sector"]}), " +
                            $"MarketCap={overview.FormattedMarketCap} (Category={indicators["MarketCapCategory"]}), " +
                            $"Exchange={overview.Exchange} ({indicators["Exchange"]})");
                    }
                    else
                    {
                        _loggingService?.Log("Warning", $"Could not retrieve company overview for {symbol}, using default static metadata values");
                        // Set default values for missing metadata (-1 indicates unknown)
                        indicators["Sector"] = -1;
                        indicators["MarketCapCategory"] = -1;
                        indicators["Exchange"] = -1;
                    }
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Failed to get company overview for {symbol}: {ex.Message}");
                    // Set default values on error
                    indicators["Sector"] = -1;
                    indicators["MarketCapCategory"] = -1;
                    indicators["Exchange"] = -1;
                }

                // Calendar features (known future inputs for TFT model)
                // These are features we know ahead of time like calendar dates
                try
                {
                    DateTime now = DateTime.UtcNow;
                    
                    // Basic calendar features
                    indicators["DayOfWeek"] = (double)now.DayOfWeek; // 0=Sunday, 6=Saturday
                    indicators["Month"] = now.Month; // 1-12
                    indicators["Quarter"] = ((now.Month - 1) / 3) + 1; // 1-4
                    indicators["DayOfYear"] = now.DayOfYear; // 1-365/366
                    
                    // Month-end and quarter-end indicators (important for financial reporting)
                    int daysInMonth = DateTime.DaysInMonth(now.Year, now.Month);
                    indicators["IsMonthEnd"] = (now.Day >= daysInMonth - 2) ? 1.0 : 0.0;
                    indicators["IsQuarterEnd"] = (now.Month % 3 == 0 && now.Day >= daysInMonth - 2) ? 1.0 : 0.0;

                    // Market hours indicators (9:30 AM - 4:00 PM ET)
                    // Use "America/New_York" for cross-platform compatibility (Linux/Windows)
                    TimeZoneInfo easternZone;
                    try
                    {
                        easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
                    }
                    catch
                    {
                        // Fallback for Windows
                        easternZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                    }
                    
                    DateTime easternTime = TimeZoneInfo.ConvertTimeFromUtc(now, easternZone);
                    int hour = easternTime.Hour;
                    int minute = easternTime.Minute;
                    double minuteOfDay = hour * 60 + minute;
                    
                    // Pre-market: before 9:30 AM ET
                    indicators["IsPreMarket"] = (hour < 9 || (hour == 9 && minute < 30)) ? 1.0 : 0.0;
                    // Regular hours: 9:30 AM - 4:00 PM ET
                    indicators["IsRegularHours"] = (minuteOfDay >= 570 && minuteOfDay < 960) ? 1.0 : 0.0; // 9:30=570min, 16:00=960min
                    // After hours: 4:00 PM ET and later
                    indicators["IsAfterHours"] = (hour >= 16) ? 1.0 : 0.0;
                    
                    _loggingService?.Log("Debug", $"Calendar features for {symbol}: DayOfWeek={indicators["DayOfWeek"]}, " +
                        $"Month={indicators["Month"]}, Quarter={indicators["Quarter"]}, " +
                        $"IsRegularHours={indicators["IsRegularHours"]}");
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Failed to calculate calendar features: {ex.Message}");
                    // Set default values on error
                    indicators["DayOfWeek"] = (double)DateTime.UtcNow.DayOfWeek;
                    indicators["Month"] = DateTime.UtcNow.Month;
                    indicators["Quarter"] = ((DateTime.UtcNow.Month - 1) / 3) + 1;
                    indicators["DayOfYear"] = DateTime.UtcNow.DayOfYear;
                    indicators["IsMonthEnd"] = 0.0;
                    indicators["IsQuarterEnd"] = 0.0;
                    indicators["IsPreMarket"] = 0.0;
                    indicators["IsRegularHours"] = 0.0;
                    indicators["IsAfterHours"] = 0.0;
                }

                // Earnings calendar features (known future inputs for TFT model)
                try
                {
                    var earningsService = new EarningsCalendarService(_alphaVantageService, _loggingService);

                    var nextEarningsDate = await earningsService.GetNextEarningsDateAsync(symbol);
                    var lastEarningsDate = await earningsService.GetLastEarningsDateAsync(symbol);

                    if (nextEarningsDate.HasValue)
                    {
                        int daysToEarnings = earningsService.GetTradingDaysToEarnings(nextEarningsDate.Value);
                        indicators["DaysToEarnings"] = daysToEarnings;
                        indicators["IsEarningsWeek"] = (daysToEarnings <= 5 && daysToEarnings >= 0) ? 1.0 : 0.0;
                        
                        _loggingService?.Log("Debug", $"Earnings for {symbol}: Next={nextEarningsDate.Value:yyyy-MM-dd}, " +
                            $"DaysTo={daysToEarnings}, IsEarningsWeek={indicators["IsEarningsWeek"]}");
                    }
                    else
                    {
                        indicators["DaysToEarnings"] = 999; // Unknown/far future
                        indicators["IsEarningsWeek"] = 0.0;
                    }

                    if (lastEarningsDate.HasValue)
                    {
                        int daysSinceEarnings = earningsService.GetTradingDaysSinceEarnings(lastEarningsDate.Value);
                        indicators["DaysSinceEarnings"] = daysSinceEarnings;
                    }
                    else
                    {
                        indicators["DaysSinceEarnings"] = 999; // Unknown
                    }
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Failed to get earnings calendar for {symbol}: {ex.Message}");
                    indicators["DaysToEarnings"] = 999;
                    indicators["DaysSinceEarnings"] = 999;
                    indicators["IsEarningsWeek"] = 0.0;
                }

                // Market context features for TFT model
                // These provide broader market conditions (bull/bear market, volatility regime, interest rate environment)
                try
                {
                    var marketContextService = new MarketContextService(_alphaVantageService, _loggingService);

                    // S&P 500 context
                    var (sp500Price, sp500Return) = await marketContextService.GetSP500DataAsync();
                    indicators["SP500_Price"] = sp500Price;
                    indicators["SP500_Return"] = sp500Return;
                    indicators["SP500_Direction"] = sp500Return > 0 ? 1.0 : -1.0;

                    // VIX (volatility regime)
                    double vix = await marketContextService.GetVIXAsync();
                    indicators["VIX"] = vix;
                    indicators["VolatilityRegime"] = vix switch
                    {
                        < 15 => 0.0,  // Low volatility
                        < 20 => 1.0,  // Normal
                        < 30 => 2.0,  // Elevated
                        _ => 3.0      // High volatility
                    };

                    // Treasury yield (interest rate environment)
                    double treasuryYield = await marketContextService.GetTreasuryYield10YAsync();
                    indicators["TreasuryYield_10Y"] = treasuryYield;

                    // Sector ETF context (if sector is known)
                    if (indicators.ContainsKey("Sector") && indicators["Sector"] >= 0)
                    {
                        string sector = marketContextService.GetSectorName((int)indicators["Sector"]);
                        var (sectorETFPrice, sectorETFReturn) = await marketContextService.GetSectorETFDataAsync(sector);
                        indicators["SectorETF_Price"] = sectorETFPrice;
                        indicators["SectorETF_Return"] = sectorETFReturn;

                        // Relative strength vs sector
                        if (currentPrice > 0 && sectorETFPrice > 0)
                        {
                            indicators["RelativeStrengthVsSector"] = currentPrice / sectorETFPrice;
                        }
                    }

                    // Market breadth
                    double marketBreadth = await marketContextService.GetMarketBreadthAsync();
                    indicators["MarketBreadth"] = marketBreadth;
                    indicators["IsBullishBreadth"] = marketBreadth > 1.0 ? 1.0 : 0.0;

                    _loggingService?.Log("Info", $"Market context: SPY={sp500Price:F2} ({sp500Return:P2}), VIX={vix:F2}, 10Y={treasuryYield:F2}%");
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Failed to get market context: {ex.Message}");
                }

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
                    // Use TFT if selected, otherwise use standard prediction
                    if (useTFT)
                    {
                        // Prepare historical sequence for TFT (convert HistoricalPrice to Dictionary format)
                        List<Dictionary<string, double>> historicalSequence = null;
                        
                        if (historicalDataForTFT != null && historicalDataForTFT.Count >= TFT_DEFAULT_LOOKBACK_DAYS)
                        {
                            // Take the last 60 days (or more) for TFT
                            // TFT needs data in chronological order (oldest first)
                            var tftHistoricalData = historicalDataForTFT
                                .OrderBy(h => h.Date)  // Oldest first for TFT
                                .Take(TFT_DEFAULT_LOOKBACK_DAYS)
                                .ToList();
                            
                            historicalSequence = tftHistoricalData.Select(h => new Dictionary<string, double>
                            {
                                ["open"] = h.Open,
                                ["high"] = h.High,
                                ["low"] = h.Low,
                                ["close"] = h.Close,
                                ["volume"] = h.Volume
                            }).ToList();
                            
                            _loggingService?.Log("Info", $"Prepared {historicalSequence.Count} days of historical data for TFT prediction");
                        }
                        else
                        {
                            _loggingService?.Log("Warning", $"Insufficient historical data for TFT: {historicalDataForTFT?.Count ?? 0} days (need {TFT_DEFAULT_LOOKBACK_DAYS})");
                        }
                        
                        // Call TFT multi-horizon prediction
                        var tftResult = await Quantra.Models.PythonStockPredictor.PredictWithTFTAsync(
                            indicators,
                            symbol,
                            historicalSequence,  // Pass actual historical data
                            new List<int> { 1, 3, 5, 10 }  // horizons from UI checkboxes
                        );
                        
                        if (tftResult == null || !tftResult.Success)
                            throw new Exception($"TFT prediction failed: {tftResult?.Error ?? "Unknown error"}");

                        action = tftResult.Action;
                        confidence = tftResult.Confidence;
                        targetPrice = tftResult.TargetPrice;
                        weights = tftResult.FeatureWeights ?? new Dictionary<string, double>();
                        
                        // Store TFT-specific data for later visualization
                        var tftPrediction = new Quantra.Models.PredictionModel
                        {
                            Symbol = symbol,
                            PredictedAction = action,
                            Confidence = confidence,
                            CurrentPrice = currentPrice,
                            TargetPrice = targetPrice,
                            Indicators = indicators,
                            PotentialReturn = (targetPrice - currentPrice) / currentPrice,
                            PredictionDate = DateTime.Now,
                            ModelType = "tft",
                            ArchitectureType = "tft",
                            Notes = $"Multi-horizon TFT prediction with uncertainty quantification",
                            FeatureWeights = weights
                        };
                        
                        // Save TFT prediction with multi-horizon data to database
                        await SaveTFTPredictionToDatabase(tftPrediction, tftResult);
                        
                        // Update UI with TFT multi-horizon data
                        await UpdateTFTVisualization(tftResult);
                        
                        _loggingService?.Log("Info", $"TFT prediction for {symbol}: {action} with {confidence:P0} confidence");
                    }
                    else
                    {
                        // Standard single-point prediction
                        var result = await Quantra.Models.PythonStockPredictor.PredictAsync(indicators);
                        if (result == null)
                            throw new Exception("Failed to get prediction result from trained ML model");

                        action = result.Action;
                        confidence = result.Confidence;
                        targetPrice = result.TargetPrice;
                        weights = result.FeatureWeights;
                        
                        //DatabaseMonolith.Log("Info", $"ML prediction for {symbol}: {action} with {confidence:P0} confidence (model-based)");
                    }
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

                // Use cached model type and architecture values (already retrieved at method start)
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
                    ModelType = cachedModelType,
                    ArchitectureType = cachedArchitectureType,
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
        private TrainingConfigurationService _trainingConfigService;
        private Quantra.DAL.Models.TrainingConfiguration _currentTrainingConfig;
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

            // Initialize training configuration service and load last used configuration
            _trainingConfigService = new TrainingConfigurationService(_loggingService);
            _currentTrainingConfig = _trainingConfigService.GetLastUsedOrDefault();

            _loggingService?.Log("Info", $"Training configuration loaded: {_currentTrainingConfig.ConfigurationName}");
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
                // Update configuration with UI selections (model type, architecture, max symbols)
                _currentTrainingConfig.ModelType = GetSelectedModelType();
                _currentTrainingConfig.ArchitectureType = GetSelectedArchitectureType();

                // Set max symbols from UI or selected symbols
                if (_selectedTrainingSymbols != null && _selectedTrainingSymbols.Count > 0)
                {
                    _currentTrainingConfig.MaxSymbols = _selectedTrainingSymbols.Count;
                    _currentTrainingConfig.SelectedSymbols = _selectedTrainingSymbols;
                    if (StatusText != null)
                        StatusText.Text = $"Preparing to train on {_selectedTrainingSymbols.Count} selected symbols...";
                }
                else if (_maxTrainingSymbols.HasValue)
                {
                    _currentTrainingConfig.MaxSymbols = _maxTrainingSymbols;
                    if (StatusText != null)
                        StatusText.Text = $"Preparing to train on up to {_maxTrainingSymbols.Value} symbols...";
                }
                else
                {
                    _currentTrainingConfig.MaxSymbols = null;
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

                // Log the configuration being used
                _loggingService?.Log("Info", $"Starting training with configuration: {_currentTrainingConfig.ConfigurationName}");
                _loggingService?.Log("Info", $"  Epochs: {_currentTrainingConfig.Epochs}, Batch: {_currentTrainingConfig.BatchSize}, LR: {_currentTrainingConfig.LearningRate}");

                // Start training with full configuration
                var result = await _modelTrainingService.TrainModelFromDatabaseAsync(
                    config: _currentTrainingConfig,
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
        // NOTE: These methods MUST be called from the UI thread
        // For async methods, cache the values before any await calls
        private string GetSelectedModelType()
        {
            // Ensure we're on the UI thread
            if (!Dispatcher.CheckAccess())
            {
                return Dispatcher.Invoke(() => GetSelectedModelType());
            }
            
            var comboBox = this.FindName("ModelTypeComboBox") as ComboBox;
            if (comboBox?.SelectedItem is ComboBoxItem modelItem)
            {
                return modelItem.Tag?.ToString() ?? "auto";
            }
            return "auto";
        }

        private string GetSelectedArchitectureType()
        {
            // Ensure we're on the UI thread
            if (!Dispatcher.CheckAccess())
            {
                return Dispatcher.Invoke(() => GetSelectedArchitectureType());
            }
            
            var comboBox = this.FindName("ArchitectureComboBox") as ComboBox;
            if (comboBox?.SelectedItem is ComboBoxItem archItem)
            {
                return archItem.Tag?.ToString() ?? "lstm";
            }
            return "lstm";
        }

        // Event handler for Configure Training button
        private void ConfigureTrainingButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var window = new Quantra.Views.PredictionAnalysis.TrainingConfigurationWindow(
                    _currentTrainingConfig,
                    _loggingService);

                window.Owner = Window.GetWindow(this);

                if (window.ShowDialog() == true)
                {
                    _currentTrainingConfig = window.Configuration;

                    if (StatusText != null)
                        StatusText.Text = $"Training configuration: {_currentTrainingConfig.ConfigurationName} " +
                                         $"(Epochs: {_currentTrainingConfig.Epochs}, " +
                                         $"Batch: {_currentTrainingConfig.BatchSize}, " +
                                         $"LR: {_currentTrainingConfig.LearningRate})";

                    _loggingService?.Log("Info", $"Training configuration updated: {_currentTrainingConfig.ConfigurationName}");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error opening training configuration window");
                MessageBox.Show($"Error opening configuration window: {ex.Message}",
                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
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
        
        /// <summary>
        /// Save TFT prediction with multi-horizon data to database
        /// </summary>
        private async Task SaveTFTPredictionToDatabase(Quantra.Models.PredictionModel prediction, 
            Quantra.DAL.Models.TFTPredictionResult tftResult)
        {
            try
            {
                // Use PredictionService to save TFT prediction
                var optionsBuilder = new Microsoft.EntityFrameworkCore.DbContextOptionsBuilder<Quantra.DAL.Data.QuantraDbContext>();
                optionsBuilder.UseSqlServer(Quantra.DAL.Data.ConnectionHelper.ConnectionString);
                var dbContext = new Quantra.DAL.Data.QuantraDbContext(optionsBuilder.Options);
                var predictionService = new Quantra.DAL.Services.PredictionService(dbContext);
                
                int predictionId = await predictionService.SaveTFTPredictionAsync(prediction, tftResult);
                
                _loggingService?.Log("Info", $"Saved TFT prediction ID {predictionId} for {prediction.Symbol}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to save TFT prediction for {prediction.Symbol}");
            }
        }
        
        /// <summary>
        /// Update UI with TFT multi-horizon visualization data
        /// </summary>
        private async Task UpdateTFTVisualization(Quantra.DAL.Models.TFTPredictionResult tftResult)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    // Update multi-horizon chart with actual TFT data
                    if (tftResult.Horizons != null && tftResult.Horizons.Count > 0)
                    {
                        // Clear existing chart data
                        var historicalPrices = this.DataContext?.GetType().GetProperty("HistoricalPrices")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                        var predictedPrices = this.DataContext?.GetType().GetProperty("PredictedPrices")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                        var upperBandPrices = this.DataContext?.GetType().GetProperty("UpperBandPrices")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                        var lowerBandPrices = this.DataContext?.GetType().GetProperty("LowerBandPrices")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                        var dateLabels = this.DataContext?.GetType().GetProperty("DateLabels")?.GetValue(this.DataContext) as List<string>;
                        
                        historicalPrices?.Clear();
                        predictedPrices?.Clear();
                        upperBandPrices?.Clear();
                        lowerBandPrices?.Clear();
                        dateLabels?.Clear();
                        
                        // Add historical data (last 30 days from current price)
                        double currentPrice = tftResult.CurrentPrice;
                        for (int i = 30; i > 0; i--)
                        {
                            // Simulate historical prices - in production, load from cache
                            double historicalPrice = currentPrice * (1.0 - (i * 0.001));
                            historicalPrices?.Add(historicalPrice);
                            dateLabels?.Add($"-{i}d");
                        }
                        
                        // Add current price as connection point
                        historicalPrices?.Add(currentPrice);
                        predictedPrices?.Add(currentPrice);
                        upperBandPrices?.Add(currentPrice);
                        lowerBandPrices?.Add(currentPrice);
                        dateLabels?.Add("Today");
                        
                        // Add future predictions from TFT horizons
                        foreach (var horizonKey in tftResult.Horizons.Keys.OrderBy(k => int.Parse(k.Replace("d", ""))))
                        {
                            var horizonData = tftResult.Horizons[horizonKey];
                            predictedPrices?.Add(horizonData.MedianPrice);
                            upperBandPrices?.Add(horizonData.UpperBound);
                            lowerBandPrices?.Add(horizonData.LowerBound);
                            dateLabels?.Add($"+{horizonKey}");
                        }
                        
                        // Update attention weights chart
                        if (tftResult.TemporalAttention != null && tftResult.TemporalAttention.Count > 0)
                        {
                            var attentionWeights = this.DataContext?.GetType().GetProperty("AttentionWeights")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                            var attentionLabels = this.DataContext?.GetType().GetProperty("AttentionLabels")?.GetValue(this.DataContext) as List<string>;
                            
                            attentionWeights?.Clear();
                            attentionLabels?.Clear();
                            
                            // Add temporal attention weights (which past days matter most)
                            foreach (var kvp in tftResult.TemporalAttention.OrderBy(x => x.Key))
                            {
                                attentionWeights?.Add(kvp.Value);
                                attentionLabels?.Add($"{kvp.Key}d");
                            }
                        }
                        
                        // Update feature importance chart
                        if (tftResult.FeatureWeights != null && tftResult.FeatureWeights.Count > 0)
                        {
                            var featureImportances = this.DataContext?.GetType().GetProperty("FeatureImportances")?.GetValue(this.DataContext) as LiveCharts.ChartValues<double>;
                            var featureNames = this.DataContext?.GetType().GetProperty("FeatureNames")?.GetValue(this.DataContext) as List<string>;
                            
                            featureImportances?.Clear();
                            featureNames?.Clear();
                            
                            // Add top 10 features by importance
                            foreach (var kvp in tftResult.FeatureWeights.OrderByDescending(x => Math.Abs(x.Value)).Take(10))
                            {
                                featureImportances?.Add(kvp.Value);
                                featureNames?.Add(kvp.Key);
                            }
                        }
                    }
                    
                    _loggingService?.Log("Info", $"Updated TFT visualization for {tftResult.Symbol}");
                }
                catch (Exception ex)
                {
                    _loggingService?.LogErrorWithContext(ex, "Failed to update TFT visualization");
                }
            }, System.Windows.Threading.DispatcherPriority.Normal);
        }
    }
}
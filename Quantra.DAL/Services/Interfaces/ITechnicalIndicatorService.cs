using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface ITechnicalIndicatorService
    {
        Task<Dictionary<string, double>> CalculateIndicators(string symbol, string timeframe);
        Task<bool> ValidateIndicators(Dictionary<string, double> indicators, string tradingAction);
        Task<double> GetTradingSignal(Dictionary<string, double> indicators);
        Task<Dictionary<string, double>> GetIndicatorsForPrediction(string symbol, string timeframe);
        Task<Dictionary<string, Dictionary<string, double>>> GetIndicatorsForPredictionBatchAsync(List<string> symbols, string timeframe = "5min");
        Task<Dictionary<string, double>> GetAlgorithmicTradingSignals(string symbol);
        Task<double> GetRSI(string symbol, string interval = "1day");
        Task<double> GetADX(string symbol, string interval = "1day");
        Task<double> GetATR(string symbol, string interval = "1day");
        Task<double> GetMomentum(string symbol, string interval = "1day");
        Task<(double K, double D)> GetStochastic(string symbol, string interval = "1day");
        // Preserve our OBV implementation
        Task<double> GetOBV(string symbol, string interval = "1day");
        // Include the MFI implementation from main branch
        Task<double> GetMFI(string symbol, string interval = "1day");
        Task<double> GetParabolicSAR(string symbol, string interval = "1day");
        
        // Additional technical indicators
        Task<(double macd, double signal)> GetMACD(string symbol, string timeframe);
        Task<double> GetVWAP(string symbol, string timeframe);
        Task<double> GetSTOCHRSI(string symbol, string timeframe);
        Task<double> GetROC(string symbol, string timeframe);
        Task<(double high, double low)> GetHighsLows(string symbol, string timeframe);
        Task<(double bullPower, double bearPower)> GetBullBearPower(string symbol, string timeframe);
        Task<double> GetWilliamsR(string symbol, string timeframe);
        Task<(double StochK, double StochD)> GetSTOCH(string symbol, string timeframe);
        Task<double> GetUltimateOscillator(string symbol, string timeframe);
        Task<double> GetCCI(string symbol, string interval = "1day");
        
        // Indicator correlation analysis methods
        Task<IndicatorCorrelationResult> CalculateIndicatorCorrelation(string symbol, string firstIndicator, string secondIndicator, string timeframe = "1day", int dataPoints = 30);
        Task<List<IndicatorCorrelationResult>> CalculateAllIndicatorCorrelations(string symbol, string timeframe = "1day", int dataPoints = 30);
        Task<List<IndicatorConfirmationPattern>> FindConfirmationPatterns(string symbol, string timeframe = "1day");
        
        // Custom indicator methods
        Task<IIndicator> GetIndicatorAsync(string indicatorId);
        Task<bool> RegisterIndicatorAsync(IIndicator indicator);
        Task<bool> UnregisterIndicatorAsync(string indicatorId);
        Task<List<IIndicator>> GetAllIndicatorsAsync();
        Task<Dictionary<string, double>> CalculateCustomIndicatorAsync(string indicatorId, string symbol, string timeframe);
        Task<CustomIndicatorDefinition> GetIndicatorDefinitionAsync(string indicatorId);
        Task<bool> SaveIndicatorDefinitionAsync(CustomIndicatorDefinition definition);
        Task<List<CustomIndicatorDefinition>> SearchIndicatorsAsync(string searchTerm, string category = null);
        
        // Indicator correlation analysis methods
        Task<double> GetIndicatorCorrelation(string symbol, string timeframe, string indicator1, string indicator2, int period = 30);
        
        // Methods for visualization framework
        (List<double> Upper, List<double> Middle, List<double> Lower) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier);
        List<double> CalculateSMA(List<double> prices, int period);
        List<double> CalculateEMA(List<double> prices, int period);
        List<double> CalculateVWAP(List<double> highPrices, List<double> lowPrices, List<double> closePrices, List<double> volumes);
        List<double> CalculateRSI(List<double> prices, int period);
        (List<double> MacdLine, List<double> SignalLine, List<double> Histogram) CalculateMACD(List<double> prices, int fastPeriod, int slowPeriod, int signalPeriod);
        (List<double> K, List<double> D) CalculateStochastic(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int kPeriod, int kSmoothing, int dPeriod);
        List<double> CalculateStochRSI(List<double> prices, int rsiPeriod, int stochPeriod, int kPeriod, int dPeriod);
        List<double> CalculateWilliamsR(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period);
        List<double> CalculateCCI(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period);
        List<double> CalculateADX(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period = 14);
        List<double> CalculateATR(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period = 14);
        List<double> CalculateROC(List<double> prices, int period = 10);
        List<double> CalculateUltimateOscillator(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period1 = 7, int period2 = 14, int period3 = 28);
    }
}
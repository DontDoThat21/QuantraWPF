using System;
using System.Collections.Generic;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    public interface IPositionSizingService
    {
        /// <summary>
        /// Calculates position size based on risk parameters
        /// </summary>
        int CalculatePositionSizeByRisk(string symbol, double price, double stopLossPrice, double riskPercentage, double accountSize);
        
        /// <summary>
        /// Calculates position size using adaptive risk adjustment
        /// </summary>
        int CalculatePositionSizeByAdaptiveRisk(string symbol, double price, double stopLossPrice, 
            double riskPercentage, double accountSize, double volatility = 0, double winRate = 0.5);
            
        /// <summary>
        /// Calculates position size using specified parameters and sizing method
        /// </summary>
        int CalculatePositionSize(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size by equity percentage
        /// </summary>
        int CalculatePositionSizeByEquityPercentage(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size based on volatility
        /// </summary>
        int CalculatePositionSizeByVolatility(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size using Kelly Formula
        /// </summary>
        int CalculatePositionSizeByKellyFormula(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size by fixed amount
        /// </summary>
        int CalculatePositionSizeByFixedAmount(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size by tier system
        /// </summary>
        int CalculatePositionSizeByTiers(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size using adaptive risk management
        /// </summary>
        int CalculatePositionSizeByAdaptiveRisk(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates position size by fixed risk
        /// </summary>
        int CalculatePositionSizeByFixedRisk(PositionSizingParameters parameters);
        
        /// <summary>
        /// Calculates maximum allowed position size
        /// </summary>
        int CalculateMaxPositionSize(PositionSizingParameters parameters);
    }
}
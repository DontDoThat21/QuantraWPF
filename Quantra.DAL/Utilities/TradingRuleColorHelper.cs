using System;

namespace Quantra.DAL.Utilities
{
    /// <summary>
    /// Provides color utilities for TradingRule without introducing WPF dependencies in the model.
    /// Returns colors as hex strings (#RRGGBB) suitable for UI conversion.
    /// </summary>
    public static class TradingRuleColorHelper
    {
        /// <summary>
        /// Gets a hex color string for a trading rule status based on active state and risk/reward ratio.
        /// </summary>
        /// <param name="isActive">Whether the trading rule is active.</param>
        /// <param name="riskRewardRatio">Computed risk/reward ratio.</param>
        /// <returns>Hex color string like #RRGGBB.</returns>
        public static string GetStatusColorHex(bool isActive, double riskRewardRatio)
        {
            if (!isActive)
                return "#808080"; // Gray

            if (riskRewardRatio >= 3)
                return "#008000"; // Green
            if (riskRewardRatio >= 2)
                return "#9ACD32"; // YellowGreen
            if (riskRewardRatio >= 1)
                return "#FFFF00"; // Yellow
            return "#FFA500";      // Orange
        }
    }
}

// Handles indicator logic for StockExplorer
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Quantra.Controls
{
    public partial class StockExplorer
    {
        // Indicator-related fields, properties, and methods
        private Dictionary<string, bool> _indicatorVisibility = new Dictionary<string, bool>
        {
            { "BollingerBands", true },
            { "Candles", true },
            { "Legend", false }
        };

        // Bollinger Bands toggle checked event handler
        private void ToggleBollingerBands_Checked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["BollingerBands"] = true;
        }

        // Bollinger Bands toggle unchecked event handler
        private void ToggleBollingerBands_Unchecked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["BollingerBands"] = false;
        }

        // Candles toggle checked event handler
        private void ToggleCandles_Checked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["Candles"] = true;
        }

        // Candles toggle unchecked event handler
        private void ToggleCandles_Unchecked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["Candles"] = false;
        }

        // Legend toggle checked event handler
        private void ToggleLegend_Checked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["Legend"] = true;
        }

        // Legend toggle unchecked event handler
        private void ToggleLegend_Unchecked(object sender, RoutedEventArgs e)
        {
            _indicatorVisibility["Legend"] = false;
        }
    }
}

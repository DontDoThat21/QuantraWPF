using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Navigation;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.Intelligence
{
    /// <summary>
    /// Converter to get appropriate color for sentiment label
    /// </summary>
    public class SentimentColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var sentiment = value as string;
            if (string.IsNullOrEmpty(sentiment))
                return new SolidColorBrush(Color.FromRgb(62, 62, 86)); // Neutral gray

            return sentiment.ToLowerInvariant() switch
            {
                "bullish" => new SolidColorBrush(Color.FromRgb(32, 192, 64)), // Green
                "somewhat-bullish" => new SolidColorBrush(Color.FromRgb(80, 224, 112)), // Light green
                "bearish" => new SolidColorBrush(Color.FromRgb(192, 32, 32)), // Red
                "somewhat-bearish" => new SolidColorBrush(Color.FromRgb(255, 107, 107)), // Light red
                "neutral" => new SolidColorBrush(Color.FromRgb(255, 204, 0)), // Yellow
                _ => new SolidColorBrush(Color.FromRgb(62, 62, 86)) // Default gray
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converter to format sentiment label for display
    /// </summary>
    public class SentimentLabelConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var sentiment = value as string;
            if (string.IsNullOrEmpty(sentiment))
                return "N/A";

            return sentiment.ToLowerInvariant() switch
            {
                "bullish" => "BULL",
                "somewhat-bullish" => "S-BULL",
                "bearish" => "BEAR",
                "somewhat-bearish" => "S-BEAR",
                "neutral" => "NEUT",
                _ => sentiment.ToUpperInvariant()
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interaction logic for NewsSentimentControl.xaml
    /// </summary>
    public partial class NewsSentimentControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private List<NewsSentimentItem> _newsItems;

        public NewsSentimentControl()
        {
            InitializeComponent();

            // Get services from DI if available
            try
            {
                _alphaVantageService = App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
                _loggingService = App.ServiceProvider?.GetService(typeof(LoggingService)) as LoggingService;
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Service initialization error: {ex.Message}";
            }
        }

        private async void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadNewsSentiment();
        }

        private async void SearchTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadNewsSentiment();
            }
        }

        private async System.Threading.Tasks.Task LoadNewsSentiment()
        {
            if (_alphaVantageService == null)
            {
                StatusText.Text = "Alpha Vantage service not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                LoadButton.IsEnabled = false;
                StatusText.Text = "Loading news sentiment data...";

                var searchInput = SearchTextBox.Text?.Trim();
                string tickers = null;
                string topics = null;

                // Parse limit parameter
                int limit = 50; // Default
                if (!string.IsNullOrWhiteSpace(LimitTextBox.Text))
                {
                    if (int.TryParse(LimitTextBox.Text, out int parsedLimit))
                    {
                        limit = Math.Min(1000, Math.Max(1, parsedLimit)); // Clamp between 1 and 1000
                    }
                }

                // Intelligent search detection
                if (!string.IsNullOrWhiteSpace(searchInput))
                {
                    // Check if input looks like a comma-separated ticker list
                    // Criteria: Contains commas OR is all uppercase letters with 1-5 chars (typical ticker)
                    bool looksLikeTickers = searchInput.Contains(',') || 
                                           (searchInput.Length <= 5 && searchInput.All(c => char.IsLetterOrDigit(c) || c == ':' || c == '.'));

                    // Additional check: if it contains spaces and no commas, it's likely a topic/phrase
                    if (searchInput.Contains(' ') && !searchInput.Contains(','))
                    {
                        looksLikeTickers = false;
                    }

                    if (looksLikeTickers)
                    {
                        // Treat as tickers - convert to uppercase and clean up
                        tickers = searchInput.ToUpper().Replace(" ", "");
                        _loggingService?.Log("Info", $"Detected ticker input: {tickers}");
                    }
                    else
                    {
                        // Treat as topic search - use as-is
                        topics = searchInput;
                        _loggingService?.Log("Info", $"Detected topic input: {topics}");
                    }
                }

                var response = await _alphaVantageService.GetNewsSentimentAsync(
                    tickers: tickers,
                    topics: topics,
                    limit: limit
                );

                // Build search criteria description for status message
                var searchCriteria = new List<string>();
                if (!string.IsNullOrEmpty(tickers))
                    searchCriteria.Add($"ticker(s): {tickers}");
                if (!string.IsNullOrEmpty(topics))
                    searchCriteria.Add($"topic: '{topics}'");
                var criteriaText = searchCriteria.Count > 0 ? string.Join(", ", searchCriteria) : "all news";

                if (response != null && response.Feed.Count > 0)
                {
                    // Sort by TimePublished in descending order (newest first)
                    _newsItems = response.Feed.OrderByDescending(item => item.TimePublished).ToList();
                    NewsListView.ItemsSource = _newsItems;
                    UpdateSummary();
                    SummaryPanel.Visibility = Visibility.Visible;
                    StatusText.Text = $"Last updated: {DateTime.Now:g} | {_newsItems.Count} articles loaded for {criteriaText} (limit: {limit})";
                    _loggingService?.Log("Info", $"Loaded {_newsItems.Count} news sentiment items");
                }
                else
                {
                    NewsListView.ItemsSource = null;
                    SummaryPanel.Visibility = Visibility.Collapsed;
                    StatusText.Text = $"No news articles found for {criteriaText}.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, "Error loading news sentiment");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void UpdateSummary()
        {
            if (_newsItems == null || _newsItems.Count == 0)
            {
                TotalArticlesText.Text = "0";
                BullishCountText.Text = "0";
                NeutralCountText.Text = "0";
                BearishCountText.Text = "0";
                return;
            }

            TotalArticlesText.Text = _newsItems.Count.ToString();

            int bullish = _newsItems.Count(n => 
                n.OverallSentimentLabel?.ToLowerInvariant() == "bullish" || 
                n.OverallSentimentLabel?.ToLowerInvariant() == "somewhat-bullish");

            int bearish = _newsItems.Count(n => 
                n.OverallSentimentLabel?.ToLowerInvariant() == "bearish" || 
                n.OverallSentimentLabel?.ToLowerInvariant() == "somewhat-bearish");

            int neutral = _newsItems.Count - bullish - bearish;

            BullishCountText.Text = bullish.ToString();
            NeutralCountText.Text = neutral.ToString();
            BearishCountText.Text = bearish.ToString();
        }

        private void Hyperlink_RequestNavigate(object sender, RequestNavigateEventArgs e)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = e.Uri.AbsoluteUri,
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error opening article link");
            }
            e.Handled = true;
        }

        /// <summary>
        /// Public method to load news for a specific ticker programmatically
        /// </summary>
        public async System.Threading.Tasks.Task LoadTickerAsync(string ticker)
        {
            if (!string.IsNullOrWhiteSpace(ticker))
            {
                SearchTextBox.Text = ticker.ToUpper();
                await LoadNewsSentiment();
            }
        }
    }
}

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

        private async void TickerTextBox_KeyDown(object sender, KeyEventArgs e)
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

                var ticker = TickerTextBox.Text?.Trim().ToUpper();
                string topic = null;

                // Get selected topic
                if (TopicComboBox.SelectedItem is ComboBoxItem selectedTopic)
                {
                    var topicText = selectedTopic.Content?.ToString();
                    if (topicText != "All Topics")
                    {
                        topic = topicText?.ToLowerInvariant();
                    }
                }

                var response = await _alphaVantageService.GetNewsSentimentAsync(
                    tickers: string.IsNullOrEmpty(ticker) ? null : ticker,
                    topics: topic,
                    limit: 50
                );

                if (response != null && response.Feed.Count > 0)
                {
                    _newsItems = response.Feed;
                    NewsListView.ItemsSource = _newsItems;
                    UpdateSummary();
                    SummaryPanel.Visibility = Visibility.Visible;
                    StatusText.Text = $"Last updated: {DateTime.Now:g} | {_newsItems.Count} articles loaded";
                    _loggingService?.Log("Info", $"Loaded {_newsItems.Count} news sentiment items");
                }
                else
                {
                    NewsListView.ItemsSource = null;
                    SummaryPanel.Visibility = Visibility.Collapsed;
                    StatusText.Text = "No news articles found for the specified criteria.";
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
                TickerTextBox.Text = ticker.ToUpper();
                await LoadNewsSentiment();
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch tweets and perform sentiment analysis using a Python script (GPU-accelerated).
    /// </summary>
    public class TwitterSentimentService : ISocialMediaSentimentService
    {
        private readonly string pythonScriptPath = "python/sentiment_analysis.py"; // Path to your Python script
        private readonly string twitterBearerToken = "YOUR_TWITTER_BEARER_TOKEN"; // TODO: Secure this

        /// <summary>
        /// Fetches recent tweets for a stock symbol using Twitter API v2.
        /// </summary>
        public async Task<List<string>> FetchRecentTweetsAsync(string symbol, int count = 20)
        {
            var tweets = new List<string>();
            try
            {
                using var client = new HttpClient();
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {twitterBearerToken}");
                string query = $"${symbol} stock lang:en -is:retweet";
                string url = $"https://api.twitter.com/2/tweets/search/recent?query={Uri.EscapeDataString(query)}&max_results={count}&tweet.fields=text";
                var response = await client.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    DatabaseMonolith.Log("Warning", $"Twitter API call failed for {symbol}: {response.StatusCode}");
                    return tweets;
                }
                var json = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("data", out var data))
                {
                    foreach (var tweet in data.EnumerateArray())
                    {
                        if (tweet.TryGetProperty("text", out var textProp))
                            tweets.Add(textProp.GetString());
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error fetching tweets for {symbol}", ex.ToString());
            }
            return tweets;
        }

        /// <summary>
        /// Calls the Python sentiment analysis script and returns the average sentiment score.
        /// </summary>
        public async Task<double> AnalyzeSentimentAsync(List<string> tweets)
        {
            if (tweets == null || tweets.Count == 0)
                return 0.0;
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"{pythonScriptPath}",
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using var process = new Process { StartInfo = psi };
                process.Start();
                // Send tweets as JSON to stdin
                var json = JsonSerializer.Serialize(tweets);
                await process.StandardInput.WriteLineAsync(json);
                process.StandardInput.Close();
                // Read output (expecting a single float value)
                string output = await process.StandardOutput.ReadLineAsync();
                string error = await process.StandardError.ReadToEndAsync();
                process.WaitForExit();
                if (!string.IsNullOrWhiteSpace(error))
                {
                    DatabaseMonolith.Log("Warning", $"Python sentiment script stderr: {error}");
                }
                if (double.TryParse(output, out double sentiment))
                    return sentiment;
                else
                    DatabaseMonolith.Log("Warning", $"Python sentiment script returned non-numeric output: {output}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error running Python sentiment analysis script", ex.ToString());
            }
            return 0.0;
        }

        /// <summary>
        /// High-level method: fetches tweets and returns average sentiment for a symbol.
        /// </summary>
        public async Task<double> GetSymbolSentimentAsync(string symbol)
        {
            var tweets = await FetchRecentTweetsAsync(symbol);
            return await AnalyzeSentimentAsync(tweets);
        }

        public Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)
        {
            throw new NotImplementedException();
        }

        public Task<List<string>> FetchRecentContentAsync(string symbol, int count = 10)
        {
            throw new NotImplementedException();
        }
    }
}

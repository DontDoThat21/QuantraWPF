using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.Data;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch earnings call transcripts and perform advanced NLP analysis
    /// </summary>
    public class EarningsTranscriptService : IEarningsTranscriptService
    {
        private readonly string pythonScriptPath = "python/earnings_transcript_analysis.py";
        private readonly HttpClient _httpClient;
        private readonly string _apiKey; // API key for transcript provider

        public EarningsTranscriptService()
        {
            _httpClient = new HttpClient();
            _apiKey = GetTranscriptApiKey();
        }

        /// <summary>
        /// Get API key for transcript provider from application configuration
        /// </summary>
        private string GetTranscriptApiKey()
        {
            // In a real application, get this from configuration
            // For prototype, we'll use a placeholder
            return "YOUR_API_KEY";
        }

        /// <summary>
        /// Fetches the most recent earnings call transcript for a stock symbol
        /// </summary>
        public async Task<string> FetchLatestEarningsTranscriptAsync(string symbol)
        {
            try
            {
                // In a production system, you'd call a transcript provider API
                // For prototype, we'll simulate fetching from a mock service or database
                
                DatabaseMonolith.Log("Info", $"Fetching earnings transcript for {symbol}");
                
                // Simulated API call - in production, replace with actual API call
                await Task.Delay(200); // Simulate network delay
                
                // For demo purposes, generate a simple mock transcript if we can't fetch a real one
                string mockTranscript = GenerateMockTranscript(symbol);
                
                return mockTranscript;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error fetching earnings transcript for {symbol}", ex.ToString());
                return string.Empty;
            }
        }

        /// <summary>
        /// Fetch recent transcripts as strings
        /// </summary>
        public async Task<List<string>> FetchRecentContentAsync(string symbol, int count = 1)
        {
            var transcripts = new List<string>();
            try
            {
                string transcript = await FetchLatestEarningsTranscriptAsync(symbol);
                if (!string.IsNullOrEmpty(transcript))
                {
                    transcripts.Add(transcript);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error fetching recent earnings content for {symbol}", ex.ToString());
            }
            return transcripts;
        }

        /// <summary>
        /// High-level method: fetches transcript and returns average sentiment for a symbol
        /// </summary>
        public async Task<double> GetSymbolSentimentAsync(string symbol)
        {
            var result = await GetEarningsTranscriptAnalysisAsync(symbol);
            return result?.SentimentScore ?? 0.0;
        }

        /// <summary>
        /// Analyzes sentiment from a list of text content
        /// </summary>
        public async Task<double> AnalyzeSentimentAsync(List<string> textContent)
        {
            if (textContent == null || textContent.Count == 0)
                return 0.0;

            try
            {
                // For earnings transcripts, we'll just take the first item in the list
                // (normally there will be only one transcript)
                var transcript = textContent[0];
                var result = await AnalyzeEarningsTranscriptAsync(transcript);
                return result.SentimentScore;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error analyzing earnings transcript sentiment", ex.ToString());
                return 0.0;
            }
        }

        /// <summary>
        /// Gets detailed sentiment data for a symbol by source.
        /// For earnings, we don't have multiple sources, so we'll use this to return sentiment breakdown.
        /// </summary>
        public async Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)
        {
            try
            {
                var result = await GetEarningsTranscriptAnalysisAsync(symbol);
                if (result != null && result.SentimentDistribution != null)
                {
                    return result.SentimentDistribution;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting detailed sentiment for {symbol}", ex.ToString());
            }
            
            return new Dictionary<string, double>();
        }

        /// <summary>
        /// Analyzes an earnings call transcript using NLP techniques
        /// </summary>
        public async Task<EarningsTranscriptAnalysisResult> AnalyzeEarningsTranscriptAsync(string transcript)
        {
            var result = new EarningsTranscriptAnalysisResult();
            
            if (string.IsNullOrWhiteSpace(transcript))
                return result;
                
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
                
                // Send transcript as JSON to stdin
                var inputData = new { transcript };
                var json = JsonSerializer.Serialize(inputData);
                await process.StandardInput.WriteLineAsync(json);
                process.StandardInput.Close();
                
                // Read output (expecting a JSON object with analysis results)
                string output = await process.StandardOutput.ReadLineAsync();
                string error = await process.StandardError.ReadToEndAsync();
                
                process.WaitForExit();
                
                if (!string.IsNullOrWhiteSpace(error))
                {
                    DatabaseMonolith.Log("Warning", $"Python transcript analysis script stderr: {error}");
                }
                
                if (!string.IsNullOrEmpty(output))
                {
                    // Parse the JSON output
                    var analysisResults = JsonDocument.Parse(output);
                    var root = analysisResults.RootElement;
                    
                    // Extract sentiment data
                    if (root.TryGetProperty("sentiment", out var sentimentElement))
                    {
                        if (sentimentElement.TryGetProperty("score", out var scoreElement))
                        {
                            result.SentimentScore = scoreElement.GetDouble();
                        }
                        
                        if (sentimentElement.TryGetProperty("distribution", out var distElement))
                        {
                            foreach (var property in distElement.EnumerateObject())
                            {
                                result.SentimentDistribution[property.Name] = property.Value.GetDouble();
                            }
                        }
                    }
                    
                    // Extract topics
                    if (root.TryGetProperty("topics", out var topicsElement))
                    {
                        foreach (var topic in topicsElement.EnumerateArray())
                        {
                            result.KeyTopics.Add(topic.GetString());
                        }
                    }
                    
                    // Extract entities
                    if (root.TryGetProperty("entities", out var entitiesElement))
                    {
                        foreach (var entityProperty in entitiesElement.EnumerateObject())
                        {
                            var entityList = new List<string>();
                            foreach (var entity in entityProperty.Value.EnumerateArray())
                            {
                                entityList.Add(entity.GetString());
                            }
                            result.NamedEntities[entityProperty.Name] = entityList;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error analyzing earnings transcript", ex.ToString());
            }
            
            return result;
        }

        /// <summary>
        /// Fetches and analyzes the most recent earnings call transcript for a stock symbol
        /// </summary>
        public async Task<EarningsTranscriptAnalysisResult> GetEarningsTranscriptAnalysisAsync(string symbol)
        {
            var transcript = await FetchLatestEarningsTranscriptAsync(symbol);
            if (string.IsNullOrEmpty(transcript))
            {
                return new EarningsTranscriptAnalysisResult { Symbol = symbol };
            }
            
            var result = await AnalyzeEarningsTranscriptAsync(transcript);
            result.Symbol = symbol;
            
            // Extract quarter and date info from the transcript
            ExtractEarningsMetadata(transcript, result);
            
            return result;
        }

        /// <summary>
        /// Gets historical earnings call transcript analysis for a stock
        /// </summary>
        public async Task<List<EarningsTranscriptAnalysisResult>> GetHistoricalEarningsAnalysisAsync(string symbol, int quarters = 4)
        {
            // In a production system, you'd fetch historical transcripts
            // For the prototype, we'll just return the most recent one
            var result = new List<EarningsTranscriptAnalysisResult>();
            
            try
            {
                var latestAnalysis = await GetEarningsTranscriptAnalysisAsync(symbol);
                if (latestAnalysis != null)
                {
                    result.Add(latestAnalysis);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting historical earnings analysis for {symbol}", ex.ToString());
            }
            
            return result;
        }

        /// <summary>
        /// Extract metadata from transcript like quarter and date
        /// </summary>
        private void ExtractEarningsMetadata(string transcript, EarningsTranscriptAnalysisResult result)
        {
            try
            {
                // Extract quarter info using regex
                var quarterMatch = Regex.Match(transcript, @"Q[1-4]\s+\d{4}");
                if (quarterMatch.Success)
                {
                    result.Quarter = quarterMatch.Value;
                }
                else
                {
                    // Default to current quarter
                    int currentQuarter = (DateTime.Now.Month - 1) / 3 + 1;
                    result.Quarter = $"Q{currentQuarter} {DateTime.Now.Year}";
                }
                
                // Extract or estimate date
                var dateMatch = Regex.Match(transcript, @"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b");
                if (dateMatch.Success)
                {
                    if (DateTime.TryParse(dateMatch.Value, out DateTime date))
                    {
                        result.EarningsDate = date;
                    }
                    else
                    {
                        result.EarningsDate = DateTime.Now.AddDays(-7); // Default to a week ago
                    }
                }
                else
                {
                    result.EarningsDate = DateTime.Now.AddDays(-7); // Default to a week ago
                }
            }
            catch
            {
                // Default values if extraction fails
                int currentQuarter = (DateTime.Now.Month - 1) / 3 + 1;
                result.Quarter = $"Q{currentQuarter} {DateTime.Now.Year}";
                result.EarningsDate = DateTime.Now.AddDays(-7);
            }
        }
        
        /// <summary>
        /// Generate a mock transcript for testing when API is unavailable
        /// </summary>
        private string GenerateMockTranscript(string symbol)
        {
            int currentQuarter = (DateTime.Now.Month - 1) / 3 + 1;
            string quarter = $"Q{currentQuarter} {DateTime.Now.Year}";
            DateTime callDate = DateTime.Now.AddDays(-7);
            string dateFormatted = callDate.ToString("MMMM d, yyyy");
            
            return $@"{symbol} Inc. ({symbol}) Q{currentQuarter} {DateTime.Now.Year} Earnings Conference Call {dateFormatted}

Operator: Good morning and welcome to the {symbol} Inc. {quarter} Earnings Conference Call. [Operator Instructions] I would now like to turn the call over to our host, Jane Smith, Investor Relations. Please go ahead.

Jane Smith: Thank you, operator. Good morning, everyone, and thank you for joining us today for {symbol}'s {quarter} earnings conference call. With me today are John Doe, Chief Executive Officer; and Sarah Johnson, Chief Financial Officer.

Before we begin, I'd like to remind you that today's discussion will contain forward-looking statements that involve risks and uncertainties. Actual results may differ materially from those discussed today.

Now, I'll turn the call over to John.

John Doe: Thank you, Jane, and good morning, everyone. We're pleased to report another strong quarter with revenue growth of 12% year-over-year, reaching $325 million. This growth was primarily driven by our expansion in the cloud services segment, which grew 28% compared to the same period last year.

Our flagship product, {symbol} Analytics Platform, continues to gain market share and we added over 200 new enterprise customers this quarter. Customer retention remains strong at 95%, reflecting the value our solutions provide.

We're particularly excited about the launch of our new AI-powered forecasting tool, which has received positive feedback from early adopters. This innovation positions us well against competitors and opens new market opportunities.

Looking at our regional performance, North American revenue increased by 15%, while international markets grew by 8%. We see significant growth potential in the European market despite current economic headwinds.

I'll now turn the call over to Sarah to provide more details on our financial results.

Sarah Johnson: Thank you, John, and hello everyone. As John mentioned, total revenue for the quarter was $325 million, up 12% year-over-year and exceeding our guidance range of $310 to $320 million.

Subscription revenue grew 18% to $245 million, representing 75% of our total revenue. Professional services revenue was $80 million, down slightly from the previous quarter but in line with our expectations as we continue to transition to a partner-led implementation model.

Gross margin improved to 72%, up from 70% in the same period last year, driven by economies of scale in our cloud operations and improved efficiency in our customer support function.

Operating expenses were $180 million, representing 55% of revenue compared to 58% last year. This improvement reflects our ongoing commitment to operational efficiency while still investing in innovation and go-to-market capabilities.

As a result, operating income was $54 million, up 25% year-over-year, with an operating margin of 16.6%, exceeding our guidance.

We generated free cash flow of $45 million in the quarter and ended with $550 million in cash and short-term investments. Our strong balance sheet positions us well for both organic investments and potential strategic acquisitions.

Looking ahead to next quarter, we expect revenue in the range of $330 to $340 million, representing year-over-year growth of approximately 10-13%. For the full year, we're raising our guidance to $1.3-1.32 billion, up from our previous range of $1.28-1.3 billion.

With that, I'll turn the call back to John for closing remarks.

John Doe: Thanks, Sarah. In conclusion, we're very pleased with our performance this quarter. Our strategy of focusing on innovation, customer success, and operational excellence continues to deliver results. We remain confident in our ability to execute against our long-term growth objectives despite the challenging macroeconomic environment.

I want to thank our employees for their hard work and dedication, our customers for their trust, and our shareholders for their continued support. We look forward to updating you on our progress next quarter.

Operator: Thank you. This concludes today's conference call. You may now disconnect.";
        }
    }
}
using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for TradingRulesControl.xaml
    /// </summary>
    public partial class TradingRulesControl : UserControl, INotifyPropertyChanged
    {
        private ObservableCollection<TradingRule> tradingRules;
        private TradingRule currentRule;
        private bool isEditMode;

        public event PropertyChangedEventHandler PropertyChanged;

        public TradingRulesControl()
        {
            InitializeComponent();
            
            // Initialize collections
            tradingRules = new ObservableCollection<TradingRule>();
            
            // Set the data context
            this.DataContext = this;
            
            // Bind the DataGrid to the rules collection
            RulesDataGrid.ItemsSource = tradingRules;
            
            // Load rules from database
            LoadRulesFromDatabase();
        }

        private void LoadRulesFromDatabase()
        {
            try
            {
                tradingRules.Clear();
                var rules = DatabaseMonolith.GetTradingRules();
                
                foreach (var rule in rules)
                {
                    // Parse indicators and conditions from saved rule
                    if (!string.IsNullOrEmpty(rule.Condition) && 
                        (rule.Conditions == null || rule.Conditions.Count == 0))
                    {
                        rule.Conditions = new List<string> { rule.Condition };
                    }

                    // Update indicators based on conditions
                    UpdateRuleIndicators(rule);

                    // Calculate risk/reward
                    if (ValidateRule(rule, out _))
                    {
                        var profitAmount = Math.Abs(rule.ExitPrice - rule.EntryPrice) * rule.Quantity;
                        var lossAmount = Math.Abs(rule.StopLoss - rule.EntryPrice) * rule.Quantity;
                        rule.Description = $"R/R Ratio: {rule.RiskRewardRatio:F2}. " +
                                        $"Potential profit: ${profitAmount:F2}. " +
                                        $"Max loss: ${lossAmount:F2}";
                    }

                    tradingRules.Add(rule);
                }

                //DatabaseMonolith.Log("Info", $"Loaded {rules.Count} trading rules from database");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading trading rules: {ex.Message}", "Error", 
                               MessageBoxButton.OK, MessageBoxImage.Error);
                //DatabaseMonolith.Log("Error", "Failed to load trading rules from database", ex.ToString());
            }
        }

        private void SearchBox_KeyUp(object sender, KeyEventArgs e)
        {
            string query = SearchBox.Text.ToUpper().Trim();
            
            if (string.IsNullOrWhiteSpace(query))
            {
                // Show all rules if search is empty
                RulesDataGrid.ItemsSource = tradingRules;
            }
            else
            {
                // Filter rules based on search
                var filtered = tradingRules.Where(r => 
                    r.Name.ToUpper().Contains(query) ||
                    r.Symbol.ToUpper().Contains(query) ||
                    r.OrderType.ToUpper().Contains(query) ||
                    r.Timeframe.ToUpper().Contains(query) ||
                    // Search in conditions
                    (r.Conditions != null && r.Conditions.Any(c => c.ToUpper().Contains(query))) ||
                    // Search in legacy condition
                    (!string.IsNullOrEmpty(r.Condition) && r.Condition.ToUpper().Contains(query)) ||
                    // Search in indicators
                    (r.Indicators != null && r.Indicators.Any(i => i.Key.ToUpper().Contains(query) ||
                                                                  i.Value.ToString().Contains(query))) ||
                    // Search in description
                    (!string.IsNullOrEmpty(r.Description) && r.Description.ToUpper().Contains(query))
                );
                
                RulesDataGrid.ItemsSource = filtered;
            }
        }

        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            // Refresh from database and reset search filter
            LoadRulesFromDatabase();
            SearchBox.Text = "";
            RulesDataGrid.ItemsSource = tradingRules;
        }

        private void RulesDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Optional: Handle row selection
        }

        private void AddRuleButton_Click(object sender, RoutedEventArgs e)
        {
            // Create a new rule in add mode
            currentRule = new TradingRule();
            isEditMode = false;
            
            // Clear and prepare the editor form
            PopulateRuleEditor(currentRule);
            
            // Show the popup
            RuleEditorPopup.IsOpen = true;
        }

        private void EditRule_Click(object sender, RoutedEventArgs e)
        {
            Button button = sender as Button;
            TradingRule rule = button.DataContext as TradingRule;
            
            // Set the current rule in edit mode
            currentRule = rule;
            isEditMode = true;
            
            // Populate the editor form with the selected rule
            PopulateRuleEditor(rule);
            
            // Show the popup
            RuleEditorPopup.IsOpen = true;
        }

        private void DeleteRule_Click(object sender, RoutedEventArgs e)
        {
            Button button = sender as Button;
            TradingRule rule = button.DataContext as TradingRule;
            
            // Confirm deletion
            MessageBoxResult result = MessageBox.Show($"Are you sure you want to delete the rule '{rule.Name}'?", 
                "Confirm Delete", MessageBoxButton.YesNo, MessageBoxImage.Question);
            
            if (result == MessageBoxResult.Yes)
            {
                try
                {
                    // Remove from database first
                    DatabaseMonolith.DeleteRule(rule.Id);
                    
                    // Then remove from the UI collection
                    tradingRules.Remove(rule);
                    
                    //DatabaseMonolith.Log("Info", $"Trading rule '{rule.Name}' deleted");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error deleting rule: {ex.Message}", "Error",
                                   MessageBoxButton.OK, MessageBoxImage.Error);
                    //DatabaseMonolith.Log("Error", "Error deleting trading rule", ex.ToString());
                }
            }
        }

        private void SaveRuleButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Basic form validation
                if (string.IsNullOrWhiteSpace(RuleNameTextBox.Text) ||
                    string.IsNullOrWhiteSpace(SymbolTextBox.Text) ||
                    OrderTypeComboBox.SelectedItem == null ||
                    TimeframeComboBox.SelectedItem == null)
                {
                    MessageBox.Show("Please fill in all required fields", "Validation Error", 
                                   MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                // Parse numeric values
                if (!double.TryParse(EntryPriceTextBox.Text, out double entryPrice) ||
                    !double.TryParse(ExitPriceTextBox.Text, out double exitPrice) ||
                    !double.TryParse(StopLossTextBox.Text, out double stopLoss) ||
                    !int.TryParse(QuantityTextBox.Text, out int quantity))
                {
                    MessageBox.Show("Please enter valid numeric values for prices and quantity", 
                                   "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                // Update the current rule from form values
                currentRule.Name = RuleNameTextBox.Text;
                currentRule.Symbol = SymbolTextBox.Text.ToUpper();
                currentRule.OrderType = ((ComboBoxItem)OrderTypeComboBox.SelectedItem).Content.ToString();
                currentRule.EntryPrice = entryPrice;
                currentRule.ExitPrice = exitPrice;
                currentRule.StopLoss = stopLoss;
                currentRule.Quantity = quantity;
                currentRule.Timeframe = ((ComboBoxItem)TimeframeComboBox.SelectedItem).Content.ToString();

                // Parse and set conditions
                var conditionsText = ConditionTextBox.Text;
                if (!string.IsNullOrWhiteSpace(conditionsText))
                {
                    currentRule.Conditions = ParseConditions(conditionsText);
                    currentRule.Condition = currentRule.Conditions.FirstOrDefault() ?? string.Empty;
                }
                else
                {
                    currentRule.Conditions = new List<string>();
                    currentRule.Condition = string.Empty;
                }

                currentRule.IsActive = IsActiveCheckBox.IsChecked ?? false;

                // Update indicators based on condition
                UpdateRuleIndicators(currentRule);

                // Validate the rule
                if (!ValidateRule(currentRule, out string errorMessage))
                {
                    MessageBox.Show(errorMessage, "Validation Error", 
                                   MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                // Set description with validation details
                currentRule.Description = $"R/R Ratio: {currentRule.RiskRewardRatio:F2}. " +
                                       $"Potential profit: ${Math.Abs(exitPrice - entryPrice) * quantity:F2}. " +
                                       $"Max loss: ${Math.Abs(stopLoss - entryPrice) * quantity:F2}";

                // Save to database and update collection
                if (!isEditMode)
                {
                    tradingRules.Add(currentRule);

                    // Let DatabaseMonolith handle ID creation
                    DatabaseMonolith.SaveTradingRule(currentRule);
                    //DatabaseMonolith.Log("Info", $"New trading rule '{currentRule.Name}' created");
                }
                else
                {
                    // The rule is already in the collection, just trigger UI refresh and update database
                    int index = tradingRules.IndexOf(currentRule);
                    tradingRules[index] = tradingRules[index]; // This triggers UI refresh
                    DatabaseMonolith.SaveTradingRule(currentRule);
                    //DatabaseMonolith.Log("Info", $"Trading rule '{currentRule.Name}' updated");
                }

                // Close the popup
                RuleEditorPopup.IsOpen = false;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving rule: {ex.Message}", "Error", 
                               MessageBoxButton.OK, MessageBoxImage.Error);
                //DatabaseMonolith.Log("Error", "Error saving trading rule", ex.ToString());
            }
        }

        private void CancelRuleButton_Click(object sender, RoutedEventArgs e)
        {
            // Just close the popup without saving
            RuleEditorPopup.IsOpen = false;
        }

        private void PopulateRuleEditor(TradingRule rule)
        {
            // Fill the editor form with the rule data
            RuleNameTextBox.Text = rule.Name;
            SymbolTextBox.Text = rule.Symbol;
            
            // Select the order type in the combo box
            foreach (ComboBoxItem item in OrderTypeComboBox.Items)
            {
                if (item.Content.ToString() == rule.OrderType)
                {
                    OrderTypeComboBox.SelectedItem = item;
                    break;
                }
            }
            
            EntryPriceTextBox.Text = rule.EntryPrice.ToString();
            ExitPriceTextBox.Text = rule.ExitPrice.ToString();
            StopLossTextBox.Text = rule.StopLoss.ToString();
            QuantityTextBox.Text = rule.Quantity.ToString();
            
            // Select the timeframe in the combo box
            foreach (ComboBoxItem item in TimeframeComboBox.Items)
            {
                if (item.Content.ToString() == rule.Timeframe)
                {
                    TimeframeComboBox.SelectedItem = item;
                    break;
                }
            }

            // Handle conditions
            var conditions = rule.Conditions ?? new List<string>();
            if (conditions.Count == 0 && !string.IsNullOrEmpty(rule.Condition))
            {
                // If we have a legacy single condition, add it to the list
                conditions.Add(rule.Condition);
            }
            
            // Join conditions with newlines for display
            ConditionTextBox.Text = string.Join(Environment.NewLine, conditions);
            
            IsActiveCheckBox.IsChecked = rule.IsActive;
            
            // Update description with risk/reward info if available
            if (rule.RiskRewardRatio > 0)
            {
                var potentialProfit = Math.Abs(rule.ExitPrice - rule.EntryPrice) * rule.Quantity;
                var maxLoss = Math.Abs(rule.StopLoss - rule.EntryPrice) * rule.Quantity;
                rule.Description = $"R/R Ratio: {rule.RiskRewardRatio:F2}. " +
                                 $"Potential profit: ${potentialProfit:F2}. " +
                                 $"Max loss: ${maxLoss:F2}";
            }
        }

        private List<string> ParseConditions(string conditionsText)
        {
            if (string.IsNullOrWhiteSpace(conditionsText))
                return new List<string>();

            return conditionsText
                .Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries)
                .Select(c => c.Trim())
                .Where(c => !string.IsNullOrWhiteSpace(c))
                .ToList();
        }

        private void GenerateRandomRules(int count)
        {
            string[] symbols = { "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM" };
            string[] orderTypes = { "BUY", "SELL" };
            string[] timeframes = { "1min", "5min", "15min", "30min", "1hour", "4hour", "1day" };
            string[] conditions = {
                "Price > 50 day MA",
                "RSI < 30",
                "RSI > 70",
                "MACD crossover",
                "Price at support",
                "Price at resistance",
                "Volume spike",
                "Bollinger band squeeze"
            };
            
            var random = new Random();
            
            for (int i = 0; i < count; i++)
            {
                string symbol = symbols[random.Next(symbols.Length)];
                string orderType = orderTypes[random.Next(orderTypes.Length)];
                double basePrice = 100 + random.NextDouble() * 200;
                
                double entryPrice = basePrice;
                double exitPrice, stopLoss;
                
                if (orderType == "BUY")
                {
                    exitPrice = entryPrice * (1 + 0.05 + random.NextDouble() * 0.1); // 5-15% profit
                    stopLoss = entryPrice * (1 - 0.02 - random.NextDouble() * 0.05); // 2-7% loss
                }
                else
                {
                    exitPrice = entryPrice * (1 - 0.05 - random.NextDouble() * 0.1); // 5-15% profit
                    stopLoss = entryPrice * (1 + 0.02 + random.NextDouble() * 0.05); // 2-7% loss
                }
                
                tradingRules.Add(new TradingRule
                {
                    Name = $"{symbol} {orderType} Rule {i+1}",
                    Symbol = symbol,
                    OrderType = orderType,
                    EntryPrice = Math.Round(entryPrice, 2),
                    ExitPrice = Math.Round(exitPrice, 2),
                    StopLoss = Math.Round(stopLoss, 2),
                    Quantity = random.Next(10, 200),
                    Timeframe = timeframes[random.Next(timeframes.Length)],
                    Condition = conditions[random.Next(conditions.Length)],
                    IsActive = random.Next(100) < 70 // 70% chance of being active
                });
            }
        }

        private bool ValidateRule(TradingRule rule, out string errorMessage)
        {
            errorMessage = string.Empty;

            try
            {
                // Use built-in validation
                if (!rule.Validate())
                {
                    errorMessage = "Rule validation failed. Please check all fields.";
                    return false;
                }

                // Additional validation logic
                if (rule.OrderType == "BUY")
                {
                    if (rule.ExitPrice <= rule.EntryPrice)
                    {
                        errorMessage = "For BUY orders, exit price must be higher than entry price";
                        return false;
                    }
                    if (rule.StopLoss >= rule.EntryPrice)
                    {
                        errorMessage = "For BUY orders, stop loss must be lower than entry price";
                        return false;
                    }
                }
                else if (rule.OrderType == "SELL")
                {
                    if (rule.ExitPrice >= rule.EntryPrice)
                    {
                        errorMessage = "For SELL orders, exit price must be lower than entry price";
                        return false;
                    }
                    if (rule.StopLoss <= rule.EntryPrice)
                    {
                        errorMessage = "For SELL orders, stop loss must be higher than entry price";
                        return false;
                    }
                }

                // Calculate and validate risk/reward ratio
                double reward = Math.Abs(rule.ExitPrice - rule.EntryPrice);
                double risk = Math.Abs(rule.StopLoss - rule.EntryPrice);
                rule.RiskRewardRatio = reward / risk;

                if (rule.RiskRewardRatio < 1.5)
                {
                    errorMessage = "Risk/Reward ratio should be at least 1.5";
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                errorMessage = $"Validation error: {ex.Message}";
                return false;
            }
        }

        private void UpdateRuleIndicators(TradingRule rule)
        {
            try
            {
                // Parse indicators from condition text
                var indicators = new Dictionary<string, double>();
                
                if (rule.Condition.Contains("MA") || rule.Condition.Contains("Moving Average"))
                {
                    indicators["MovingAverage"] = rule.EntryPrice;
                }
                if (rule.Condition.Contains("RSI"))
                {
                    var rsiMatch = System.Text.RegularExpressions.Regex.Match(rule.Condition, @"RSI [<>] (\d+)");
                    if (rsiMatch.Success)
                    {
                        indicators["RSI"] = double.Parse(rsiMatch.Groups[1].Value);
                    }
                }
                if (rule.Condition.Contains("MACD"))
                {
                    indicators["MACD"] = 0; // Default value, would be calculated in real implementation
                    indicators["Signal"] = 0;
                }

                rule.Indicators = indicators;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error updating rule indicators", ex.ToString());
            }
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

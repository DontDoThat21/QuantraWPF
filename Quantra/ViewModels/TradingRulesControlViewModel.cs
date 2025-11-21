using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.Models;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Trading Rules Control
    /// </summary>
    public class TradingRulesControlViewModel : ViewModelBase
    {
        private readonly ITradingRuleService _tradingRuleService;
        private TradingRule _currentRule;
        private bool _isEditMode;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public TradingRulesControlViewModel(ITradingRuleService tradingRuleService)
        {
            _tradingRuleService = tradingRuleService ?? throw new ArgumentNullException(nameof(tradingRuleService));
            
            TradingRules = new ObservableCollection<TradingRule>();
            
            InitializeCommands();
            _ = LoadRulesAsync();
        }

        #region Properties

        /// <summary>
        /// Collection of trading rules
        /// </summary>
        public ObservableCollection<TradingRule> TradingRules { get; }

        /// <summary>
        /// Currently selected/edited rule
        /// </summary>
        public TradingRule CurrentRule
        {
            get => _currentRule;
            set => SetProperty(ref _currentRule, value);
        }

        /// <summary>
        /// Indicates if in edit mode
        /// </summary>
        public bool IsEditMode
        {
            get => _isEditMode;
            set => SetProperty(ref _isEditMode, value);
        }

        #endregion

        #region Commands

        public ICommand AddRuleCommand { get; private set; }
        public ICommand EditRuleCommand { get; private set; }
        public ICommand DeleteRuleCommand { get; private set; }
        public ICommand SaveRuleCommand { get; private set; }
        public ICommand CancelEditCommand { get; private set; }
        public ICommand RefreshRulesCommand { get; private set; }

        #endregion

        #region Public Methods

        /// <summary>
        /// Load rules from database
        /// </summary>
        public async Task LoadRulesAsync()
        {
            try
            {
                TradingRules.Clear();
                var rules = await _tradingRuleService.GetTradingRulesAsync();
                
                foreach (var rule in rules)
                {
                    // Parse indicators and conditions from saved rule
                    if (!string.IsNullOrEmpty(rule.Condition) && 
                        (rule.Conditions == null || rule.Conditions.Count == 0))
                    {
                        rule.Conditions = new System.Collections.Generic.List<string> { rule.Condition };
                    }

                    TradingRules.Add(rule);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading trading rules: {ex.Message}");
            }
        }

        /// <summary>
        /// Save a trading rule
        /// </summary>
        public async Task<bool> SaveRuleAsync(TradingRule rule)
        {
            if (rule == null) return false;

            try
            {
                if (string.IsNullOrEmpty(rule.RuleId))
                {
                    // New rule
                    rule.RuleId = Guid.NewGuid().ToString();
                    await _tradingRuleService.AddTradingRuleAsync(rule);
                    TradingRules.Add(rule);
                }
                else
                {
                    // Update existing rule
                    await _tradingRuleService.UpdateTradingRuleAsync(rule);
                    
                    // Update in collection
                    var existingRule = TradingRules.FirstOrDefault(r => r.RuleId == rule.RuleId);
                    if (existingRule != null)
                    {
                        var index = TradingRules.IndexOf(existingRule);
                        TradingRules[index] = rule;
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving trading rule: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Delete a trading rule
        /// </summary>
        public async Task<bool> DeleteRuleAsync(TradingRule rule)
        {
            if (rule == null) return false;

            try
            {
                await _tradingRuleService.DeleteTradingRuleAsync(rule.RuleId);
                TradingRules.Remove(rule);
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error deleting trading rule: {ex.Message}");
                return false;
            }
        }

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            AddRuleCommand = new RelayCommand(ExecuteAddRule);
            EditRuleCommand = new RelayCommand(ExecuteEditRule, CanExecuteEditRule);
            DeleteRuleCommand = new RelayCommand(async param => await ExecuteDeleteRuleAsync(param), CanExecuteDeleteRule);
            SaveRuleCommand = new RelayCommand(async param => await ExecuteSaveRuleAsync(), CanExecuteSaveRule);
            CancelEditCommand = new RelayCommand(ExecuteCancelEdit);
            RefreshRulesCommand = new RelayCommand(async param => await LoadRulesAsync());
        }

        #endregion

        #region Command Implementations

        private void ExecuteAddRule(object parameter)
        {
            CurrentRule = new TradingRule
            {
                RuleId = Guid.NewGuid().ToString(),
                CreatedDate = DateTime.Now,
                IsActive = true
            };
            IsEditMode = true;
        }

        private bool CanExecuteEditRule(object parameter)
        {
            return parameter is TradingRule;
        }

        private void ExecuteEditRule(object parameter)
        {
            if (parameter is TradingRule rule)
            {
                CurrentRule = rule;
                IsEditMode = true;
            }
        }

        private bool CanExecuteDeleteRule(object parameter)
        {
            return parameter is TradingRule;
        }

        private async Task ExecuteDeleteRuleAsync(object parameter)
        {
            if (parameter is TradingRule rule)
            {
                await DeleteRuleAsync(rule);
            }
        }

        private bool CanExecuteSaveRule(object parameter)
        {
            return CurrentRule != null && !string.IsNullOrWhiteSpace(CurrentRule.RuleName);
        }

        private async Task ExecuteSaveRuleAsync()
        {
            if (CurrentRule != null)
            {
                var success = await SaveRuleAsync(CurrentRule);
                if (success)
                {
                    IsEditMode = false;
                    CurrentRule = null;
                }
            }
        }

        private void ExecuteCancelEdit(object parameter)
        {
            IsEditMode = false;
            CurrentRule = null;
        }

        #endregion
    }
}

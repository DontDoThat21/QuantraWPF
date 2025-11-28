using System;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for MarketChatControl.xaml
    /// </summary>
    public partial class MarketChatControl : UserControl
    {
        private MarketChatViewModel _viewModel;

        public MarketChatControl()
        {
            InitializeComponent();
            
            // Initialize the ViewModel
            try
            {
                // Create simple logger implementations
                var logger = new SimpleLogger("MarketChatService");
                var viewModelLogger = new SimpleLogger("MarketChatViewModel");
                
                var marketChatService = new MarketChatService(logger, null);
                _viewModel = new MarketChatViewModel(marketChatService, viewModelLogger);
                
                DataContext = _viewModel;
            }
            catch (Exception ex)
            {
                // Log error and show fallback UI
                MessageBox.Show($"Error initializing Market Chat: {ex.Message}", "Initialization Error", 
                               MessageBoxButton.OK, MessageBoxImage.Error);
            }

            // Subscribe to scroll to bottom when new messages arrive
            if (_viewModel != null)
            {
                _viewModel.Messages.CollectionChanged += (s, e) => ScrollToBottom();
            }
        }

        /// <summary>
        /// Handle Enter key to send message
        /// </summary>
        private void MessageTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                if (Keyboard.Modifiers == ModifierKeys.Shift)
                {
                    // Shift+Enter adds a new line, let default behavior handle it
                    return;
                }
                else
                {
                    // Enter sends the message
                    e.Handled = true;
                    if (_viewModel?.SendMessageCommand?.CanExecute(null) == true)
                    {
                        _viewModel.SendMessageCommand.Execute(null);
                    }
                }
            }
        }

        /// <summary>
        /// Scroll to the bottom of the chat
        /// </summary>
        private void ScrollToBottom()
        {
            // Update dispatcher monitoring before making the call
            SharedTitleBar.UpdateDispatcherMonitoring("ScrollToBottom");
            
            Dispatcher.InvokeAsync(() =>
            {
                ChatScrollViewer?.ScrollToBottom();
            });
        }
    }

    /// <summary>
    /// Template selector for different message types
    /// </summary>
    public class MessageTemplateSelector : DataTemplateSelector
    {
        public DataTemplate UserMessageTemplate { get; set; }
        public DataTemplate AssistantMessageTemplate { get; set; }
        public DataTemplate SystemMessageTemplate { get; set; }
        public DataTemplate LoadingMessageTemplate { get; set; }
        public DataTemplate QueryResultMessageTemplate { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item is MarketChatMessage message)
            {
                switch (message.MessageType)
                {
                    case MessageType.UserQuestion:
                        return UserMessageTemplate;
                    case MessageType.AssistantResponse:
                        // Use QueryResult template if this is a query result
                        if (message.IsQueryResult && QueryResultMessageTemplate != null)
                        {
                            return QueryResultMessageTemplate;
                        }
                        return AssistantMessageTemplate;
                    case MessageType.SystemMessage:
                        return SystemMessageTemplate;
                    case MessageType.LoadingMessage:
                        return LoadingMessageTemplate;
                    case MessageType.QueryResult:
                        return QueryResultMessageTemplate ?? AssistantMessageTemplate;
                }
            }

            return base.SelectTemplate(item, container);
        }
    }

    /// <summary>
    /// Simple logger implementation for Market Chat
    /// </summary>
    public class SimpleLogger : ILogger<MarketChatService>, ILogger<MarketChatViewModel>
    {
        private readonly string _name;

        public SimpleLogger(string name)
        {
            _name = name;
        }

        public IDisposable BeginScope<TState>(TState state) => null;
        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            var message = formatter(state, exception);
            System.Diagnostics.Debug.WriteLine($"[{logLevel}] {_name}: {message}");
            
            // Also log to DatabaseMonolith for consistency
            try
            {
                var level = logLevel switch
                {
                    LogLevel.Error => "Error",
                    LogLevel.Warning => "Warning",
                    LogLevel.Information => "Info",
                    _ => "Debug"
                };
                //DatabaseMonolith.Log(level, $"{_name}: {message}", exception?.ToString());
            }
            catch
            {
                // Ignore logging errors
            }
        }
    }
}
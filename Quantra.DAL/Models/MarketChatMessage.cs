using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a message in the market chat interface
    /// </summary>
    public class MarketChatMessage : INotifyPropertyChanged
    {
        private string _content;
        private bool _isFromUser;
        private DateTime _timestamp;
        private bool _isLoading;
        private MessageType _messageType;

        /// <summary>
        /// The content of the message
        /// </summary>
        public string Content
        {
            get => _content;
            set => SetProperty(ref _content, value);
        }

        /// <summary>
        /// Whether this message is from the user (true) or from the AI assistant (false)
        /// </summary>
        public bool IsFromUser
        {
            get => _isFromUser;
            set => SetProperty(ref _isFromUser, value);
        }

        /// <summary>
        /// When the message was created
        /// </summary>
        public DateTime Timestamp
        {
            get => _timestamp;
            set => SetProperty(ref _timestamp, value);
        }

        /// <summary>
        /// Whether this message is currently being loaded/generated
        /// </summary>
        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        /// <summary>
        /// The type of message for styling purposes
        /// </summary>
        public MessageType MessageType
        {
            get => _messageType;
            set => SetProperty(ref _messageType, value);
        }

        /// <summary>
        /// Formatted timestamp for display
        /// </summary>
        public string TimestampDisplay => Timestamp.ToString("HH:mm:ss");

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string propertyName = null)
        {
            if (Equals(storage, value))
            {
                return false;
            }

            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    /// <summary>
    /// Types of chat messages for styling and categorization
    /// </summary>
    public enum MessageType
    {
        /// <summary>
        /// Regular user question
        /// </summary>
        UserQuestion,

        /// <summary>
        /// AI assistant response
        /// </summary>
        AssistantResponse,

        /// <summary>
        /// System message (errors, status updates)
        /// </summary>
        SystemMessage,

        /// <summary>
        /// Loading/thinking indicator
        /// </summary>
        LoadingMessage
    }
}
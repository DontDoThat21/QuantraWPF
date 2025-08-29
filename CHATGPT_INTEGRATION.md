# ChatGPT Market Analysis Integration

This document describes the ChatGPT integration feature that provides a natural language interface for market analysis requests within the Quantra trading platform.

## Overview

The ChatGPT integration allows users to ask market-related questions in plain English and receive intelligent, context-aware responses powered by OpenAI's GPT models. The system is designed to provide professional financial analysis while maintaining conversation context.

## Features

### ✅ Implemented
- **Natural Language Interface**: Chat-style UI for asking market questions
- **Context-Aware Responses**: Maintains conversation history for follow-up questions  
- **Professional Financial Analysis**: System prompts optimized for trading insights
- **Real-time Market Context**: Includes current market session and timestamp in prompts
- **Error Handling**: Graceful fallbacks for API failures or connectivity issues
- **Conversation History**: Track and display previous Q&A pairs
- **Professional UI**: Material Design-based chat interface with message templates

### 🚧 Future Enhancements
- **Trading Plan Suggestions**: AI-generated trading strategies based on analysis
- **Voice Input**: Speech-to-text integration for hands-free operation
- **Feedback Mechanism**: User rating system to improve AI responses
- **Enhanced Context**: Integration with live market data, news, and sentiment

## Architecture

### Components

1. **MarketChatService** (`Services/MarketChatService.cs`)
   - Handles OpenAI API communication
   - Manages conversation context and history
   - Provides error handling and fallback responses

2. **MarketChatViewModel** (`ViewModels/MarketChatViewModel.cs`)
   - MVVM pattern implementation for UI state management
   - Command handling for user interactions
   - Observable collections for real-time UI updates

3. **MarketChatControl** (`Controls/MarketChatControl.xaml/.cs`)
   - WPF UserControl providing the chat interface
   - Message templates for different message types
   - Keyboard shortcuts and accessibility features

4. **MarketChatMessage** (`Models/MarketChatMessage.cs`)
   - Data model for chat messages
   - Support for different message types (user, assistant, system, loading)

## Setup Instructions

### 1. Configure OpenAI API Key

Add your OpenAI API key to the settings file:

```json
{
  "AlphaVantageApiKey": "your_alpha_vantage_key",
  "NewsApiKey": "your_news_api_key",
  "OpenAiApiKey": "sk-your-openai-api-key-here"
}
```

**File**: `Quantra/alphaVantageSettings.json`

### 2. Add the Chat Control

1. Run the application
2. Click the "+" tab to add a new control
3. Select "Market Chat" from the control type dropdown
4. Choose position and size for the chat interface
5. Click "Add" to integrate the chat into your dashboard

### 3. Start Chatting

Example questions you can ask:

```
- "What's the current sentiment around AAPL?"
- "Analyze the risks for tech stocks this quarter"
- "Explain the recent volatility in the market"
- "What should I watch for in tomorrow's trading session?"
- "Compare the outlook for energy vs technology sectors"
```

## Technical Details

### API Integration

- **Model**: Uses GPT-3.5-turbo by default (configurable) ⚠️ **OUTDATED - Requires upgrade to GPT-4.1+**
- **Temperature**: 0.3 for consistent, focused responses
- **Max Tokens**: 1000 per response for detailed analysis
- **Timeout**: 60 seconds per request
- **Rate Limiting**: Built-in through ResilienceHelper

### System Prompts

The system uses carefully crafted prompts to ensure professional financial analysis:

```
You are a professional financial analyst assistant for Quantra, an advanced algorithmic trading platform. 
Provide clear, concise, and actionable market analysis to help traders make informed decisions. 
Focus on technical analysis, market sentiment, risk factors, and actionable insights. 
Always mention relevant risks and avoid giving direct investment advice. 
Use professional yet accessible language appropriate for experienced traders.
```

### Context Management

- **Conversation History**: Maintains last 20 messages for context
- **Follow-up Support**: Previous Q&A pairs are included in new requests
- **Market Context**: Automatically includes current market session and timestamp

### Error Handling

- **API Failures**: Graceful fallback messages
- **Network Issues**: Timeout handling with retry logic
- **Invalid Responses**: Validation and error logging
- **Configuration Errors**: Clear error messages for missing API keys

## Security Considerations

- **API Key Storage**: Stored in local configuration file (not in code)
- **Request Logging**: Sanitized logging without sensitive data
- **Rate Limiting**: Built-in protection against API quota exhaustion
- **Input Validation**: Sanitization of user input before API calls

## Performance

- **Response Time**: Typically 2-5 seconds for standard queries
- **Memory Usage**: Conversation history limited to 20 messages
- **Token Management**: Automatic truncation of long contexts
- **UI Responsiveness**: Async operations prevent UI blocking

## Testing

Run the test script to verify setup:

```bash
python test_market_chat.py
```

This will check:
- ✅ Component files are present
- ✅ Configuration file exists
- ⚠️ API key configuration status

## Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Solution: Add valid API key to `alphaVantageSettings.json`

2. **"Error initializing Market Chat"**
   - Check API key format (should start with 'sk-')
   - Verify internet connectivity
   - Check application logs for detailed errors

3. **Chat responses are slow**
   - Normal for initial requests (2-5 seconds)
   - Check network connectivity
   - Verify OpenAI service status

4. **Control doesn't appear in Add Control list**
   - Rebuild the application
   - Check for compilation errors

### Logging

All chat interactions are logged for debugging:
- Request/response pairs (sanitized)
- Error conditions and exceptions
- Performance metrics

Check the application logs in the database for detailed troubleshooting information.

## Integration with Existing Features

The ChatGPT integration leverages existing Quantra infrastructure:

- **Configuration System**: Uses existing API key management
- **Logging System**: Integrates with DatabaseMonolith logging
- **Error Handling**: Uses ResilienceHelper for robust API calls
- **UI Framework**: Follows established MVVM patterns
- **Control Management**: Integrated with tab and grid management system

## Future Roadmap

### Phase 1 (Current)
- ✅ Basic chat interface
- ✅ Market analysis responses
- ✅ Conversation history

### Phase 2 (Planned)
- 🔄 Live data integration (prices, news, sentiment)
- 🔄 Trading plan generation
- 🔄 Voice input/output

### Phase 3 (Future)
- 📋 Custom prompt templates
- 📋 Response rating and feedback
- 📋 Multi-language support
- 📋 Advanced context sources

## Contributing

When contributing to the ChatGPT integration:

1. Follow existing code patterns and architecture
2. Add comprehensive error handling
3. Include logging for debugging
4. Test with various input scenarios
5. Update documentation as needed

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Run the test script for diagnostics
4. Check OpenAI API status and quotas
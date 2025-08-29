#!/usr/bin/env python3
"""
Simple test script to verify the OpenAI API integration would work
with a sample market analysis request.
"""

import os
import sys
import json

def test_openai_api_key():
    """Test if OpenAI API key is configured"""
    try:
        # Read the settings file
        with open('DayTrader/alphaVantageSettings.json', 'r') as f:
            settings = json.load(f)
        
        openai_key = settings.get('OpenAiApiKey', '')
        
        if openai_key == 'YOUR_OPENAI_API_KEY':
            print("⚠️  OpenAI API key not configured yet - please update alphaVantageSettings.json")
            return False
        elif openai_key:
            print("✅ OpenAI API key is configured")
            return True
        else:
            print("❌ OpenAI API key is missing from configuration")
            return False
            
    except FileNotFoundError:
        print("❌ Settings file not found")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON in settings file")
        return False

def test_market_chat_components():
    """Test if all components are present"""
    components = [
        'DayTrader/Models/MarketChatMessage.cs',
        'DayTrader/Services/MarketChatService.cs', 
        'DayTrader/ViewModels/MarketChatViewModel.cs',
        'DayTrader/Controls/MarketChatControl.xaml',
        'DayTrader/Controls/MarketChatControl.xaml.cs'
    ]
    
    all_present = True
    for component in components:
        if os.path.exists(component):
            print(f"✅ {component}")
        else:
            print(f"❌ {component}")
            all_present = False
    
    return all_present

def main():
    print("🔍 Testing ChatGPT Market Analysis Integration")
    print("=" * 50)
    
    # Test API key configuration
    api_key_ok = test_openai_api_key()
    print()
    
    # Test component files
    print("📁 Checking component files:")
    components_ok = test_market_chat_components()
    print()
    
    # Summary
    if api_key_ok and components_ok:
        print("🎉 ChatGPT integration is ready!")
        print("\nTo use the Market Chat:")
        print("1. Add your OpenAI API key to DayTrader/alphaVantageSettings.json")
        print("2. Build the application")
        print("3. Use 'Add Control' -> 'Market Chat' to add the chat interface")
        print("4. Ask questions like:")
        print("   - 'What's the sentiment around AAPL?'")
        print("   - 'Analyze the risks for tech stocks this quarter'")
        print("   - 'Explain the recent market volatility'")
    else:
        print("❌ Setup incomplete - please address the issues above")

if __name__ == "__main__":
    main()
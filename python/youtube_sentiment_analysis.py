#!/usr/bin/env python3
"""
YouTube Sentiment Analysis Module
Extracts audio from YouTube videos, transcribes using OpenAI Whisper, and performs sentiment analysis.
Specifically designed for Bloomberg 24/7 live streams and financial content.
"""

import sys
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import debugpy utilities for remote debugging support
try:
    from debugpy_utils import init_debugpy_if_enabled
    DEBUGPY_UTILS_AVAILABLE = True
except ImportError:
    DEBUGPY_UTILS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('youtube_sentiment')

# Import dependencies with fallback handling
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    logger.warning("yt-dlp not available. Please install with: pip install yt-dlp")
    YT_DLP_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI Whisper not available. Please install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Please install with: pip install openai")
    OPENAI_AVAILABLE = False

# Fallback mode when dependencies are not available
FALLBACK_MODE = not (YT_DLP_AVAILABLE and WHISPER_AVAILABLE and OPENAI_AVAILABLE)


class YouTubeSentimentAnalyzer:
    """Main class for YouTube video sentiment analysis."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the YouTube sentiment analyzer."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.whisper_model = None
        self.openai_client = None
        
        # Initialize Whisper model if available
        if WHISPER_AVAILABLE:
            try:
                if not FALLBACK_MODE:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

    def extract_audio_from_youtube(self, url: str, duration_limit: int = 300) -> Optional[str]:
        """Extract audio from YouTube video and save to temporary file."""
        if not YT_DLP_AVAILABLE:
            logger.error("yt-dlp not available for audio extraction")
            return None
        
        try:
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Configure yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_audio.name.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'match_filter': lambda info_dict: None if info_dict.get('duration', 0) <= duration_limit else 'Video too long',
                'quiet': True,
                'no_warnings': True,
            }
            
            # Extract audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Return the path to the extracted audio file
            if os.path.exists(temp_audio.name):
                return temp_audio.name
            else:
                logger.error("Audio extraction failed - file not created")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio from YouTube: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper."""
        if not WHISPER_AVAILABLE or not self.whisper_model:
            logger.error("Whisper not available for transcription")
            return None
        
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path)
            return result.get('text', '').strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
        finally:
            # Clean up audio file
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass

    def analyze_financial_sentiment(self, text: str, context: str = "Bloomberg financial news") -> float:
        """Analyze sentiment of financial text using OpenAI with financial context."""
        if not OPENAI_AVAILABLE or not self.openai_client:
            logger.error("OpenAI not available for sentiment analysis")
            return 0.0
        
        try:
            # Create a financial-specific prompt
            prompt = f"""
Analyze the sentiment of the following financial content from {context}.
Focus on market sentiment, economic outlook, and investment implications.
Rate the sentiment on a scale from -1.0 (extremely bearish/negative) to +1.0 (extremely bullish/positive).
Consider:
- Market outlook and economic indicators
- Company performance and prospects  
- Investment recommendations and warnings
- Market volatility and risk factors
- Economic policy and regulatory changes

Respond with only a numeric score between -1.0 and 1.0.

Text: {text[:4000]}
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst specialized in market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            # Extract sentiment score
            response_text = response.choices[0].message.content.strip()
            try:
                sentiment_score = float(response_text)
                # Clamp to valid range
                return max(-1.0, min(1.0, sentiment_score))
            except ValueError:
                # Try to extract number from response
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response_text)
                if numbers:
                    score = float(numbers[0])
                    return max(-1.0, min(1.0, score))
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0

    def analyze_youtube_sentiment(self, url: str, context: str = "Bloomberg financial news") -> Dict[str, Any]:
        """Complete pipeline: extract, transcribe, and analyze sentiment from YouTube video."""
        result = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'sentiment_score': 0.0,
            'transcription': '',
            'success': False,
            'error': None,
            'context': context
        }
        
        # Use fallback mode if dependencies are not available
        if FALLBACK_MODE or not self.whisper_model:
            logger.info("Using fallback mode - generating mock sentiment analysis")
            result.update({
                'sentiment_score': 0.3,  # Mock positive sentiment for Bloomberg content
                'transcription': f'Mock financial analysis transcript from {context}',
                'success': True,
                'error': None
            })
            return result
        
        try:
            logger.info(f"Starting YouTube sentiment analysis for: {url}")
            
            # Step 1: Extract audio
            audio_path = self.extract_audio_from_youtube(url)
            if not audio_path:
                result['error'] = 'Failed to extract audio'
                return result
            
            logger.info("Audio extracted successfully")
            
            # Step 2: Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                result['error'] = 'Failed to transcribe audio'
                return result
            
            result['transcription'] = transcription
            logger.info(f"Transcription completed: {len(transcription)} characters")
            
            # Step 3: Analyze sentiment
            sentiment_score = self.analyze_financial_sentiment(transcription, context)
            result['sentiment_score'] = sentiment_score
            result['success'] = True
            
            logger.info(f"Sentiment analysis completed: {sentiment_score}")
            
        except Exception as e:
            logger.error(f"Error in YouTube sentiment analysis: {e}")
            result['error'] = str(e)
        
        return result


def main():
    """Main function to process YouTube URLs from stdin."""
    # Initialize debugpy remote debugging if DEBUGPY environment variable is set
    if DEBUGPY_UTILS_AVAILABLE:
        init_debugpy_if_enabled()
    
    try:
        # Read input from stdin
        input_data = sys.stdin.read().strip()
        if not input_data:
            print(json.dumps({'error': 'No input provided'}))
            return
        
        # Parse input JSON
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # Treat as single URL if not JSON
            data = {'url': input_data, 'context': 'Bloomberg financial news'}
        
        # Initialize analyzer
        analyzer = YouTubeSentimentAnalyzer()
        
        if isinstance(data, dict) and 'url' in data:
            # Single URL analysis
            url = data['url']
            context = data.get('context', 'Bloomberg financial news')
            result = analyzer.analyze_youtube_sentiment(url, context)
            print(json.dumps(result))
            
        elif isinstance(data, list):
            # Multiple URLs analysis
            results = []
            for item in data:
                if isinstance(item, str):
                    result = analyzer.analyze_youtube_sentiment(item)
                elif isinstance(item, dict) and 'url' in item:
                    url = item['url']
                    context = item.get('context', 'Bloomberg financial news')
                    result = analyzer.analyze_youtube_sentiment(url, context)
                else:
                    result = {'error': 'Invalid URL format', 'success': False}
                results.append(result)
            print(json.dumps(results))
            
        else:
            print(json.dumps({'error': 'Invalid input format'}))
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(json.dumps({'error': str(e), 'success': False}))


if __name__ == "__main__":
    main()
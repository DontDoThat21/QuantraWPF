#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFT prediction script callable from C#.
Accepts JSON input with real historical sequences and calendar features.
Produces multi-horizon forecasts with uncertainty quantification.
"""

import sys
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tft_predict')

def main():
    if len(sys.argv) < 3:
        print("Usage: python tft_predict.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Load input
        with open(input_file, 'r') as f:
            request = json.load(f)
        
        logger.info("Processing TFT prediction request")
        
        # Extract data
        symbol = request['symbol']
        historical_sequence = request['historical_sequence']
        calendar_features = request['calendar_features']
        lookback_days = request.get('lookback_days', 60)
        future_horizon = request.get('future_horizon', 30)
        forecast_horizons = request.get('forecast_horizons', [5, 10, 20, 30])
        
        logger.info(f"Symbol: {symbol}, Historical days: {len(historical_sequence)}, " +
                   f"Calendar features: {len(calendar_features)}, Lookback: {lookback_days}")
        
        # Validate data
        if len(historical_sequence) < lookback_days:
            raise ValueError(f"Insufficient historical data: got {len(historical_sequence)}, need {lookback_days}")
        
        # Import TFT integration
        try:
            from tft_integration import TFTStockPredictor, create_static_features
            logger.info("TFT integration module loaded successfully")
        except ImportError as e:
            raise ImportError(f"TFT integration not available: {e}")
        
        # Load TFT model
        logger.info("Loading TFT model...")
        predictor = TFTStockPredictor(
            input_dim=50,  # Will be adjusted based on features
            static_dim=10,
            forecast_horizons=forecast_horizons
        )
        
        if not predictor.load():
            raise Exception("Failed to load TFT model. Train the model first using train_from_database.py")
        
        logger.info(f"TFT model loaded successfully. Forecasting for horizons: {forecast_horizons}")
        
        # Make prediction with real historical sequences
        result = predictor.predict_single(
            historical_sequence=historical_sequence,
            calendar_features=calendar_features,
            static_dict=None  # TODO: Add static features from C# (sector, market_cap, etc.)
        )
        
        logger.info(f"TFT prediction complete: {result.get('action')} with confidence {result.get('confidence', 0):.2%}")
        
        # Write output
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results written to {output_file}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"TFT prediction failed: {str(e)}", exc_info=True)
        error_result = {
            'symbol': request.get('symbol', 'UNKNOWN') if 'request' in locals() else 'UNKNOWN',
            'action': 'HOLD',
            'confidence': 0.5,
            'targetPrice': 0.0,
            'error': str(e),
            'success': False
        }
        with open(output_file, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the market anomaly detection module.
This script generates synthetic market data with injected anomalies
and tests the anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime, timedelta

# Import the anomaly detection module
import anomaly_detection

def generate_synthetic_market_data(days=500, with_anomalies=True):
    """
    Generate synthetic market data with optional anomalies.
    
    Args:
        days (int): Number of days of data to generate
        with_anomalies (bool): Whether to inject anomalies
        
    Returns:
        dict: Dictionary with OHLCV data
    """
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price data with realistic properties
    # Start with a random walk
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, 0.015, days)  # Mean positive drift
    
    # Introduce some autocorrelation
    for i in range(1, len(returns)):
        returns[i] = 0.2 * returns[i-1] + 0.8 * returns[i]
    
    # Generate price from returns
    close = 100.0  # Starting price
    closes = [close]
    for ret in returns[:-1]:
        close = close * (1 + ret)
        closes.append(close)
    
    # Generate realistic OHLCV data
    highs = []
    lows = []
    opens = []
    volumes = []
    
    for i, close in enumerate(closes):
        # First price is the same for all
        if i == 0:
            opens.append(close)
            highs.append(close * 1.005)
            lows.append(close * 0.995)
            volumes.append(1000000)
            continue
            
        prev_close = closes[i-1]
        
        # Open is close +/- small random move
        open_change = np.random.normal(0, 0.002)
        open_price = prev_close * (1 + open_change)
        opens.append(open_price)
        
        # High is max of open and close plus a random amount
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
        highs.append(high_price)
        
        # Low is min of open and close minus a random amount
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
        lows.append(low_price)
        
        # Volume is based on price volatility
        price_change = abs(close / prev_close - 1)
        volume = 1000000 * (1 + 10 * price_change) * np.random.uniform(0.8, 1.2)
        volumes.append(volume)
    
    # Inject anomalies if requested
    if with_anomalies:
        # Inject price spike anomalies
        for _ in range(5):
            idx = np.random.randint(10, days-10)
            direction = np.random.choice([-1, 1])
            spike_size = np.random.uniform(0.05, 0.15)  # 5-15% price spike
            
            closes[idx] = closes[idx-1] * (1 + direction * spike_size)
            opens[idx] = closes[idx-1] * (1 + direction * spike_size * 0.5)
            highs[idx] = max(opens[idx], closes[idx]) * 1.01
            lows[idx] = min(opens[idx], closes[idx]) * 0.99
            
            # Usually volume spikes with price anomalies
            volumes[idx] = volumes[idx-1] * np.random.uniform(2, 5)
        
        # Inject volatility anomalies
        for _ in range(3):
            idx = np.random.randint(50, days-50)
            for j in range(idx, idx + 10):
                # Increase volatility
                if j < days:
                    volatility_mult = np.random.uniform(1.5, 3)
                    price_change = returns[j] * volatility_mult
                    closes[j] = closes[j-1] * (1 + price_change)
                    spread = abs(closes[j] - closes[j-1]) * 0.5
                    highs[j] = max(closes[j], closes[j-1]) + spread
                    lows[j] = min(closes[j], closes[j-1]) - spread
                    volumes[j] = volumes[j] * np.random.uniform(1.2, 2.0)
        
        # Inject volume anomalies without price changes
        for _ in range(3):
            idx = np.random.randint(10, days-10)
            volumes[idx] = volumes[idx] * np.random.uniform(3, 8)
    
    # Create a dictionary with the synthetic data
    data = {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    return data

def plot_anomalies(data, anomaly_result):
    """
    Plot the market data with detected anomalies highlighted.
    
    Args:
        data (dict): Market data dictionary
        anomaly_result (dict): Result from anomaly detection
    """
    try:
        import matplotlib.pyplot as plt

        # Create a figure with price and volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Format dates for x-axis
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['dates']]
        
        # Plot price data
        ax1.plot(dates, data['close'], label='Close Price')
        
        # Highlight anomalies if any were detected
        if anomaly_result['anomalies_detected']:
            anomaly_indices = anomaly_result['anomaly_indices']
            anomaly_dates = [dates[i] for i in anomaly_indices if i < len(dates)]
            anomaly_prices = [data['close'][i] for i in anomaly_indices if i < len(data['close'])]
            
            # Mark anomalies with red dots
            ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=50, label='Anomalies')
            
            # Annotate the type of anomaly
            for i, idx in enumerate(anomaly_indices):
                if idx < len(dates):
                    anomaly_types = anomaly_result['anomaly_types'].get(str(idx), [])
                    if not anomaly_types and idx in anomaly_result['anomaly_types']:
                        anomaly_types = anomaly_result['anomaly_types'][idx]
                    
                    if anomaly_types:
                        # Get the first anomaly type for annotation
                        atype = anomaly_types[0].split('_')[0] if isinstance(anomaly_types, list) else "ANOMALY"
                        ax1.annotate(
                            atype, 
                            (dates[idx], data['close'][idx]),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center'
                        )
        
        # Plot volume in the lower subplot
        ax2.bar(dates, data['volume'], color='gray', alpha=0.5)
        
        # Highlight volume anomalies
        if anomaly_result['anomalies_detected']:
            for idx in anomaly_indices:
                if idx < len(dates) and idx < len(data['volume']):
                    anomaly_types = anomaly_result['anomaly_types'].get(str(idx), [])
                    if not anomaly_types and idx in anomaly_result['anomaly_types']:
                        anomaly_types = anomaly_result['anomaly_types'][idx]
                    
                    if "VOLUME_ANOMALY" in anomaly_types:
                        ax2.bar(dates[idx], data['volume'][idx], color='red', alpha=0.7)
        
        # Format the plot
        ax1.set_title('Market Data with Detected Anomalies')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Show only a few x labels to avoid crowding
        for ax in [ax1, ax2]:
            if len(dates) > 30:
                step = len(dates) // 10
                ax.set_xticks(dates[::step])
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_test.png')
        print("Plot saved as 'anomaly_detection_test.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plot")
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

def main():
    """Main function to test the anomaly detection module"""
    print("Generating synthetic market data...")
    data = generate_synthetic_market_data(days=500, with_anomalies=True)
    
    print("Running anomaly detection...")
    result = anomaly_detection.detect_market_anomalies(data, use_feature_engineering=True, sensitivity=1.0)
    
    # Print results
    print("\nAnomaly Detection Results:")
    print(f"Anomalies detected: {result['anomalies_detected']}")
    print(f"Number of anomalies: {result.get('anomaly_count', 0)}")
    
    # Print indices and types of detected anomalies
    if result['anomalies_detected']:
        print("\nDetected anomalies:")
        for idx in result['anomaly_indices'][:10]:  # Show first 10
            types = result['anomaly_types'].get(str(idx), [])
            if not types and idx in result['anomaly_types']:
                types = result['anomaly_types'][idx]
                
            date = data['dates'][idx] if idx < len(data['dates']) else "unknown"
            print(f"  {date} (index {idx}): {types}")
        
        if len(result['anomaly_indices']) > 10:
            print(f"  ... and {len(result['anomaly_indices']) - 10} more")
            
        # Plot the results
        plot_anomalies(data, result)
        
        # Save results to JSON
        with open('anomaly_detection_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\nDetailed results saved to 'anomaly_detection_results.json'")
    
if __name__ == "__main__":
    main()
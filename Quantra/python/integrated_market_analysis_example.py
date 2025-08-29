#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Market Analysis Example

This script demonstrates how to combine the market regime detection and
anomaly detection modules to provide comprehensive market analysis.
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import market analysis modules
import market_regime_detection
import anomaly_detection

def generate_sample_market_data(days=200):
    """Generate sample market data for demonstration"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price data with realistic properties
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, 0.012, days)  # Mean positive drift
    
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

def analyze_market(market_data):
    """
    Perform comprehensive market analysis using multiple ML techniques.
    
    Args:
        market_data (dict): Dictionary containing market data
        
    Returns:
        dict: Analysis results
    """
    print("Analyzing market regime...")
    regime_result = market_regime_detection.detect_market_regime(market_data)
    
    print("Detecting market anomalies...")
    anomaly_result = anomaly_detection.detect_market_anomalies(market_data)
    
    # Combine the results into a comprehensive market analysis
    combined_analysis = {
        "market_regime": {
            "current_regime": regime_result["regime"],
            "regime_confidence": regime_result["confidence"],
            "recommended_approaches": regime_result.get("tradingApproaches", []),
        },
        "anomalies": {
            "anomalies_detected": anomaly_result["anomalies_detected"],
            "anomaly_count": anomaly_result.get("anomaly_count", 0),
            "recent_anomalies": anomaly_result.get("recent_anomalies", [])
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add additional risk metrics based on both analyses
    risk_level = "LOW"  # Default
    
    # Increase risk level if we're in a trending down regime
    if regime_result["regime"] == market_regime_detection.REGIME_TRENDING_DOWN:
        risk_level = "MEDIUM"
        
    # Further increase risk if anomalies are detected
    if anomaly_result["anomalies_detected"]:
        if risk_level == "MEDIUM":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    # Final risk assessment
    combined_analysis["risk_assessment"] = {
        "risk_level": risk_level,
        "concerns": []
    }
    
    # Add specific concerns based on detected issues
    if regime_result["regime"] == market_regime_detection.REGIME_TRENDING_DOWN:
        combined_analysis["risk_assessment"]["concerns"].append(
            "Market in downtrend regime - consider defensive positioning"
        )
        
    if anomaly_result["anomalies_detected"]:
        for anomaly in anomaly_result.get("recent_anomalies", [])[:3]:  # Top 3 recent anomalies
            if "insights" in anomaly and "description" in anomaly["insights"]:
                for desc in anomaly["insights"]["description"]:
                    combined_analysis["risk_assessment"]["concerns"].append(desc)
            else:
                combined_analysis["risk_assessment"]["concerns"].append(
                    f"Anomaly detected on {anomaly.get('date', 'recent date')}"
                )
    
    # Generate strategic recommendations based on combined analysis
    recommendations = []
    
    if regime_result["regime"] == market_regime_detection.REGIME_TRENDING_UP and not anomaly_result["anomalies_detected"]:
        recommendations.append("Market in healthy uptrend - consider trend-following strategies")
        recommendations.append("Focus on momentum-based entries with trailing stops")
    elif regime_result["regime"] == market_regime_detection.REGIME_TRENDING_UP and anomaly_result["anomalies_detected"]:
        recommendations.append("Uptrend with anomalies detected - use caution with trend-following approaches")
        recommendations.append("Consider tightening stops and reducing position sizes")
    elif regime_result["regime"] == market_regime_detection.REGIME_TRENDING_DOWN:
        recommendations.append("Downtrend detected - consider defensive positioning or counter-trend strategies")
        recommendations.append("Focus on capital preservation and hedging strategies")
    elif regime_result["regime"] == market_regime_detection.REGIME_RANGING:
        recommendations.append("Ranging market detected - consider mean-reversion strategies")
        recommendations.append("Look for overbought/oversold conditions at range boundaries")
    
    # Add anomaly-specific recommendations
    if anomaly_result["anomalies_detected"]:
        recommendations.append("Market anomalies detected - increase risk management vigilance")
        
        # Add more specific recommendations from anomaly insights
        for anomaly in anomaly_result.get("recent_anomalies", [])[:2]:
            if "insights" in anomaly and "suggested_actions" in anomaly["insights"]:
                recommendations.extend(anomaly["insights"]["suggested_actions"][:2])
    
    combined_analysis["recommendations"] = recommendations
    
    return combined_analysis

def main():
    """Main function to demonstrate integrated market analysis"""
    try:
        # Check if data file is provided as argument
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            with open(input_file, 'r') as f:
                market_data = json.load(f)
        else:
            print("Generating sample market data...")
            market_data = generate_sample_market_data()
        
        print("Performing integrated market analysis...")
        result = analyze_market(market_data)
        
        # Print summary of results
        print("\n=== MARKET ANALYSIS SUMMARY ===")
        print(f"Current Market Regime: {result['market_regime']['current_regime']}")
        print(f"Regime Confidence: {result['market_regime']['regime_confidence']:.2f}")
        print(f"Anomalies Detected: {result['anomalies']['anomalies_detected']}")
        print(f"Risk Assessment: {result['risk_assessment']['risk_level']}")
        
        if result['risk_assessment']['concerns']:
            print("\nRisk Concerns:")
            for concern in result['risk_assessment']['concerns']:
                print(f"- {concern}")
        
        print("\nStrategic Recommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
        
        # Save results to file
        output_file = "integrated_market_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nDetailed analysis saved to {output_file}")
        
    except Exception as e:
        print(f"Error in market analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
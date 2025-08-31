using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    public class BreadthThrustIndicator
    {
        private readonly List<double> advances;
        private readonly List<double> declines;
        private readonly List<double> thrustValues;
        private readonly List<double> ratioValues;
        private readonly int period;

        /// <summary>
        /// Initializes a new instance of the BreadthThrustIndicator.
        /// </summary>
        /// <param name="period">The lookback period for calculating the indicator, typically 10 days.</param>
        public BreadthThrustIndicator(int period = 10)
        {
            this.period = period;
            advances = new List<double>();
            declines = new List<double>();
            thrustValues = new List<double>();
            ratioValues = new List<double>();
        }

        /// <summary>
        /// Adds a new data point to the indicator.
        /// </summary>
        /// <param name="advancingIssues">Number of advancing issues for the period.</param>
        /// <param name="decliningIssues">Number of declining issues for the period.</param>
        public void AddDataPoint(int advancingIssues, int decliningIssues)
        {
            advances.Add(advancingIssues);
            declines.Add(decliningIssues);
            
            // Calculate the Breadth Thrust value
            CalculateBreadthThrust();
        }

        /// <summary>
        /// Calculate the Breadth Thrust indicator based on the available data.
        /// Breadth Thrust is the ratio of advancing issues to total issues (advancing + declining),
        /// typically averaged over a specific period.
        /// </summary>
        private void CalculateBreadthThrust()
        {
            // Need at least one data point
            if (advances.Count == 0 || declines.Count == 0)
                return;

            // Calculate the current ratio
            double currentAdvances = advances.Last();
            double currentDeclines = declines.Last();
            double totalIssues = currentAdvances + currentDeclines;
            
            // Avoid division by zero
            if (totalIssues > 0)
            {
                double ratio = currentAdvances / totalIssues;
                ratioValues.Add(ratio);
                
                // If we have enough data, calculate the Breadth Thrust value (typically a 10-day moving average)
                if (ratioValues.Count >= period)
                {
                    double thrustValue = ratioValues.TakeLast(period).Average();
                    thrustValues.Add(thrustValue);
                }
                else
                {
                    // If we don't have enough data yet, just use what we have
                    double thrustValue = ratioValues.Average();
                    thrustValues.Add(thrustValue);
                }
            }
            else
            {
                // If there are no issues recorded, add a neutral value
                ratioValues.Add(0.5);
                thrustValues.Add(0.5);
            }
        }

        /// <summary>
        /// Gets the current Breadth Thrust value.
        /// </summary>
        /// <returns>The current Breadth Thrust value.</returns>
        public double GetCurrentValue()
        {
            return thrustValues.Count > 0 ? thrustValues.Last() : 0;
        }

        /// <summary>
        /// Gets the current ratio of advancing issues to total issues.
        /// </summary>
        /// <returns>The current ratio value.</returns>
        public double GetCurrentRatio()
        {
            return ratioValues.Count > 0 ? ratioValues.Last() : 0;
        }

        /// <summary>
        /// Gets all calculated Breadth Thrust values.
        /// </summary>
        /// <returns>A list of all Breadth Thrust values.</returns>
        public List<double> GetAllValues()
        {
            return new List<double>(thrustValues);
        }

        /// <summary>
        /// Generates a signal based on the current Breadth Thrust value.
        /// </summary>
        /// <returns>A signal string ("Strong Buy", "Buy", "Neutral", "Sell").</returns>
        public string GetSignal()
        {
            if (thrustValues.Count == 0)
                return "Neutral";

            double currentValue = thrustValues.Last();

            // Martin Zweig's original Breadth Thrust theory used 1.05 as a threshold for a buy signal
            if (currentValue > 1.05)
                return "Strong Buy";
            else if (currentValue > 0.95)
                return "Buy";
            else if (currentValue < 0.75)
                return "Sell";
            else
                return "Neutral";
        }

        /// <summary>
        /// Simulates Breadth Thrust data for testing and demonstration purposes.
        /// </summary>
        /// <param name="dataPoints">Number of data points to generate.</param>
        /// <param name="trendBias">Bias towards positive (>0) or negative (<0) trend.</param>
        /// <returns>A BreadthThrustIndicator with simulated data.</returns>
        public static BreadthThrustIndicator GenerateSimulatedData(int dataPoints = 30, double trendBias = 0.1)
        {
            var indicator = new BreadthThrustIndicator();
            Random random = new Random();

            double trend = 0.5; // Start at neutral
            
            for (int i = 0; i < dataPoints; i++)
            {
                // Apply random walk with bias
                trend += (random.NextDouble() - 0.5) * 0.1 + trendBias * 0.01;
                
                // Ensure trend stays between 0.1 and 0.9 for realistic simulation
                trend = Math.Max(0.1, Math.Min(0.9, trend));
                
                // Calculate simulated advancing and declining issues
                int totalIssues = 500; // Typical number for a major index
                int advancing = (int)(totalIssues * trend);
                int declining = totalIssues - advancing;
                
                // Add the simulated data point
                indicator.AddDataPoint(advancing, declining);
            }
            
            return indicator;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.Models.Examples
{
    /// <summary>
    /// Example indicator that calculates the average of two other indicators
    /// </summary>
    public class AverageIndicator : CompositeIndicator
    {
        /// <summary>
        /// Creates a new average indicator
        /// </summary>
        /// <param name="firstIndicatorId">ID of the first indicator</param>
        /// <param name="secondIndicatorId">ID of the second indicator</param>
        /// <param name="weightFirst">Weight of the first indicator (0-1)</param>
        /// <param name="name">Name for this indicator</param>
        public AverageIndicator(
            string firstIndicatorId,
            string secondIndicatorId,
            double weightFirst = 0.5,
            string name = "Weighted Average")
            : base(
                name,
                (inputs) => CalculateWeightedAverage(inputs, firstIndicatorId, secondIndicatorId, weightFirst),
                new List<string> { firstIndicatorId, secondIndicatorId },
                "Value",
                "Composite",
                "Calculates a weighted average of two indicators")
        {
            // Register parameters
            RegisterParameter("WeightFirst", "Weight of the first indicator", weightFirst, 0.0, 1.0);
        }

        /// <summary>
        /// Static calculation method for the weighted average
        /// </summary>
        private static double CalculateWeightedAverage(
            Dictionary<string, Dictionary<string, double>> inputs,
            string firstIndicatorId,
            string secondIndicatorId,
            double weightFirst)
        {
            // Get input values
            if (!inputs.TryGetValue(firstIndicatorId, out var firstValues) ||
                !inputs.TryGetValue(secondIndicatorId, out var secondValues))
            {
                throw new InvalidOperationException("Input indicators not found");
            }

            // Get primary values
            double firstValue = firstValues.Values.FirstOrDefault();
            double secondValue = secondValues.Values.FirstOrDefault();

            // Calculate weighted average
            return (firstValue * weightFirst) + (secondValue * (1 - weightFirst));
        }
    }
}
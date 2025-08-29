using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.Models.Examples
{
    /// <summary>
    /// Example indicator that calculates the relative strength of one indicator compared to another
    /// </summary>
    public class RelativeStrengthIndicator : CompositeIndicator
    {
        /// <summary>
        /// Creates a new relative strength indicator
        /// </summary>
        /// <param name="primaryIndicatorId">ID of the primary indicator</param>
        /// <param name="referenceIndicatorId">ID of the reference indicator</param>
        /// <param name="name">Name for this indicator</param>
        public RelativeStrengthIndicator(
            string primaryIndicatorId,
            string referenceIndicatorId,
            string name = "Relative Strength")
            : base(
                name,
                (inputs) => CalculateRelativeStrength(inputs, primaryIndicatorId, referenceIndicatorId),
                new List<string> { primaryIndicatorId, referenceIndicatorId },
                "Value",
                "Composite",
                "Calculates the relative strength of one indicator compared to another")
        {
            // Register parameters
            RegisterParameter("SmoothingPeriod", "Period for smoothing the result", 3, 1, 20);
        }

        /// <summary>
        /// Static calculation method for the relative strength
        /// </summary>
        private static double CalculateRelativeStrength(
            Dictionary<string, Dictionary<string, double>> inputs,
            string primaryIndicatorId,
            string referenceIndicatorId)
        {
            // Get input values
            if (!inputs.TryGetValue(primaryIndicatorId, out var primaryValues) ||
                !inputs.TryGetValue(referenceIndicatorId, out var referenceValues))
            {
                throw new InvalidOperationException("Input indicators not found");
            }

            // Get primary values
            double primaryValue = primaryValues.Values.FirstOrDefault();
            double referenceValue = referenceValues.Values.FirstOrDefault();

            // Avoid division by zero
            if (Math.Abs(referenceValue) < 0.000001)
            {
                return primaryValue > 0 ? 100 : -100; // Arbitrary large value
            }

            // Calculate relative strength
            return (primaryValue / referenceValue) * 100;
        }
    }
}
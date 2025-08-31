using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IPortfolioManagementService
    {
        /// <summary>
        /// Sets portfolio target allocations
        /// </summary>
        bool SetPortfolioAllocations(Dictionary<string, double> allocations);
        
        /// <summary>
        /// Gets current portfolio allocations
        /// </summary>
        Task<Dictionary<string, double>> GetCurrentPortfolioAllocations();
        
        /// <summary>
        /// Basic portfolio rebalancing without advanced profile features
        /// </summary>
        Task<bool> RebalancePortfolioBasic(double tolerancePercentage = 0.02);
        
        /// <summary>
        /// Rebalances the portfolio to match target allocations
        /// </summary>
        Task<bool> RebalancePortfolio(double tolerancePercentage = 0.02);
        
        /// <summary>
        /// Rebalances the portfolio using a specific profile with advanced settings
        /// </summary>
        Task<bool> RebalancePortfolioWithProfile(string profileId, double tolerancePercentage = 0.02);
        
        /// <summary>
        /// Creates a new rebalancing profile
        /// </summary>
        bool CreateRebalancingProfile(RebalancingProfile profile);
        
        /// <summary>
        /// Gets all available rebalancing profiles
        /// </summary>
        List<RebalancingProfile> GetRebalancingProfiles();
        
        /// <summary>
        /// Sets the active rebalancing profile
        /// </summary>
        bool SetActiveRebalancingProfile(string profileId);
        
        /// <summary>
        /// Gets the current active rebalancing profile
        /// </summary>
        RebalancingProfile GetActiveRebalancingProfile();
    }
}
using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Services
{
    /// <summary>
    /// Manages strategy profiles and selection per PAC (Prediction Analysis Control)
    /// </summary>
    public class StrategyProfileManager
    {
        // Singleton instance
        private static StrategyProfileManager _instance;
        public static StrategyProfileManager Instance => _instance ??= new StrategyProfileManager();

        // All available strategy profiles (could be loaded from DB or config)
        private readonly List<StrategyProfile> _profiles = new List<StrategyProfile>();

        // Mapping: PAC identifier (string or int) -> selected profile name
        private readonly Dictionary<string, string> _pacProfileSelections = new();

        public IReadOnlyList<StrategyProfile> Profiles => _profiles;

        private StrategyProfileManager()
        {
            // Register built-in strategies (expand as needed)
            _profiles.Add(new SmaCrossoverStrategy());
            // Add more strategies here
        }

        public void RegisterProfile(StrategyProfile profile)
        {
            if (_profiles.All(p => p.Name != profile.Name))
                _profiles.Add(profile);
        }

        public void SetProfileForPac(string pacId, string profileName)
        {
            if (_profiles.Any(p => p.Name == profileName))
                _pacProfileSelections[pacId] = profileName;
        }

        public StrategyProfile GetProfileForPac(string pacId)
        {
            if (_pacProfileSelections.TryGetValue(pacId, out var name))
                return _profiles.FirstOrDefault(p => p.Name == name);
            return _profiles.FirstOrDefault(); // Default to first
        }

        public IEnumerable<string> GetProfileNames() => _profiles.Select(p => p.Name);
    }
}

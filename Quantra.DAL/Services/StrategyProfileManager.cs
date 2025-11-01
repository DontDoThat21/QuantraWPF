using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Quantra.Models;
using Quantra;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Singleton manager for trading strategy profiles
    /// </summary>
    public class StrategyProfileManager
    {
        private static readonly string ProfilesPath = "strategyprofiles.json";
        private static readonly string BackupPath = "strategyprofiles.backup.json";
        private static readonly int CurrentVersion = 1;
        private static readonly Lazy<StrategyProfileManager> _instance = new Lazy<StrategyProfileManager>(() => new StrategyProfileManager());
        
        /// <summary>
        /// The singleton instance of the strategy profile manager
        /// </summary>
        public static StrategyProfileManager Instance => _instance.Value;

        private readonly Dictionary<string, TradingStrategyProfile> _profiles = new Dictionary<string, TradingStrategyProfile>();
        private readonly Dictionary<string, string> _pacProfileMapping = new Dictionary<string, string>();

        /// <summary>
        /// Private constructor to enforce singleton pattern
        /// </summary>
        private StrategyProfileManager()
        {
            LoadDefaultProfiles();
            LoadProfiles();
        }

        /// <summary>
        /// Get all available strategy profiles
        /// </summary>
        /// <returns>List of strategy profiles</returns>
        public IReadOnlyList<TradingStrategyProfile> GetProfiles()
        {
            return _profiles.Values.ToList();
        }

        /// <summary>
        /// Get all available strategy profile names
        /// </summary>
        /// <returns>List of strategy profile names</returns>
        public IReadOnlyList<string> GetProfileNames()
        {
            return _profiles.Keys.ToList();
        }

        /// <summary>
        /// Get a strategy profile by name
        /// </summary>
        /// <param name="profileName">Name of the profile to retrieve</param>
        /// <returns>The strategy profile, or null if not found</returns>
        public TradingStrategyProfile GetProfile(string profileName)
        {
            if (string.IsNullOrEmpty(profileName)) 
                return GetDefaultProfile();

            if (_profiles.TryGetValue(profileName, out TradingStrategyProfile profile))
                return profile;

            return GetDefaultProfile();
        }

        /// <summary>
        /// Validates a trading strategy profile
        /// </summary>
        /// <param name="profile">Profile to validate</param>
        /// <returns>True if valid, false otherwise</returns>
        public bool ValidateProfile(TradingStrategyProfile profile)
        {
            if (profile == null || string.IsNullOrEmpty(profile.Name))
                return false;

            // Basic validation criteria
            if (profile.MinConfidence < 0 || profile.MinConfidence > 1)
                return false;

            if (profile.RiskLevel < 0 || profile.RiskLevel > 1)
                return false;

            return true;
        }

        /// <summary>
        /// Add or update a strategy profile with validation
        /// </summary>
        /// <param name="profile">Profile to add or update</param>
        public void SaveProfile(TradingStrategyProfile profile)
        {
            if (!ValidateProfile(profile))
                throw new ArgumentException("Invalid profile configuration");

            if (profile == null || string.IsNullOrEmpty(profile.Name))
                return;

            _profiles[profile.Name] = profile;
            SaveProfiles();
            BackupProfiles(); // Create backup after successful save
        }

        /// <summary>
        /// Creates a backup of current profiles
        /// </summary>
        public void BackupProfiles()
        {
            try
            {
                if (File.Exists(ProfilesPath))
                    File.Copy(ProfilesPath, BackupPath, true);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to backup strategy profiles", ex.ToString());
            }
        }

        /// <summary>
        /// Restores profiles from backup
        /// </summary>
        /// <returns>True if restore was successful, false otherwise</returns>
        public bool RestoreFromBackup()
        {
            try
            {
                if (File.Exists(BackupPath))
                {
                    File.Copy(BackupPath, ProfilesPath, true);
                    LoadProfiles();
                    return true;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to restore strategy profiles from backup", ex.ToString());
            }
            return false;
        }

        /// <summary>
        /// Export profiles to a specified file path
        /// </summary>
        /// <param name="filePath">The file path to export profiles to</param>
        /// <returns>True if export was successful, false otherwise</returns>
        public bool ExportProfiles(string filePath)
        {
            try
            {
                var data = new Dictionary<string, object>
                {
                    ["Version"] = CurrentVersion,
                    ["Profiles"] = _profiles.Values.Select(p => JObject.FromObject(p)).ToList(),
                    ["PacMappings"] = _pacProfileMapping
                };

                string json = JsonConvert.SerializeObject(data, Formatting.Indented);
                File.WriteAllText(filePath, json);
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to export strategy profiles", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Import profiles from a specified file path
        /// </summary>
        /// <param name="filePath">The file path to import profiles from</param>
        /// <param name="merge">Whether to merge with existing profiles or replace them</param>
        /// <returns>True if import was successful, false otherwise</returns>
        public bool ImportProfiles(string filePath, bool merge = false)
        {
            try
            {
                string json = File.ReadAllText(filePath);
                var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);

                if (!data.TryGetValue("Version", out object versionObj))
                    return false;

                int version = Convert.ToInt32(versionObj);
                if (version > CurrentVersion)
                    return false;

                if (!merge)
                    _profiles.Clear();

                if (data.TryGetValue("Profiles", out object profilesObj) && 
                    profilesObj is JArray profileArray)
                {
                    foreach (JObject profileObj in profileArray)
                    {
                        string type = profileObj["Type"]?.ToString();
                        string name = profileObj["Name"]?.ToString();
                        
                        if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(type))
                            continue;

                        TradingStrategyProfile profile = DeserializeProfile(type, profileObj);
                        if (profile != null && ValidateProfile(profile))
                            _profiles[name] = profile;
                    }
                }

                SaveProfiles();
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to import strategy profiles", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Delete a strategy profile by name
        /// </summary>
        /// <param name="profileName">Name of the profile to delete</param>
        /// <returns>True if deleted, false if not found</returns>
        public bool DeleteProfile(string profileName)
        {
            if (string.IsNullOrEmpty(profileName))
                return false;

            if (_profiles.Remove(profileName))
            {
                // Remove any PAC mappings using this profile
                var mappingsToRemove = _pacProfileMapping.Where(m => m.Value == profileName).Select(m => m.Key).ToList();
                foreach (var pacId in mappingsToRemove)
                    _pacProfileMapping.Remove(pacId);

                SaveProfiles();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Clone an existing profile with a new name
        /// </summary>
        /// <param name="sourceProfileName">Name of the source profile to clone</param>
        /// <param name="newProfileName">Name of the new profile</param>
        /// <returns>The cloned profile, or null if cloning failed</returns>
        public TradingStrategyProfile CloneProfile(string sourceProfileName, string newProfileName)
        {
            if (string.IsNullOrEmpty(sourceProfileName) || string.IsNullOrEmpty(newProfileName))
                return null;

            if (!_profiles.TryGetValue(sourceProfileName, out TradingStrategyProfile sourceProfile))
                return null;

            // Serialize and deserialize to create deep clone
            var json = JObject.FromObject(sourceProfile);
            json["Name"] = newProfileName;
            var clonedProfile = DeserializeProfile(sourceProfile.GetType().Name, json);

            if (clonedProfile != null)
            {
                SaveProfile(clonedProfile);
                return clonedProfile;
            }

            return null;
        }

        /// <summary>
        /// Sets the active strategy profile for a specific PAC (Prediction Analysis Control) instance
        /// </summary>
        /// <param name="pacId">Unique ID of the PAC instance</param>
        /// <param name="profileName">Name of the profile to set</param>
        public void SetProfileForPac(string pacId, string profileName)
        {
            if (string.IsNullOrEmpty(pacId))
                return;

            if (string.IsNullOrEmpty(profileName))
                _pacProfileMapping.Remove(pacId);
            else
                _pacProfileMapping[pacId] = profileName;

            SaveProfiles();
        }

        /// <summary>
        /// Gets the active strategy profile for a specific PAC (Prediction Analysis Control) instance
        /// </summary>
        /// <param name="pacId">Unique ID of the PAC instance</param>
        /// <returns>The active strategy profile for this PAC, or the default profile if none set</returns>
        public TradingStrategyProfile GetProfileForPac(string pacId)
        {
            if (_pacProfileMapping.TryGetValue(pacId, out string profileName))
                return GetProfile(profileName);

            return GetDefaultProfile();
        }

        /// <summary>
        /// Get the default strategy profile (RSI Divergence Strategy)
        /// </summary>
        /// <returns>The default strategy profile</returns>
        public TradingStrategyProfile GetDefaultProfile()
        {
            // Try to get the RSI Divergence Strategy first
            if (_profiles.TryGetValue("RSI Divergence", out TradingStrategyProfile rsiStrategy))
                return rsiStrategy;

            // Otherwise return the first available profile
            return _profiles.Values.FirstOrDefault();
        }

        /// <summary>
        /// Load strategy profiles from disk
        /// </summary>
        private void LoadProfiles()
        {
            try
            {
                if (File.Exists(ProfilesPath))
                {
                    string json = File.ReadAllText(ProfilesPath);
                    var profileData = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);

                    if (profileData.TryGetValue("Profiles", out object profilesObj) && 
                        profilesObj is JArray profileArray)
                    {
                        foreach (JObject profileObj in profileArray)
                        {
                            string type = profileObj["Type"]?.ToString();
                            string name = profileObj["Name"]?.ToString();
                            
                            if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(type))
                                continue;

                            TradingStrategyProfile profile = DeserializeProfile(type, profileObj);
                            if (profile != null && ValidateProfile(profile))
                                _profiles[name] = profile;
                        }
                    }

                    if (profileData.TryGetValue("PacMappings", out object mappingsObj) && 
                        mappingsObj is JObject mappingObj)
                    {
                        foreach (var prop in mappingObj.Properties())
                        {
                            string pacId = prop.Name;
                            string profileName = prop.Value.ToString();
                            if (!string.IsNullOrEmpty(pacId) && !string.IsNullOrEmpty(profileName))
                            {
                                _pacProfileMapping[pacId] = profileName;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load strategy profiles", ex.ToString());
                LoadDefaultProfiles(); // Fallback to defaults
            }
        }

        /// <summary>
        /// Save profiles to disk
        /// </summary>
        private void SaveProfiles()
        {
            try
            {
                var profilesList = new List<object>();
                foreach (var profile in _profiles.Values)
                {
                    string type = profile.GetType().Name;
                    var profileJson = JObject.FromObject(profile);
                    profileJson["Type"] = type;
                    
                    // Special handling for aggregated strategies
                    if (profile is AggregatedStrategyProfile aggregatedProfile)
                    {
                        // Replace strategy objects with references to their names
                        var strategiesArray = new JArray();
                        foreach (var strategyWeight in aggregatedProfile.Strategies)
                        {
                            var strategyObj = new JObject
                            {
                                ["StrategyName"] = strategyWeight.Strategy.Name,
                                ["Weight"] = strategyWeight.Weight
                            };
                            strategiesArray.Add(strategyObj);
                        }
                        profileJson["Strategies"] = strategiesArray;
                    }
                    
                    profilesList.Add(profileJson);
                }

                var data = new Dictionary<string, object>
                {
                    ["Version"] = CurrentVersion,
                    ["Profiles"] = profilesList,
                    ["PacMappings"] = _pacProfileMapping
                };

                string json = JsonConvert.SerializeObject(data, Formatting.Indented);
                File.WriteAllText(ProfilesPath, json);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save strategy profiles", ex.ToString());
            }
        }

        /// <summary>
        /// Load default strategy profiles
        /// </summary>
        private void LoadDefaultProfiles()
        {
            _profiles.Clear();

            // Add default strategies
            var rsiStrategy = new RsiDivergenceStrategy();
            _profiles[rsiStrategy.Name] = rsiStrategy;

            var smaStrategy = new Quantra.Models.SmaCrossoverStrategy();
            _profiles[smaStrategy.Name] = smaStrategy;

            var bollingerStrategy = new BollingerBandsMeanReversionStrategy();
            _profiles[bollingerStrategy.Name] = bollingerStrategy;

            var macdStrategy = new MacdCrossoverStrategy();
            _profiles[macdStrategy.Name] = macdStrategy;
            

            // Add Support/Resistance strategy
            var supportResistanceStrategy = new SupportResistanceStrategy();
            _profiles[supportResistanceStrategy.Name] = supportResistanceStrategy;
            // Create a default aggregated strategy as an example
            var aggregatedStrategy = new AggregatedStrategyProfile
            {
                Name = "Combined Strategy (RSI+MACD)",
                Description = "Example aggregated strategy combining RSI and MACD signals using majority voting",
                Method = AggregatedStrategyProfile.AggregationMethod.MajorityVote,
                MinConfidence = 0.7,
                RiskLevel = 0.5
            };
            
            // Add the existing strategies to the aggregated one
            aggregatedStrategy.AddStrategy(rsiStrategy, 1.0);
            aggregatedStrategy.AddStrategy(macdStrategy, 1.0);
            
            _profiles[aggregatedStrategy.Name] = aggregatedStrategy;

            var ichimokuStrategy = new IchimokuCloudStrategy();
            _profiles[ichimokuStrategy.Name] = ichimokuStrategy;
            
            var parabolicSarStrategy = new ParabolicSARStrategy();
            _profiles[parabolicSarStrategy.Name] = parabolicSarStrategy;
        }

        /// <summary>
        /// Deserialize a profile from JSON based on its type
        /// </summary>
        /// <param name="type">Type name of the profile</param>
        /// <param name="profileData">JSON data for the profile</param>
        /// <returns>The deserialized profile, or null if type is not supported</returns>
        private TradingStrategyProfile DeserializeProfile(string type, JObject profileData)
        {
            TradingStrategyProfile profile = null;

            // Create the appropriate type based on the profile type
            switch (type)
            {
                case nameof(RsiDivergenceStrategy):
                    profile = profileData.ToObject<RsiDivergenceStrategy>();
                    break;
                case nameof(SmaCrossoverStrategy):
                    profile = profileData.ToObject<Quantra.Models.SmaCrossoverStrategy>();
                    break;
                case nameof(BollingerBandsMeanReversionStrategy):
                    profile = profileData.ToObject<BollingerBandsMeanReversionStrategy>();
                    break;
                case nameof(MacdCrossoverStrategy):
                    profile = profileData.ToObject<MacdCrossoverStrategy>();
                    break;
                case nameof(SupportResistanceStrategy):
                    profile = profileData.ToObject<SupportResistanceStrategy>();
                    break;
                case nameof(AggregatedStrategyProfile):
                    profile = DeserializeAggregatedStrategy(profileData);
                    break;
                case nameof(IchimokuCloudStrategy):
                    profile = profileData.ToObject<IchimokuCloudStrategy>();
                    break;
                case nameof(ParabolicSARStrategy):
                    profile = profileData.ToObject<ParabolicSARStrategy>();
                    break;
                default:
                    // Try to instantiate unknown types using reflection
                    try
                    {
                        var typeObj = Type.GetType($"Quantra.Models.{type}");
                        if (typeObj != null)
                        {
                            profile = profileData.ToObject(typeObj) as TradingStrategyProfile;
                        }
                    }
                    catch
                    {
                        // Silently fail and return null
                    }
                    break;
            }

            return profile;
        }
        
        /// <summary>
        /// Special deserializer for aggregated strategy profiles
        /// </summary>
        /// <param name="profileData">JSON data for the aggregated profile</param>
        /// <returns>The deserialized aggregated strategy profile</returns>
        private TradingStrategyProfile DeserializeAggregatedStrategy(JObject profileData)
        {
            var aggregatedProfile = new AggregatedStrategyProfile();
            
            // Deserialize basic properties
            aggregatedProfile.Name = profileData["Name"]?.ToString();
            aggregatedProfile.Description = profileData["Description"]?.ToString();
            
            if (profileData["MinConfidence"] != null)
                aggregatedProfile.MinConfidence = profileData["MinConfidence"].Value<double>();
            
            if (profileData["RiskLevel"] != null)
                aggregatedProfile.RiskLevel = profileData["RiskLevel"].Value<double>();
                
            if (profileData["IsEnabled"] != null)
                aggregatedProfile.IsEnabled = profileData["IsEnabled"].Value<bool>();
                
            if (profileData["Method"] != null)
                aggregatedProfile.Method = (AggregatedStrategyProfile.AggregationMethod)profileData["Method"].Value<int>();
                
            if (profileData["ConsensusThreshold"] != null)
                aggregatedProfile.ConsensusThreshold = profileData["ConsensusThreshold"].Value<double>();
                
            // Deserialize strategies array if it exists
            if (profileData["Strategies"] is JArray strategiesArray)
            {
                foreach (JObject strategyObj in strategiesArray)
                {
                    string strategyName = strategyObj["StrategyName"]?.ToString();
                    double weight = strategyObj["Weight"]?.Value<double>() ?? 1.0;
                    
                    if (!string.IsNullOrEmpty(strategyName) && _profiles.TryGetValue(strategyName, out TradingStrategyProfile strategy))
                    {
                        aggregatedProfile.AddStrategy(strategy, weight);
                    }
                }
            }
            
            return aggregatedProfile;
        }
    }
}
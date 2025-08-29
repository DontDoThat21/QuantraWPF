using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    /// <summary>
    /// Interface for settings service (profile CRUD and ensure methods).
    /// </summary>
    public interface ISettingsService
    {
        void EnsureSettingsProfilesTable();
        int CreateSettingsProfile(DatabaseSettingsProfile profile);
        bool UpdateSettingsProfile(DatabaseSettingsProfile profile);
        bool DeleteSettingsProfile(int profileId);
        DatabaseSettingsProfile GetSettingsProfile(int profileId);
        DatabaseSettingsProfile GetDefaultSettingsProfile();
        Task<DatabaseSettingsProfile> GetDefaultSettingsProfileAsync();
        List<DatabaseSettingsProfile> GetAllSettingsProfiles();
        void EnsureSettingsProfiles();
        bool SetProfileAsDefault(int profileId);
    }
}

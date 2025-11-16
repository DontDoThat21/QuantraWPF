using System.Collections.Generic;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public interface IUserSettingsService
    {
        void SaveUserSettings(UserSettings settings);
        UserSettings GetUserSettings();
        string GetUserPreference(string key, string defaultValue = null);
        void SaveUserPreference(string key, string value);
        Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts();
        void RememberAccount(string username, string password, string pin);
        (string type, string id) GetActiveBenchmark();
        void SetActiveBenchmark(string benchmarkType);
        void SetActiveCustomBenchmark(string customBenchmarkId);
        void SaveWindowState(System.Windows.WindowState windowState);
        System.Windows.WindowState? GetSavedWindowState();
        
        // DataGrid configuration methods
        void SaveDataGridConfig(string tabName, string controlName, DataGridSettings settings);
        DataGridSettings LoadDataGridConfig(string tabName, string controlName);
    }
}

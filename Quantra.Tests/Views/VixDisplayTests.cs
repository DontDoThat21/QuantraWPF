using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.Tests.Views
{
    [TestClass]
    public class VixDisplayTests
    {
        private string testDbPath = "test_vix_display.db";

        [TestInitialize]  
        public void Setup()
        {
            // Clean up any existing test database
            if (File.Exists(testDbPath))
            {
                File.Delete(testDbPath);
            }
            
            // Initialize the database for testing
            DatabaseMonolith.Initialize();
        }

        [TestCleanup]
        public void Cleanup()
        {
            // Clean up test database
            if (File.Exists(testDbPath))
            {
                File.Delete(testDbPath);
            }
        }

        [TestMethod]
        public void GetDefaultSettingsProfile_ShouldEnableVixMonitoringByDefault()
        {
            // Arrange - ensure settings profiles are initialized
            SettingsService.EnsureSettingsProfiles();

            // Act - get the default settings profile
            var profile = SettingsService.GetDefaultSettingsProfile();

            // Assert
            Assert.IsNotNull(profile, "Default settings profile should not be null");
            Assert.IsTrue(profile.EnableVixMonitoring, "VIX monitoring should be enabled by default");
            //DatabaseMonolith.Log("Info", $"Test: VIX monitoring enabled = {profile.EnableVixMonitoring}");
        }

        [TestMethod]
        public void EnsureSettingsProfiles_ShouldCreateDefaultProfileWithVixEnabled()
        {
            // Act - ensure settings profiles exist
            SettingsService.EnsureSettingsProfiles();
            
            // Get the default profile
            var profile = SettingsService.GetDefaultSettingsProfile();

            // Assert
            Assert.IsNotNull(profile, "Should create a default profile");
            Assert.IsTrue(profile.IsDefault, "Profile should be marked as default");
            Assert.IsTrue(profile.EnableVixMonitoring, "Default profile should have VIX monitoring enabled");
            Assert.AreEqual("Default", profile.Name, "Default profile should have correct name");
        }

        [TestMethod]
        public void CreateCustomProfile_WithVixDisabled_ShouldWork()
        {
            // Arrange - ensure settings profiles are initialized
            SettingsService.EnsureSettingsProfiles();
            
            var customProfile = new DatabaseSettingsProfile
            {
                Name = "Custom Test Profile",
                Description = "Test profile with VIX disabled", 
                IsDefault = false,
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                EnableVixMonitoring = false // Explicitly disable VIX monitoring
            };

            // Act - create the profile
            int profileId = SettingsService.CreateSettingsProfile(customProfile);
            var retrievedProfile = SettingsService.GetSettingsProfile(profileId);

            // Assert
            Assert.IsNotNull(retrievedProfile, "Custom profile should be created and retrievable");
            Assert.IsFalse(retrievedProfile.EnableVixMonitoring, "Custom profile should have VIX monitoring disabled");
            Assert.AreEqual("Custom Test Profile", retrievedProfile.Name, "Profile name should match");
        }

        [TestMethod]
        public void VixDisplayLogic_ShouldReturnDisabledWhenSettingsDisableIt()
        {
            // Arrange - create a profile with VIX disabled
            var disabledProfile = new DatabaseSettingsProfile
            {
                Name = "VIX Disabled Profile",
                Description = "Profile with VIX monitoring disabled",
                IsDefault = true,
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                EnableVixMonitoring = false
            };

            // Clean existing profiles and create our test profile
            SettingsService.EnsureSettingsProfiles();
            int profileId = SettingsService.CreateSettingsProfile(disabledProfile);
            SettingsService.SetProfileAsDefault(profileId);

            // Act - get the profile and check VIX monitoring setting
            var profile = SettingsService.GetDefaultSettingsProfile();

            // Assert
            Assert.IsNotNull(profile, "Profile should exist");
            Assert.IsFalse(profile.EnableVixMonitoring, "VIX monitoring should be disabled");
            
            // This simulates what RefreshVixDisplay() checks
            bool shouldShowVix = profile?.EnableVixMonitoring == true;
            Assert.IsFalse(shouldShowVix, "VIX display should be disabled when settings disable it");
        }

        [TestMethod] 
        public void VixDisplayLogic_ShouldReturnEnabledWhenSettingsEnableIt()
        {
            // Arrange - ensure default profile exists (which should have VIX enabled)
            SettingsService.EnsureSettingsProfiles();

            // Act - get the profile and check VIX monitoring setting
            var profile = SettingsService.GetDefaultSettingsProfile();

            // Assert
            Assert.IsNotNull(profile, "Profile should exist");
            Assert.IsTrue(profile.EnableVixMonitoring, "VIX monitoring should be enabled by default");
            
            // This simulates what RefreshVixDisplay() checks
            bool shouldShowVix = profile?.EnableVixMonitoring == true;
            Assert.IsTrue(shouldShowVix, "VIX display should be enabled when settings enable it");
        }
    }
}

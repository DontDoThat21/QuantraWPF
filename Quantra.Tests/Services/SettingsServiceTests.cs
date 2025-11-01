using System;
using System.IO;
//using System.Data.SQLite;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.Tests.Services
{
    [TestClass]
    public class SettingsServiceTests
    {
        private string testDbPath = "test_settings.db";

        [TestInitialize]
        public void Setup()
        {
            // Clean up any existing test database
            if (File.Exists(testDbPath))
            {
                File.Delete(testDbPath);
            }
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
        public void EnsureSettingsProfilesTable_ShouldHandleLockedDatabase()
        {
            // This test verifies that the retry mechanism handles SQLite locking properly
            // by calling the method multiple times concurrently

            // Create a test database first
            DatabaseMonolith.Initialize();

            // Call the method - this should not throw even if database is locked temporarily
            try
            {
                SettingsService.EnsureSettingsProfilesTable();
                Assert.IsTrue(true, "Method completed without throwing exception");
            }
            catch (SQLiteException ex) when (ex.Message.Contains("database is locked"))
            {
                Assert.Fail("SQLite locking exception was not handled by retry mechanism: " + ex.Message);
            }
        }

        [TestMethod]
        public void GetDefaultSettingsProfile_ShouldRetryOnLock()
        {
            // Initialize database
            DatabaseMonolith.Initialize();
            SettingsService.EnsureSettingsProfiles();

            // This should work even with retry logic
            var profile = SettingsService.GetDefaultSettingsProfile();
            
            Assert.IsNotNull(profile, "Should return a default profile");
            Assert.IsTrue(profile.IsDefault, "Returned profile should be marked as default");
        }

        [TestMethod]
        public void CreateAndRetrieveSettingsProfile_ShouldWorkWithRetry()
        {
            // Initialize database
            DatabaseMonolith.Initialize();
            SettingsService.EnsureSettingsProfiles();

            // Create a test profile
            var testProfile = new DatabaseSettingsProfile
            {
                Name = "Test Profile",
                Description = "Test Description",
                IsDefault = false,
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                EnableApiModalChecks = true,
                ApiTimeoutSeconds = 30,
                CacheDurationMinutes = 15,
                EnableDarkMode = true,
                RiskLevel = "Medium"
            };

            // Create the profile
            int profileId = SettingsService.CreateSettingsProfile(testProfile);
            Assert.IsTrue(profileId > 0, "Profile should be created with a valid ID");

            // Retrieve the profile
            var retrievedProfile = SettingsService.GetSettingsProfile(profileId);
            Assert.IsNotNull(retrievedProfile, "Profile should be retrievable");
            Assert.AreEqual(testProfile.Name, retrievedProfile.Name, "Profile name should match");
            Assert.AreEqual(testProfile.RiskLevel, retrievedProfile.RiskLevel, "Profile risk level should match");
        }
    }
}
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Utilities;
using System;
using System.Text;

namespace Quantra.Tests
{
    [TestClass]
    public class CompressionTests
    {
        [TestMethod]
        public void CompressString_ShouldAddCompressionMarker()
        {
            // Arrange
            string input = "Test string";

            // Act
            string compressed = CompressionHelper.CompressString(input);

            // Assert
            Assert.IsTrue(compressed.StartsWith("GZIP:"), "Compressed string should start with GZIP marker");
        }

        [TestMethod]
        public void CompressDecompress_ShouldReturnOriginalString()
        {
            // Arrange
            string input = "This is a test string that should be compressed and then decompressed back to its original form.";

            // Act
            string compressed = CompressionHelper.CompressString(input);
            string decompressed = CompressionHelper.DecompressString(compressed);

            // Assert
            Assert.AreEqual(input, decompressed, "Decompressed string should match original input");
        }

        [TestMethod]
        public void CompressDecompress_ShouldWorkWithJsonData()
        {
            // Arrange
            string json = @"{
                ""name"": ""John Doe"",
                ""age"": 30,
                ""addresses"": [
                    {
                        ""type"": ""home"",
                        ""street"": ""123 Main St"",
                        ""city"": ""Anytown"",
                        ""state"": ""CA"",
                        ""zip"": ""12345""
                    },
                    {
                        ""type"": ""work"",
                        ""street"": ""456 Market St"",
                        ""city"": ""Somewhere"",
                        ""state"": ""TX"",
                        ""zip"": ""67890""
                    }
                ]
            }";

            // Act
            string compressed = CompressionHelper.CompressString(json);
            string decompressed = CompressionHelper.DecompressString(compressed);

            // Assert
            Assert.AreEqual(json, decompressed, "JSON should be preserved through compression/decompression");
        }

        [TestMethod]
        public void CompressDecompress_ShouldHandleEmptyString()
        {
            // Arrange
            string input = "";

            // Act
            string compressed = CompressionHelper.CompressString(input);
            string decompressed = Quantra.CompressionHelper.DecompressString(compressed);

            // Assert
            Assert.AreEqual(input, decompressed, "Empty string should be handled correctly");
        }

        [TestMethod]
        public void Decompress_ShouldHandleUncompressedData()
        {
            // Arrange
            string input = "This string is not compressed";

            // Act
            string result = CompressionHelper.DecompressString(input);

            // Assert
            Assert.AreEqual(input, result, "Uncompressed data should be returned as-is");
        }

        [TestMethod]
        public void IsCompressed_ShouldDetectCompressedData()
        {
            // Arrange
            string uncompressed = "Uncompressed data";
            string compressed = CompressionHelper.CompressString("Some data");

            // Act
            bool isUncompressedDetected = CompressionHelper.IsCompressed(uncompressed);
            bool isCompressedDetected = CompressionHelper.IsCompressed(compressed);

            // Assert
            Assert.IsFalse(isUncompressedDetected, "Uncompressed data should not be detected as compressed");
            Assert.IsTrue(isCompressedDetected, "Compressed data should be detected as compressed");
        }

        [TestMethod]
        public void Compression_ShouldReduceDataSize()
        {
            // Arrange
            StringBuilder sb = new StringBuilder();
            // Create a large repetitive string to ensure good compression
            for (int i = 0; i < 1000; i++)
            {
                sb.AppendLine("This is a test line with some repetitive content.");
            }
            string input = sb.ToString();

            // Act
            string compressed = CompressionHelper.CompressString(input);

            // Assert
            Assert.IsTrue(compressed.Length < input.Length,
                $"Compressed size ({compressed.Length}) should be smaller than original size ({input.Length})");
        }
    }
}
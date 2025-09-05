using System;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace Quantra.Utilities
{
    /// <summary>
    /// Helper class for compression and decompression operations
    /// </summary>
    public static class CompressionHelper
    {
        // Marker to identify compressed data
        private const string CompressionMarker = "GZIP:";

        /// <summary>
        /// Compresses a string using GZip compression
        /// </summary>
        /// <param name="input">String to compress</param>
        /// <returns>Compressed string as Base64 with a marker prefix</returns>
        public static string CompressString(string input)
        {
            if (string.IsNullOrEmpty(input))
                return input;

            byte[] inputBytes = Encoding.UTF8.GetBytes(input);

            using (var memoryStream = new MemoryStream())
            {
                using (var gzipStream = new GZipStream(memoryStream, CompressionMode.Compress))
                {
                    gzipStream.Write(inputBytes, 0, inputBytes.Length);
                }

                byte[] compressedBytes = memoryStream.ToArray();
                return CompressionMarker + Convert.ToBase64String(compressedBytes);
            }
        }

        /// <summary>
        /// Decompresses a string that was compressed with CompressString
        /// </summary>
        /// <param name="compressedInput">The compressed string in Base64 format with marker</param>
        /// <returns>The original uncompressed string</returns>
        public static string DecompressString(string compressedInput)
        {
            if (string.IsNullOrEmpty(compressedInput))
                return compressedInput;

            if (!compressedInput.StartsWith(CompressionMarker))
                return compressedInput; // Not compressed with our scheme

            string base64 = compressedInput.Substring(CompressionMarker.Length);
            byte[] compressedBytes = Convert.FromBase64String(base64);

            using (var memoryStream = new MemoryStream(compressedBytes))
            using (var resultStream = new MemoryStream())
            using (var gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress))
            {
                gzipStream.CopyTo(resultStream);
                return Encoding.UTF8.GetString(resultStream.ToArray());
            }
        }

        /// <summary>
        /// Checks if a string has been compressed with our compression method
        /// </summary>
        /// <param name="data">The string to check</param>
        /// <returns>True if the string is compressed</returns>
        public static bool IsCompressed(string data)
        {
            return !string.IsNullOrEmpty(data) && data.StartsWith(CompressionMarker);
        }
    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using AroopaApi.Encoders;

namespace UnitTestsProject.Encoders
{
    [TestClass]
    public class lambdaEncoderTests
    {
        lambdaEncoder lambdaEncoder = new lambdaEncoder(); // Instance of lambdaEncoder for testing

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_SingleCharacter()
        {
            // Arrange
            string input = "A"; // Single character input
            List<char> charset = new List<char> { 'A' }; // Character set containing 'A'

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset); // Encode the input
            string decodedText = lambdaEncoder.Decode(encodedData, charset); // Decode the encoded data

            // Assert
            Assert.AreEqual(input, decodedText); // Check if decoding returns the original input
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_MultipleCharacters()
        {
            // Arrange
            string input = "Hello"; // Input string
            List<char> charset = new List<char> { 'H', 'e', 'l', 'o' }; // Character set containing 'H', 'e', 'l', 'o'

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset); // Encode the input
            string decodedText = lambdaEncoder.Decode(encodedData, charset); // Decode the encoded data

            // Assert
            Assert.AreEqual(input, decodedText); // Check if decoding returns the original input
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_EmptyInput()
        {
            // Arrange
            string input = ""; // Empty input string
            List<char> charset = new List<char> { 'H', 'e', 'l', 'o' }; // Character set containing 'H', 'e', 'l', 'o'

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset); // Encode the input
            string decodedText = lambdaEncoder.Decode(encodedData, charset); // Decode the encoded data

            // Assert
            Assert.AreEqual(input, decodedText); // Check if decoding returns the original input
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_UnicodeCharacters()
        {
            // Arrange
            string input = "你好"; // Input string with Unicode characters
            List<char> charset = new List<char> { '你', '好' }; // Character set containing Unicode characters '你' and '好'

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset); // Encode the input
            string decodedText = lambdaEncoder.Decode(encodedData, charset); // Decode the encoded data

            // Assert
            Assert.AreEqual(input, decodedText); // Check if decoding returns the original input
        }
    }
}

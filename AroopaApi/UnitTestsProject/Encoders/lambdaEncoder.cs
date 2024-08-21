using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using static System.Net.Mime.MediaTypeNames;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System.Text;

using AroopaApi.Encoders;

namespace UnitTestsProject.Encoders
{
    [TestClass]
    public class lambdaEncoderTests
    {
        lambdaEncoder lambdaEncoder = new lambdaEncoder();

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_SingleCharacter()
        {
            // Arrange
            string input = "A";
            List<char> charset = new List<char> { 'A' };

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset);
            string decodedText = lambdaEncoder.Decode(encodedData, charset);

            // Assert
            Assert.AreEqual(input, decodedText);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_MultipleCharacters()
        {
            // Arrange
            string input = "Hello";
            List<char> charset = new List<char> { 'H', 'e', 'l', 'o' };

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset);
            string decodedText = lambdaEncoder.Decode(encodedData, charset);

            // Assert
            Assert.AreEqual(input, decodedText);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_EmptyInput()
        {
            // Arrange
            string input = "";
            List<char> charset = new List<char> { 'H', 'e', 'l', 'o' };

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset);
            string decodedText = lambdaEncoder.Decode(encodedData, charset);

            // Assert
            Assert.AreEqual(input, decodedText);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Encode_Decode_UnicodeCharacters()
        {
            // Arrange
            string input = "你好";
            List<char> charset = new List<char> { '你', '好' };

            // Act
            List<int> encodedData = lambdaEncoder.Encode(input, charset);
            string decodedText = lambdaEncoder.Decode(encodedData, charset);

            // Assert
            Assert.AreEqual(input, decodedText);
        }
    }
}
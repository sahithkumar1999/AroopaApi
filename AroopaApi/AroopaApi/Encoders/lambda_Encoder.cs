using System;
using System.Collections.Generic;
using System.Linq;

namespace AroopaApi.Encoders
{
    public class lambda_Encoder
    {
        private string chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\u007f";

        public List<int> Encode(string Data)
        {
            // Remove duplicate characters from chars
            string uniqueChars = new string(chars.Distinct().ToArray());

            // Create dictionary mapping characters to their indices
            Dictionary<char, int> stoi = uniqueChars.Select((ch, index) => new { ch, index }).ToDictionary(pair => pair.ch, pair => pair.index);

            // Encode function that converts a string to a list of integers
            Func<string, List<int>> encoder = s => s.Select(c => stoi[c]).ToList();

            // Encode the input Data using the encoder function
            List<int> encodedData = encoder(Data);

            return encodedData;
        }
    }
}

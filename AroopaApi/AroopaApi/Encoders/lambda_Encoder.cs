using System;
using System.Collections.Generic;
using System.Linq;

namespace AroopaApi.Encoders
{
    public class lambda_Encoder
    {
        /// <summary>
        /// Encodes a string into a list of integers based on the provided character set.
        /// </summary>
        /// <param name="Data">The input string to encode.</param>
        /// <param name="chars">The list of characters used for encoding.</param>
        /// <returns>The encoded data as a list of integers.</returns>
        public List<int> Encode(string Data, List<char> chars)
        {
            // Create a dictionary mapping characters to their indices in the chars list
            Dictionary<char, int> stoi = chars
                .Select((ch, index) => new { ch, index })
                .ToDictionary(pair => pair.ch, pair => pair.index);

            // Encoder function that converts a string to a list of integers
            Func<string, List<int>> encoder = s => s.Select(c => stoi[c]).ToList();

            // Encode the input Data using the encoder function
            List<int> encodedData = encoder(Data);

            return encodedData;
        }

        /// <summary>
        /// Decodes a list of integers into a string based on the provided character set.
        /// </summary>
        /// <param name="Data">The encoded data as a list of integers.</param>
        /// <param name="chars">The list of characters used for decoding.</param> 
        /// <returns>The decoded string.</returns>
        public string Decode(List<int> Data, List<char> chars)
        {
            // Create a dictionary mapping indices to characters in the chars list
            Dictionary<int, char> itos = chars
                .Select((ch, index) => new { index, ch })
                .ToDictionary(pair => pair.index, pair => pair.ch);

            // Decoder function that converts a list of integers to a string
            Func<List<int>, string> decoder = l => string.Join("", l.Select(i => itos[i]));

            // Decode the input Data using the decoder function
            string decodedText = decoder(Data);

            return decodedText;
        }
    }
}

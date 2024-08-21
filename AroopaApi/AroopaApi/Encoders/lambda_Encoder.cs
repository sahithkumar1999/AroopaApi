using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace AroopaApi.Encoders
{
    public class lambda_Encoder
    {

        public List<int> Encode(string Data, List<char> chars)
        {
            Dictionary<char, int> stoi = chars.Select((ch, index) => new { ch, index }).ToDictionary(pair => pair.ch, pair => pair.index);

            Func<string, List<int>> encoder = s => s.Select(c => stoi[c]).ToList();

            List<int> encodedData = encoder(Data);

            return encodedData;
        }

        public string Decode(List<int> Data, List<char> chars)
        {
            Dictionary<int, char> itos = chars.Select((ch, index) => new { index, ch }).ToDictionary(pair => pair.index, pair => pair.ch);

            Func<List<int>, string> decoder = l => string.Join("", l.Select(i => itos[i]));

            string decodedText = decoder(Data);

            return decodedText;
        }
    }
}

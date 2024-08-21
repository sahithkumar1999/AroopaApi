using AroopaApi;
using AroopaApi.Encoders;
using System.Text;
using GPT;


namespace GPT
{
    public class Program
    {
        
        static void Main(string[] args)
        {
            GPTmethods GPTmethods = new GPTmethods();
            lambda_Encoder lambda_Encoder = new lambda_Encoder();

            string filePath = "D:\\Company Aroopa\\Aroopa\\AroopaApi\\GPT\\TrainingFiles\\shakespeare.txt";
            var text = GPTmethods.ReadTextFromFile(filePath);
            Console.WriteLine($"Length of dataset in Characters: {text.Length}");

            (List<char> chars,int vocabSize) = GPTmethods.UniqueChars(text);

            // Print sorted characters
            Console.WriteLine(string.Join("", chars));

            // Print vocabulary size
            Console.WriteLine(vocabSize);


            List<int> encodedData = lambda_Encoder.Encode("hii there", chars);

            Console.WriteLine("Encoded result:");
            Console.WriteLine(string.Join(", ", encodedData));

            string decodedText = lambda_Encoder.Decode(encodedData, chars);

            Console.WriteLine($"decodedText : {decodedText}");

        }
    }
}

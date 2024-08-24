using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AroopaApi.Encoders;
using AroopaApi.Models.NeuralNetworks;

namespace GPT
{
    public class basicGPT
    {
        
        private static Random rand = new Random(1337); // Set seed for reproducibility
        private static int batchSize = 4; // Number of independent sequences processed in parallel
        private static int blockSize = 8; // Maximum context length for predictions

        public static void trainGPT()
        {
            GPTmethods GPTmethods = new GPTmethods();
            lambdaEncoder lambdaEncoder = new lambdaEncoder();

            string filePath = "D:\\Company Aroopa\\Aroopa\\AroopaApi\\GPT\\TrainingFiles\\shakespeare.txt";
            var text = GPTmethods.ReadTextFromFile(filePath);
            Console.WriteLine($"Length of dataset in Characters: {text.Length}");

            (List<char> chars, int vocabSize) = GPTmethods.UniqueChars(text);

            // Print sorted characters
            Console.WriteLine(string.Join("", chars));

            // Print vocabulary size
            Console.WriteLine(vocabSize);


            List<int> encodedData = lambdaEncoder.Encode(text, chars);

            //Console.WriteLine("Encoded result:");
            //Console.WriteLine(string.Join(", ", encodedData));

            //string decodedText = lambdaEncoder.Decode(encodedData, chars);

            //Console.WriteLine($"decodedText : {decodedText}");


            int n = (int)(0.9 * encodedData.Count); // Calculate the index for splitting the data
            List<int> trainData = encodedData.GetRange(0, n);// Get the training data from index 0 to n-1
            List<int> valData = encodedData.GetRange(n, encodedData.Count - n); // Get the validation data from index n to end of list

            //Console.WriteLine("Training Data:");
            //Console.WriteLine(string.Join(", ", trainData));

            //Console.WriteLine("Validation Data:");
            //Console.WriteLine(string.Join(", ", valData));
            int blockSize = 8;
            List<int> slicedData = trainData.Take(blockSize + 1).ToList();


            Console.WriteLine("Sliced Data:");
            Console.WriteLine(string.Join(", ", slicedData));

            List<int> x = trainData.GetRange(0, blockSize);
            List<int> y = trainData.GetRange(0, blockSize + 1);

            for (int t = 0; t < blockSize; t++)
            {
                List<int> context = x.GetRange(0, t + 1);
                int target = y[t + 1];

                Console.WriteLine($"when input is [{string.Join(", ", context)}] the target: {target}");
            }

            //int[] trainData = GenerateData(100); // Example data generation for training
            //int[] valData = GenerateData(50); // Example data generation for validation

            int[][] xb, yb;
            GetBatch("train", trainData, out xb, out yb);

            Console.WriteLine("Inputs:");
            PrintBatch(xb);
            Console.WriteLine("\nTargets:");
            PrintBatch(yb);
            Console.WriteLine("\n----");

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < blockSize; t++)
                {
                    int[] context = new int[t + 1];
                    Array.Copy(xb[b], context, t + 1);
                    int target = yb[b][t];
                    Console.WriteLine($"When input is [{string.Join(", ", context)}], the target: {target}");
                }
            }
            // Initialize the BigramLanguageModel with the vocabulary size
            BigramLanguageModel bigramLanguageModel = new BigramLanguageModel(vocabSize);

            // Perform a forward pass by explicitly calling the Forward method
            var (logits, loss) = bigramLanguageModel.Forward(xb, yb);

            // Output the results
            Console.WriteLine($"Logits shape: {logits.shape}");
            Console.WriteLine($"Loss: {loss}");
        }


        private static void GetBatch(string split, List<int> data, out int[][] xb, out int[][] yb)
        {
            xb = new int[batchSize][];
            yb = new int[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                // Select a random starting index for xb[i]
                int startIndex = rand.Next(data.Count - blockSize);

                // Create xb[i] with blockSize elements starting from startIndex
                xb[i] = data.Skip(startIndex).Take(blockSize).ToArray();

                // Create yb[i] with blockSize elements starting from startIndex + 1
                yb[i] = data.Skip(startIndex + 1).Take(blockSize).ToArray();
            }
        }
        

        // Function to print batch data
        private static void PrintBatch(int[][] batch)
        {
            foreach (var item in batch)
            {
                Console.WriteLine($"[{string.Join(", ", item)}]");
            }
        }
    }
}

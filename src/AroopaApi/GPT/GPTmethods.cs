using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GPT
{
    public class GPTmethods
    {
        /// <summary>
        /// Reads the content of a text file given its file path.
        /// </summary>
        /// <param name="filePath">The path to the text file.</param>
        /// <returns>The content of the text file as a string.</returns>
        public string ReadTextFromFile(string filePath)
        {
            string text = string.Empty;

            try
            {
                // Open the file using StreamReader with UTF-8 encoding
                using (StreamReader sr = new StreamReader(filePath, System.Text.Encoding.UTF8))
                {
                    // Read the entire file into a string
                    text = sr.ReadToEnd();
                }
            }
            catch (Exception ex)
            {
                // Handle any exceptions that occur during file reading
                Console.WriteLine($"Error reading the file: {ex.Message}");
            }

            return text;
        }

        /// <summary>
        /// Extracts unique characters from a given text and calculates the vocabulary size.
        /// </summary>
        /// <param name="text">The input text from which to extract unique characters.</param>
        /// <returns>A tuple containing a list of unique characters and the vocabulary size.</returns>
        public (List<char>, int) UniqueChars(string text)
        {
            // Extract unique characters and order them alphabetically
            List<char> chars = text.Distinct().OrderBy(c => c).ToList();

            // Calculate the vocabulary size
            int vocabSize = chars.Count;

            // Return the list of unique characters and the vocabulary size
            return (chars, vocabSize);
        }



        

    }
}

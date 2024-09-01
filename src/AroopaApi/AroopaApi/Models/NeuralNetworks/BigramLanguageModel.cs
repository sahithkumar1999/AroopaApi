using AroopaApi.Interfaces;
using NumSharp;
using System;
using System.Linq;

namespace AroopaApi.Models.NeuralNetworks
{
    /// <summary>
    /// Bigram Language Model using token embeddings and softmax for predictions.
    /// </summary>
    public class BigramLanguageModel : IBigramLanguageModel
    {
        /// <summary>
        /// Embedding table for token representations.
        /// </summary>
        public Embedding TokenEmbeddingTable { get; private set; }

        /// <summary>
        /// Initializes the Bigram Language Model with the given vocabulary size.
        /// </summary>
        /// <param name="vocabSize">The size of the vocabulary (number of unique tokens).</param>
        public BigramLanguageModel(int vocabSize)
        {
            // Initialize the token embedding table with dimensions (vocabSize, vocabSize)
            TokenEmbeddingTable = new Embedding(vocabSize, vocabSize);
        }

        /// <summary>
        /// Performs forward propagation through the model to obtain logits.
        /// </summary>
        /// <param name="idx">Input indices for the tokens (shape: batch_size x sequence_length).</param>
        /// <param name="targets">Optional target indices for computing loss (shape: batch_size x sequence_length).</param>
        /// <returns>
        /// A tuple containing:
        /// - logits: The model's output logits (shape: batch_size x sequence_length x vocab_size).
        /// - targets: The reshaped target indices if provided (shape: batch_size x sequence_length).
        /// </returns>
        public (NDArray logits, NDArray targets) Forward(NDArray idx, NDArray targets = null)
        {
            // Compute logits from the token embedding table
            NDArray logits = TokenEmbeddingTable.Forward(idx); // Shape: (batch_size, sequence_length, vocab_size)

            if (targets == null)
            {
                return (logits, null);
            }

            // Flatten logits and targets for cross-entropy computation
            int batchSize = idx.shape[0];
            int sequenceLength = idx.shape[1];
            int vocabSize = logits.shape[2];

            // Reshape logits and targets for loss computation
            logits = logits.reshape(batchSize * sequenceLength, vocabSize);
            targets = targets.reshape(batchSize * sequenceLength);

            return (logits, targets);
        }

        /// <summary>
        /// Generates a sequence of tokens given an initial input.
        /// </summary>
        /// <param name="idx">Initial input indices for the sequence (shape: batch_size x sequence_length).</param>
        /// <param name="maxNewTokens">Maximum number of new tokens to generate.</param>
        /// <returns>The generated sequence of tokens (shape: batch_size x (sequence_length + maxNewTokens)).</returns>
        public NDArray Generate(NDArray idx, int maxNewTokens)
        {
            for (int _ = 0; _ < maxNewTokens; _++)
            {
                // Get the predictions (logits) from the model
                var (logits, _) = Forward(idx);
                // Focus only on the last time step
                logits = logits[":", -1, ":"];
                // Apply softmax to get probabilities
                NDArray probs = Softmax(logits);
                // Sample from the distribution to get the next token index
                NDArray idxNext = np.argmax(probs, axis: 1).reshape(idx.shape[0], 1);
                // Append the sampled index to the running sequence
                idx = np.concatenate(new[] { idx, idxNext }, axis: 1);
            }
            return idx;
        }

        /// <summary>
        /// Computes the softmax probabilities of the logits.
        /// </summary>
        /// <param name="logits">The logits to apply softmax to (shape: batch_size x vocab_size).</param>
        /// <returns>The computed softmax probabilities (shape: batch_size x vocab_size).</returns>
        public NDArray Softmax(NDArray logits)
        {
            var maxLogits = np.max(logits, axis: 1, keepdims: true);
            var expLogits = np.exp(logits - maxLogits);

            // Initialize sumExpLogits as zeros with the shape (batchSize, 1)
            var sumExpLogits = np.zeros(new Shape(expLogits.shape[0], 1));

            // Manually sum the exponentiated logits across the second axis (sequence length)
            for (int i = 0; i < expLogits.shape[0]; i++) // Iterate over batches
            {
                double sum = 0.0;
                for (int j = 0; j < expLogits.shape[1]; j++) // Iterate over the vocabulary size
                {
                    sum += expLogits[i, j];
                }
                sumExpLogits[i, 0] = sum;
            }

            // Perform the division to get the softmax probabilities
            return expLogits / sumExpLogits;
        }
    }
}

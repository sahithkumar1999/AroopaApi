using NumSharp;
using System;
using System.Linq;

namespace AroopaApi.Models.NeuralNetworks
{
    public class BigramLanguageModel
    {
        public Embedding TokenEmbeddingTable { get; private set; }

        public BigramLanguageModel(int vocabSize)
        {
            // Initialize the token embedding table
            TokenEmbeddingTable = new Embedding(vocabSize, vocabSize);
        }

        public (NDArray logits, float? loss) Forward(NDArray idx, NDArray targets = null)
        {
            // Compute logits
            NDArray logits = TokenEmbeddingTable.Forward(idx); // (B, T, C)

            if (targets == null)
            {
                return (logits, null);
            }

            // Flatten logits and targets for cross-entropy computation
            int batchSize = idx.shape[0];
            int sequenceLength = idx.shape[1];
            int vocabSize = logits.shape[2];

            logits = logits.reshape(batchSize * sequenceLength, vocabSize);
            targets = targets.reshape(batchSize * sequenceLength);

            // Compute cross-entropy loss
            float loss = ComputeCrossEntropyLoss(logits, targets);

            return (logits, loss);
        }

        private float ComputeCrossEntropyLoss(NDArray logits, NDArray targets)
        {
            int batchSize = logits.shape[0];
            int vocabSize = logits.shape[1];
            int[] targetIndices = targets.ToArray<int>();

            // Calculate the logits for the target classes
            NDArray targetLogits = np.empty(new Shape(batchSize));
            for (int i = 0; i < batchSize; i++)
            {
                targetLogits[i] = logits[i, targetIndices[i]];
            }

            // Apply softmax to logits (manual implementation)
            NDArray maxLogits = np.max(logits, axis: 1).reshape(batchSize, 1);
            NDArray expLogits = np.exp(logits - maxLogits);

            // Compute the softmax manually
            NDArray softmax = np.zeros(logits.shape);
            for (int i = 0; i < batchSize; i++)
            {
                float sumExp = 0;
                for (int j = 0; j < vocabSize; j++)
                {
                    sumExp += expLogits[i, j];
                }
                for (int j = 0; j < vocabSize; j++)
                {
                    softmax[i, j] = expLogits[i, j] / sumExp;
                }
            }

            // Compute the log probabilities
            NDArray logProbs = np.log(softmax);

            // Calculate log probabilities of the target classes
            NDArray targetLogProbs = np.empty(new Shape(batchSize));
            for (int i = 0; i < batchSize; i++)
            {
                targetLogProbs[i] = logProbs[i, targetIndices[i]];
            }

            // Compute the cross-entropy loss
            float loss = -0f;
            for (int i = 0; i < batchSize; i++)
            {
                loss += targetLogProbs[i];
            }
            loss /= batchSize;

            return loss;
        }











        public NDArray Generate(NDArray idx, int maxNewTokens)
        {
            for (int _ = 0; _ < maxNewTokens; _++)
            {
                // Get the predictions
                var (logits, _) = Forward(idx);
                // Focus only on the last time step
                logits = logits[":", -1, ":"];
                // Apply softmax to get probabilities
                NDArray probs = Softmax(logits);
                // Sample from the distribution
                NDArray idxNext = np.argmax(probs, axis: 1).reshape(idx.shape[0], 1);
                // Append sampled index to the running sequence
                idx = np.concatenate(new[] { idx, idxNext }, axis: 1);
            }
            return idx;
        }

        private NDArray Softmax(NDArray logits)
        {
            var maxLogits = np.max(logits, axis: 1, keepdims: true);
            var expLogits = np.exp(logits - maxLogits); // Numerical stability
            return expLogits / np.sum(expLogits, axis: 1, keepdims: true);
        }
    }
}

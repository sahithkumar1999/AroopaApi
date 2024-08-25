using NumSharp;
using System;
using System.Diagnostics;

namespace AroopaApi.Models.NeuralNetworks
{
    public class Embedding
    {
        public int num_embeddings;
        public int embedding_dim;
        public int? padding_idx;
        public float? max_norm;
        public float norm_type;
        public bool scale_grad_by_freq;
        public bool sparse;
        public NDArray weight;

        public Embedding(int num_embeddings, int embedding_dim, int? padding_idx = null,
                         float? max_norm = null, float norm_type = 2.0f,
                         bool scale_grad_by_freq = false, bool sparse = false,
                         NDArray weight = null)
        {
            this.num_embeddings = num_embeddings;
            this.embedding_dim = embedding_dim;
            this.padding_idx = padding_idx;
            this.max_norm = max_norm;
            this.norm_type = norm_type;
            this.scale_grad_by_freq = scale_grad_by_freq;
            this.sparse = sparse;

            // Initialize weight using NumSharp if weight parameter is not provided
            if (weight == null)
            {
                // Corrected this line to use integers directly
                this.weight = np.random.normal(0, 1, num_embeddings, embedding_dim);
                Debug.WriteLine($"Weight initialized with shape: {this.weight.shape}"); // Debugging line
                this.FillPaddingIndexWithZero(); // Ensure padding index is set to zero
            }
            else
            {
                // Validate provided weight shape matches num_embeddings and embedding_dim
                if (weight.shape[0] != num_embeddings || weight.shape[1] != embedding_dim)
                {
                    throw new ArgumentException("Shape of weight does not match num_embeddings and embedding_dim");
                }
                this.weight = weight.Clone(); // Clone the provided weight
                this.FillPaddingIndexWithZero(); // Ensure padding index is set to zero
            }

            // Debugging line to print the shape of the weight matrix
            Debug.WriteLine($"Final weight matrix shape: {this.weight.shape}");
        }

        private void FillPaddingIndexWithZero()
        {
            if (padding_idx != null)
            {
                // Create a zero vector with the correct shape (embedding_dim,)
                NDArray paddingVec = np.zeros(new Shape(embedding_dim));

                // Assign the zero vector to the specific row for the padding index
                this.weight[padding_idx.Value, ":"] = paddingVec; // Use ':' for slicing to select the entire row
            }
        }

        public NDArray Forward(NDArray input)
        {
            // Check if the input is 2D
            if (input.ndim != 2)
            {
                throw new ArgumentException("Input must be a 2D array of indices.");
            }

            int batchSize = input.shape[0];
            int sequenceLength = input.shape[1];
            int vocabSize = this.weight.shape[0];  // Number of rows in weight matrix
            int embeddingDim = this.weight.shape[1]; // Number of columns in weight matrix

            // Debugging lines to check dimensions
            Debug.WriteLine($"Forward called with input shape: {input.shape}");
            Debug.WriteLine($"Weight matrix shape: {this.weight.shape}");

            // Initialize the result NDArray
            NDArray result = np.zeros(new Shape(batchSize, sequenceLength, embeddingDim));

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    int idx = (int)input[i, j]; // Ensure input values are integers

                    // Add a debug statement to check the index and shapes
                    Debug.WriteLine($"Processing batch {i}, sequence {j}: idx = {idx}");

                    if (idx >= vocabSize || idx < 0)
                    {
                        throw new ArgumentException($"Index {idx} is out of bounds for weight matrix with shape {this.weight.shape}.");
                    }

                    result[i, j, ":"] = this.weight[idx];
                }
            }

            return result;
        }
    }
}

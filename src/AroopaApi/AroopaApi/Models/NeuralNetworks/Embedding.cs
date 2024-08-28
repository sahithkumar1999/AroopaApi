using NumSharp;
using System;
using System.Diagnostics;

namespace AroopaApi.Models.NeuralNetworks
{
    /// <summary>
    /// Represents an embedding layer that maps token indices to dense vectors.
    /// </summary>
    public class Embedding
    {
        // Number of embeddings (vocabulary size)
        public int num_embeddings;
        // Dimensionality of the embeddings
        public int embedding_dim;
        // Optional padding index
        public int? padding_idx;
        // Optional maximum norm for embeddings
        public float? max_norm;
        // Norm type for max_norm
        public float norm_type;
        // Flag to scale gradients by frequency
        public bool scale_grad_by_freq;
        // Flag to use sparse gradient updates
        public bool sparse;
        // Weight matrix holding the embeddings
        public NDArray weight;

        /// <summary>
        /// Initializes the Embedding layer with the specified parameters.
        /// </summary>
        /// <param name="num_embeddings">The size of the vocabulary (number of unique tokens).</param>
        /// <param name="embedding_dim">The dimensionality of the embeddings.</param>
        /// <param name="padding_idx">Optional index to be used for padding (if any).</param>
        /// <param name="max_norm">Optional maximum norm for embeddings.</param>
        /// <param name="norm_type">Norm type used for the maximum norm.</param>
        /// <param name="scale_grad_by_freq">Flag indicating whether to scale gradients by frequency.</param>
        /// <param name="sparse">Flag indicating whether to use sparse gradients.</param>
        /// <param name="weight">Optional pre-initialized weight matrix for embeddings.</param>
        /// <exception cref="ArgumentException">Thrown if the provided weight shape does not match the specified dimensions.</exception>
        public Embedding(int num_embeddings, int embedding_dim, int? padding_idx = null,
                         float? max_norm = null, float norm_type = 2.0f,
                         bool scale_grad_by_freq = false, bool sparse = false,
                         NDArray weight = null)
        {
            // Set the parameters for the embedding layer
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

        /// <summary>
        /// Sets the embedding vector for the padding index to zero.
        /// </summary>
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

        /// <summary>
        /// Retrieves the embeddings for the given input indices.
        /// </summary>
        /// <param name="input">2D array of indices where each element represents a token index.</param>
        /// <returns>A 3D array where each token index is replaced by its corresponding embedding vector.</returns>
        /// <exception cref="ArgumentException">Thrown if the input is not a 2D array or contains indices out of bounds.</exception>
        public NDArray Forward(NDArray input)
        {
            // Validate that the input is a 2D array
            if (input.ndim != 2)
            {
                throw new ArgumentException("Input must be a 2D array of indices.");
            }

            int batchSize = input.shape[0];
            int sequenceLength = input.shape[1];
            int vocabSize = this.weight.shape[0];  // Number of embeddings
            int embeddingDim = this.weight.shape[1]; // Dimensionality of each embedding

            // Debugging lines to check input and weight dimensions
            Debug.WriteLine($"Forward called with input shape: {input.shape}");
            Debug.WriteLine($"Weight matrix shape: {this.weight.shape}");

            // Initialize the result NDArray to hold the embeddings
            NDArray result = np.zeros(new Shape(batchSize, sequenceLength, embeddingDim));

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    int idx = (int)input[i, j]; // Ensure input values are integers

                    // Debugging statement to check the index and shapes
                    Debug.WriteLine($"Processing batch {i}, sequence {j}: idx = {idx}");

                    if (idx >= vocabSize || idx < 0)
                    {
                        throw new ArgumentException($"Index {idx} is out of bounds for weight matrix with shape {this.weight.shape}.");
                    }

                    // Retrieve the embedding for the given index and store it in the result
                    result[i, j, ":"] = this.weight[idx];
                }
            }

            return result;
        }
    }
}

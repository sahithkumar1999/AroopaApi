using System;
using System.Linq;
using AroopaApi.Interfaces;
using NumSharp;

namespace AroopaApi.Losses
{
    /// <summary>
    /// Provides methods to compute cross-entropy loss and softmax probabilities.
    /// </summary>
    public class CrossEntropyLoss : ICrossEntropyLoss
    {
        /// <summary>
        /// Computes the cross-entropy loss between predicted probabilities and target labels.
        /// </summary>
        /// <param name="input">2D array of predicted logits or probabilities (batch_size x num_classes).</param>
        /// <param name="target">1D array of target class indices (batch_size).</param>
        /// <param name="weight">Optional 1D array of class weights for adjusting loss based on class importance.</param>
        /// <param name="ignoreIndex">Index to be ignored in loss computation (e.g., for padding).</param>
        /// <param name="reduction">Type of reduction to apply to the loss: "mean" or "sum".</param>
        /// <param name="labelSmoothing">Amount of label smoothing to apply (0.0 means no smoothing).</param>
        /// <returns>The computed cross-entropy loss as a double.</returns>
        public double ComputeCrossEntropyLoss(
            NDArray input,
            NDArray target,
            double[] weight = null,
            int ignoreIndex = -100,
            string reduction = "mean",
            double labelSmoothing = 0.0)
        {
            // Number of classes (vocabulary size)
            int numClasses = input.shape[1];
            // Batch size (number of samples)
            int batchSize = input.shape[0];
            // Initialize loss and valid element counter
            double loss = 0.0;
            int validElements = 0;

            // Small epsilon value to prevent log(0) issues
            double epsilon = 1e-10;

            // Iterate over each sample in the batch
            for (int i = 0; i < batchSize; i++)
            {
                // Retrieve the target class index for the current sample
                int targetClass = (int)target[i];

                // Skip this sample if its target index is to be ignored
                if (targetClass == ignoreIndex)
                    continue;

                validElements++; // Count valid elements (excluding ignored ones)
                double logProb;

                // Access the predicted probability for the target class
                double trueClassProb = (double)input[i, targetClass];

                // Apply epsilon to avoid taking log of zero
                trueClassProb = Math.Max(trueClassProb, epsilon);

                // Apply label smoothing if specified
                if (labelSmoothing > 0.0)
                {
                    // Calculate smoothed probability
                    double smooth = labelSmoothing / numClasses;
                    double smoothedProb = trueClassProb * (1.0 - labelSmoothing) + smooth;
                    smoothedProb = Math.Max(smoothedProb, epsilon); // Apply epsilon to smoothed probability
                    logProb = Math.Log(smoothedProb);
                }
                else
                {
                    // Directly compute the log probability if no smoothing
                    logProb = Math.Log(trueClassProb);
                }

                // Compute the loss for this sample
                double sampleLoss = -logProb;

                // Apply class weight if provided
                if (weight != null)
                {
                    sampleLoss *= weight[targetClass];
                }

                // Accumulate the loss
                loss += sampleLoss;
            }

            // Apply reduction method if specified
            if (reduction == "mean" && validElements > 0)
            {
                // Compute the average loss if using mean reduction
                loss /= validElements;
            }

            return loss;
        }

        /// <summary>
        /// Computes the softmax probabilities from logits.
        /// </summary>
        /// <param name="logits">Array of logits (scores) from which to compute the probabilities.</param>
        /// <returns>An array of softmax probabilities.</returns>
        public double[] Softmax(double[] logits)
        {
            // Find the maximum logit value for numerical stability
            double maxLogit = logits.Max();
            // Compute exponentials of logits shifted by maxLogit
            double[] expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            // Compute the sum of exponentials
            double sumExpLogits = expLogits.Sum();
            // Compute the softmax probabilities by normalizing exponentials
            return expLogits.Select(exp => exp / sumExpLogits).ToArray();
        }
    }
}

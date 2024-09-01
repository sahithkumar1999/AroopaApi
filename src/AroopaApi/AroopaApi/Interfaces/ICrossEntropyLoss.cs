using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace AroopaApi.Interfaces
{
    public interface ICrossEntropyLoss
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
        double ComputeCrossEntropyLoss(
            NDArray input,
            NDArray target,
            double[] weight = null,
            int ignoreIndex = -100,
            string reduction = "mean",
            double labelSmoothing = 0.0);

        /// <summary>
        /// Computes the softmax probabilities from logits.
        /// </summary>
        /// <param name="logits">Array of logits (scores) from which to compute the probabilities.</param>
        /// <returns>An array of softmax probabilities.</returns>
        double[] Softmax(double[] logits);
    }
}

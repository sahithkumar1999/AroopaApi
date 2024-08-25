using System;
using System.Linq;
using NumSharp;

namespace AroopaApi.Losses
{
    public class CrossEntropyLoss
    {
        public static double ComputeCrossEntropyLoss(
            NDArray input,
            NDArray target,
            double[] weight = null,
            int ignoreIndex = -100,
            string reduction = "mean",
            double labelSmoothing = 0.0)
        {
            int numClasses = input.shape[1];
            int batchSize = input.shape[0];
            double loss = 0.0;
            int validElements = 0;

            // Small epsilon to prevent log(0) issues
            double epsilon = 1e-10;

            for (int i = 0; i < batchSize; i++)
            {
                int targetClass = (int)target[i]; // Access the target class index
                if (targetClass == ignoreIndex)
                    continue;

                validElements++;
                double logProb;

                // Access the probability for the target class
                double trueClassProb = (double)input[i, targetClass];

                // Apply epsilon to avoid log(0) issues
                trueClassProb = Math.Max(trueClassProb, epsilon);

                if (labelSmoothing > 0.0)
                {
                    double smooth = labelSmoothing / numClasses;
                    double smoothedProb = trueClassProb * (1.0 - labelSmoothing) + smooth;
                    smoothedProb = Math.Max(smoothedProb, epsilon); // Apply epsilon to smoothed probability
                    logProb = Math.Log(smoothedProb);
                }
                else
                {
                    logProb = Math.Log(trueClassProb);
                }

                double sampleLoss = -logProb;

                if (weight != null)
                {
                    sampleLoss *= weight[targetClass];
                }

                loss += sampleLoss;
            }

            if (reduction == "mean" && validElements > 0)
            {
                loss /= validElements;
            }

            return loss;
        }

        // Method to compute the softmax of input logits
        public static double[] Softmax(double[] logits)
        {
            double maxLogit = logits.Max();
            double[] expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            double sumExpLogits = expLogits.Sum();
            return expLogits.Select(exp => exp / sumExpLogits).ToArray();
        }
    }
}

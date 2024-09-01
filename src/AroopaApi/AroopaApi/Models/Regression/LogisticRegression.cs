using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AroopaApi.Models.Regression
{
    /// <summary>
    /// Logistic Regression implementation from scratch in C#.
    /// Supports basic binary classification with L2 regularization.
    /// </summary>
    public class LogisticRegression
    {
        public double[] Coefficients { get; private set; } // Model coefficients
        public double Intercept { get; private set; } // Model intercept
        public double LearningRate { get; set; } // Learning rate for gradient descent
        public int MaxIterations { get; set; } // Maximum number of iterations
        public double Regularization { get; set; } // Regularization strength (C)

        // Constructor with optional parameters
        public LogisticRegression(double learningRate = 0.01, int maxIterations = 1000, double regularization = 1.0)
        {
            LearningRate = learningRate;
            MaxIterations = maxIterations;
            Regularization = regularization;
        }

        /// <summary>
        /// Fits the logistic regression model using gradient descent.
        /// </summary>
        /// <param name="X">2D array of input features (n_samples x n_features)</param>
        /// <param name="y">Array of target values (n_samples), should be 0 or 1</param>
        public void Fit(double[,] X, double[] y)
        {
            int nSamples = X.GetLength(0);
            int nFeatures = X.GetLength(1);

            // Initialize coefficients and intercept
            Coefficients = new double[nFeatures];
            Intercept = 0.0;

            // Gradient descent
            for (int iter = 0; iter < MaxIterations; iter++)
            {
                // Compute gradients
                double[] gradients = new double[nFeatures];
                double interceptGradient = 0.0;

                for (int i = 0; i < nSamples; i++)
                {
                    double linearModel = Intercept;
                    for (int j = 0; j < nFeatures; j++)
                    {
                        linearModel += X[i, j] * Coefficients[j];
                    }
                    double prediction = Sigmoid(linearModel);
                    double error = prediction - y[i];

                    interceptGradient += error;
                    for (int j = 0; j < nFeatures; j++)
                    {
                        gradients[j] += error * X[i, j];
                    }
                }

                // Update coefficients and intercept with regularization
                for (int j = 0; j < nFeatures; j++)
                {
                    Coefficients[j] -= LearningRate * (gradients[j] / nSamples + (Regularization * Coefficients[j] / nSamples));
                }
                Intercept -= LearningRate * interceptGradient / nSamples;
            }
        }

        /// <summary>
        /// Predicts binary class labels for the input data.
        /// </summary>
        /// <param name="X">2D array of input features (n_samples x n_features)</param>
        /// <returns>Array of predicted class labels (0 or 1)</returns>
        public int[] Predict(double[,] X)
        {
            int nSamples = X.GetLength(0);
            int[] predictions = new int[nSamples];

            for (int i = 0; i < nSamples; i++)
            {
                double linearModel = Intercept;
                for (int j = 0; j < Coefficients.Length; j++)
                {
                    linearModel += X[i, j] * Coefficients[j];
                }
                predictions[i] = Sigmoid(linearModel) >= 0.5 ? 1 : 0;
            }

            return predictions;
        }

        /// <summary>
        /// Predicts probabilities of the positive class for the input data.
        /// </summary>
        /// <param name="X">2D array of input features (n_samples x n_features)</param>
        /// <returns>Array of predicted probabilities</returns>
        public double[] PredictProba(double[,] X)
        {
            int nSamples = X.GetLength(0);
            double[] probabilities = new double[nSamples];

            for (int i = 0; i < nSamples; i++)
            {
                double linearModel = Intercept;
                for (int j = 0; j < Coefficients.Length; j++)
                {
                    linearModel += X[i, j] * Coefficients[j];
                }
                probabilities[i] = Sigmoid(linearModel);
            }

            return probabilities;
        }

        /// <summary>
        /// Computes the sigmoid function.
        /// </summary>
        /// <param name="z">Input value</param>
        /// <returns>Sigmoid output</returns>
        private double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }
    }
}

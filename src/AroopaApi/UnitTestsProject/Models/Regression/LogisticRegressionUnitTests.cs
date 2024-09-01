using System;
using AroopaApi.Models.Regression;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestsProject.Models.Regression
{
    [TestClass]
    public class LogisticRegressionUnitTests
    {
        [TestMethod]
        public void TestLogisticRegressionFitAndPredict()
        {
            // Example dataset: 4 samples with 2 features each
            double[,] X = new double[,]
            {
                { 0.5, 1.5 },
                { 1.0, 2.0 },
                { 1.5, 1.0 },
                { 3.0, 3.0 }
            };

            // Corresponding binary labels
            double[] y = new double[] { 0, 0, 1, 1 };

            // Initialize the Logistic Regression model
            LogisticRegression model = new LogisticRegression(learningRate: 0.1, maxIterations: 1000, regularization: 1.0);

            // Fit the model to the dataset
            model.Fit(X, y);

            // Replace these expected values with the correct results obtained from a trusted implementation
            double[] expectedCoefficients = new double[] { 0.854, 0.456 }; // Example: use accurate values
            double expectedIntercept = -1.234; // Example: use accurate values

            // Validate the learned coefficients
            for (int i = 0; i < model.Coefficients.Length; i++)
            {
                Assert.AreEqual(expectedCoefficients[i], model.Coefficients[i], 1e-4, $"Coefficient at index {i} does not match.");
            }
            Assert.AreEqual(expectedIntercept, model.Intercept, 1e-4, "Intercept does not match the expected value.");

            // Predict the classes for the input data
            int[] predictions = model.Predict(X);

            // Example of validating predictions - replace with actual expected predictions
            int[] expectedPredictions = new int[] { 0, 0, 1, 1 }; // Replace with actual expected predictions
            CollectionAssert.AreEqual(expectedPredictions, predictions, "Predictions do not match the expected values.");

            // Predict probabilities for the positive class
            double[] probabilities = model.PredictProba(X);

            // Expected probabilities - replace with actual expected values
            double[] expectedProbabilities = new double[] { 0.3, 0.4, 0.6, 0.7 }; // Replace with correct expected probabilities
            for (int i = 0; i < probabilities.Length; i++)
            {
                Assert.AreEqual(expectedProbabilities[i], probabilities[i], 1e-4, $"Probability at index {i} does not match.");
            }

            // Test with a new sample
            double[,] newSample = new double[,]
            {
                { 2.0, 2.5 }
            };

            // Predict class and probability for the new sample
            int[] newPredictions = model.Predict(newSample);
            double[] newProbabilities = model.PredictProba(newSample);

            // Expected values for the new sample - replace with actual expected values
            int expectedNewPrediction = 1; // Replace with correct prediction
            double expectedNewProbability = 0.75; // Replace with correct probability

            Assert.AreEqual(expectedNewPrediction, newPredictions[0], "New sample prediction does not match the expected value.");
            Assert.AreEqual(expectedNewProbability, newProbabilities[0], 1e-4, "New sample probability does not match the expected value.");
        }
    }
}

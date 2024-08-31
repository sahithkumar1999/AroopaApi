using AroopaApi.Models.Regression;
using Microsoft.VisualStudio.TestTools.UnitTesting; // Ensure MSTest framework is referenced

namespace UnitTestsProject.Models.Regression
{
    [TestClass]
    public class LinearRegressionUnitTests
    {
        [TestMethod]
        [TestCategory("Prod")]
        public void TestLinearRegressionFitAndPredict()
        {
            // Define the input data
            double[,] X = new double[,]
            {
                {1, 1},
                {1, 2},
                {2, 2},
                {2, 3}
            };

            // Define the target values
            double[] y = new double[] { 6, 8, 9, 11 };

            // Initialize and fit the linear regression model
            LinearRegression lr = new LinearRegression();
            lr.Fit(X, y);

            // Predict the values using the model
            double[] predictions = lr.Predict(X);

            // Expected coefficients based on manual calculations or prior knowledge
            double[] expectedCoefficients = new double[] { 1.0, 2.0 };
            double expectedIntercept = 3.0;

            // Validate the coefficients using assertions
            CollectionAssert.AreEqual(expectedCoefficients, lr.coef_, "Coefficients do not match the expected values.");
            Assert.AreEqual(expectedIntercept, lr.intercept_, 1e-6, "Intercept does not match the expected value.");

            // Validate predictions using assertions (optional: tolerance for floating-point comparison)
            double[] expectedPredictions = new double[] { 6.0, 8.0, 9.0, 11.0 };
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.AreEqual(expectedPredictions[i], predictions[i], 1e-6, $"Prediction at index {i} does not match.");
            }
        }
    }
}

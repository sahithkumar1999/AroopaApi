using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AroopaApi.Brain;

namespace UnitTestsProject.Brain
{
    /// <summary>
    /// Unit tests for the <see cref="Neuron"/> class, which represents a basic artificial neuron.
    /// </summary>
    [TestClass]
    public class NeuronUnitTest
    {
        /// <summary>
        /// Tests if the <see cref="Neuron"/> class initializes with the correct number of weights based on the input size.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        public void Neuron_Initialization_CreatesCorrectNumberOfWeights()
        {
            int inputSize = 5;
            double bias = 0.5;
            Neuron neuron = new Neuron(inputSize, bias);

            // Access Weights and Bias through the neuron instance
            Assert.AreEqual(inputSize, neuron.Weights.Count, "Number of weights should match input size.");
            Assert.AreEqual(bias, neuron.Bias, "Bias should be correctly set during initialization.");
        }

        /// <summary>
        /// Tests if the <see cref="Neuron.ComputeOutput"/> method returns the expected output for specific inputs and weights.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        public void ComputeOutput_ReturnsExpectedOutput_ForGivenInputs()
        {
            int inputSize = 3;
            double bias = 1.0;
            Neuron neuron = new Neuron(inputSize, bias);

            // Set predefined weights for testing
            neuron.UpdateWeights(new List<double> { 0.5, -0.5, 1.0 });
            List<double> inputs = new List<double> { 1.0, 2.0, -1.0 };

            // Expected result calculations
            double expectedNetInput = 0.5 * 1.0 + (-0.5) * 2.0 + 1.0 * (-1.0) + bias;
            double expectedOutput = 1.0 / (1.0 + Math.Exp(-expectedNetInput)); // Sigmoid activation

            // Verify output
            double output = neuron.ComputeOutput(inputs);
            Assert.AreEqual(expectedOutput, output, 0.0001, "Output should match expected value.");
        }

        /// <summary>
        /// Tests if <see cref="Neuron.ComputeOutput"/> throws an exception when the number of inputs does not match the number of weights.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]

        public void ComputeOutput_ThrowsException_ForMismatchedInputSize()
        {
            Neuron neuron = new Neuron(3);
            List<double> inputs = new List<double> { 1.0, 2.0 }; // Only 2 inputs for 3 weights
            // Should throw an exception due to mismatched input size
            neuron.ComputeOutput(inputs);

        }

        /// <summary>
        /// Tests if <see cref="Neuron.UpdateWeights"/> correctly updates the neuron's weights to new values.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        public void UpdateWeights_SetsNewWeightsCorrectly()
        {
            Neuron neuron = new Neuron(3);
            List<double> newWeights = new List<double> { 0.3, 0.6, -0.2 };

            neuron.UpdateWeights(newWeights);

            // Verify if weights are updated correctly
            CollectionAssert.AreEqual(newWeights, neuron.Weights, "Weights should be updated to new values.");
        }

        /// <summary>
        /// Tests if <see cref="Neuron.UpdateWeights"/> throws an exception when the new weights list size does not match the current weights count.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        [ExpectedException(typeof(ArgumentException))]
        public void UpdateWeights_ThrowsException_ForMismatchedWeightSize()
        {
            Neuron neuron = new Neuron(3);
            List<double> newWeights = new List<double> { 0.3, 0.6 }; // Only 2 weights for 3 input neuron

            // Should throw an exception due to mismatched weight size
            neuron.UpdateWeights(newWeights);
        }
    }
}

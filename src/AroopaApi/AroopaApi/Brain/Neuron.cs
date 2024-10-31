//A neuron is an information-processing unit that is fundamental to the operation of a
//neural network. we identify three basic elements of the neural model:

//1. A set of synapses, or connecting links, each of which is characterized by a weight
// or strength of its own.Specifically, a signal xj at the input of synapse j connected to
// neuron k is multiplied by the synaptic weight wkj. It is important to make a note
// of the manner in which the subscripts of the synaptic weight kj are written.The
// first subscript in wkj refers to the neuron in question, and the second subscript
// refers to the input end of the synapse to which the weight refers.Unlike the weight
// of a synapse in the brain, the synaptic weight of an artificial neuron may lie in a
// range that includes negative as well as positive values.

// 2. Anadderfor summing the input signals, weighted by the respective synaptic strengths
// of the neuron; the operations described here constitute a linear combiner.

// 3. Anactivation function for limiting the amplitude of the output of a neuron.The ac
//tivation function is also referred to as a squashing function, in that it squashes
// (limits) the permissible amplitude range of the output signal to some finite value


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AroopaApi.Brain
{
    public class Neuron
    {
        // Fields
        public List<double> Weights;
        public double Bias;

        // Constructor
        public Neuron(int inputSize, double Bias = 0.0)
        {
            this.Weights = new List<double>();
            this.Bias = Bias;

            // Initialize weights with random values
            Random rand = new Random();
            for (int i = 0; i < inputSize; i++)
            {
                Weights.Add(rand.NextDouble() * 2 - 1); // Random weights between -1 and 1
            }
        }

        // Activation function (sigmoid in this case)
        private double ActivationFunction(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input)); // Sigmoid function
        }

        // Method to compute neuron output
        public double ComputeOutput(List<double> inputs)
        {
            if (inputs.Count != Weights.Count)
                throw new ArgumentOutOfRangeException("Number of inputs must match number of weights");


            // Linear combination of inputs and weights
            double netInput = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                netInput += inputs[i] * Weights[i];
            }

            // Adding bias
            netInput += Bias;

            // Applying activation function
            return ActivationFunction(netInput);
        }

        // Update weights (if you want to apply training logic)
        public void UpdateWeights(List<double> newWeights)
        {
            if (newWeights.Count != Weights.Count)
                throw new ArgumentException("Number of new weights must match current weights count");

            Weights = newWeights;
        }
    }
}
